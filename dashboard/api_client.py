import requests

DEFAULT_TIMEOUT_SECONDS = 10
FORECAST_HORIZON = "model_artifact_horizon"
FORECAST_HORIZON_DESCRIPTION = (
    "Demo and live responses keep legacy next_hour keys, but should be interpreted "
    "as the model-defined forecast horizon."
)
DEMO_STATION_MAP = {
    "500101001": "捷運公館站 (大安區)",
    "500101002": "臺大資訊大樓 (大安區)",
    "500101003": "捷運市政府站 (信義區)",
    "500101004": "捷運西門站 (萬華區)",
    "500101005": "捷運士林站 (士林區)",
    "500101006": "捷運北投站 (北投區)",
}
DEMO_STATION_BIAS = {
    "500101001": -4,
    "500101002": -6,
    "500101003": 3,
    "500101004": 1,
    "500101005": 4,
    "500101006": 5,
}
DEMO_MODEL_LINEAGE = {
    "model_version": "dashboard-demo-fixtures-v1",
    "model_artifact_hash": "sha256:dashboard-demo",
    "model_metadata_loaded": False,
    "model_metadata_generated_at": None,
}


class DashboardApiError(RuntimeError):
    pass


def _api_url(api_base_url: str, path: str) -> str:
    return f"{api_base_url.rstrip('/')}{path}"


def get_station_data(api_base_url: str, timeout: int = 5) -> dict:
    response = requests.get(_api_url(api_base_url, "/stations"), timeout=timeout)
    if response.status_code != 200:
        raise DashboardApiError(f"API returned {response.status_code}: {response.text}")
    return response.json().get("stations", {})


def get_demo_station_data() -> dict:
    return DEMO_STATION_MAP.copy()


def predict_station(
    api_base_url: str,
    station_no: str,
    bikes_available: int,
    temperature: float,
    rain: float,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    payload = {
        "station_no": str(station_no),
        "bikes_available": int(bikes_available),
        "temperature": float(temperature),
        "rain": float(rain),
    }
    response = requests.post(
        _api_url(api_base_url, "/predict"),
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise DashboardApiError(f"API returned {response.status_code}: {response.text}")
    return response.json()


def rank_station_risks(
    api_base_url: str,
    stations: list[dict],
    temperature: float,
    rain: float,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict]:
    payload = {
        "temperature": float(temperature),
        "rain": float(rain),
        "stations": [
            {
                "station_no": str(station["station_no"]),
                "bikes_available": int(station["bikes_available"]),
                "spaces_available": int(station["spaces_available"]),
            }
            for station in stations
        ],
    }
    response = requests.post(
        _api_url(api_base_url, "/stations/risk"),
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise DashboardApiError(f"API returned {response.status_code}: {response.text}")
    return response.json().get("risks", [])


def demo_predict_station(
    station_no: str,
    bikes_available: int,
    temperature: float,
    rain: float,
) -> dict:
    rain_penalty = min(12, int(round(float(rain) * 0.8)))
    heat_penalty = 2 if temperature >= 34 else 0
    mild_weather_boost = 2 if 20 <= temperature <= 30 and rain == 0 else 0
    station_bias = DEMO_STATION_BIAS.get(str(station_no), 0)
    predicted = int(bikes_available) + station_bias + mild_weather_boost - rain_penalty - heat_penalty
    predicted = max(0, min(80, predicted))
    return {
        "station_no": str(station_no),
        "predicted_bikes_next_hour": predicted,
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_horizon_description": FORECAST_HORIZON_DESCRIPTION,
        **DEMO_MODEL_LINEAGE,
    }


def demo_rank_station_risks(
    stations: list[dict],
    temperature: float,
    rain: float,
) -> list[dict]:
    risks = []
    for station in stations:
        station_no = str(station["station_no"])
        current_bikes = int(station["bikes_available"])
        current_spaces = int(station["spaces_available"])
        predicted_bikes = demo_predict_station(
            station_no,
            current_bikes,
            temperature,
            rain,
        )["predicted_bikes_next_hour"]
        observed_capacity = current_bikes + current_spaces
        predicted_spaces = max(0, observed_capacity - predicted_bikes)
        risk_level, risk_score, suggested_action = classify_station_risk(
            predicted_bikes,
            predicted_spaces,
        )
        risks.append(
            {
                "station_no": station_no,
                "current_bikes_available": current_bikes,
                "current_spaces_available": current_spaces,
                "predicted_bikes_next_hour": predicted_bikes,
                "predicted_spaces_next_hour": predicted_spaces,
                "forecast_horizon": FORECAST_HORIZON,
                **DEMO_MODEL_LINEAGE,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "suggested_action": suggested_action,
            }
        )
    risks.sort(key=lambda item: (-item["risk_score"], item["station_no"]))
    return risks


def classify_station_risk(predicted_bikes: int, predicted_spaces: int) -> tuple[str, int, str]:
    if predicted_bikes <= 2:
        return "stock_out", 100 + (2 - predicted_bikes), "rebalance_in"
    if predicted_spaces <= 2:
        return "full_load", 90 + (2 - predicted_spaces), "rebalance_out"
    if predicted_bikes <= 5:
        return "low_supply", 50 + (5 - predicted_bikes), "monitor_supply"
    if predicted_spaces <= 5:
        return "low_dock", 40 + (5 - predicted_spaces), "monitor_docks"
    return "normal", 0, "monitor"


def station_display_options(station_map: dict) -> list[str]:
    return [f"{name} [{station_no}]" for station_no, name in sorted(station_map.items())]


def parse_station_option(option: str) -> tuple[str, str]:
    station_no = option.split("[")[-1].replace("]", "")
    station_name = option.split(" [")[0]
    return station_no, station_name


def risk_level_label(risk_level: str) -> str:
    labels = {
        "stock_out": "嚴重缺車",
        "full_load": "滿站風險",
        "low_supply": "車輛偏低",
        "low_dock": "空位偏低",
        "normal": "供需穩定",
    }
    return labels.get(risk_level, risk_level)


def suggested_action_label(action: str) -> str:
    labels = {
        "rebalance_in": "建議補車",
        "rebalance_out": "建議移出車輛",
        "monitor_supply": "觀察車輛水位",
        "monitor_docks": "觀察空位水位",
        "monitor": "維持監控",
    }
    return labels.get(action, action)
