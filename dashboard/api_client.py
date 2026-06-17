import requests

DEFAULT_TIMEOUT_SECONDS = 10


class DashboardApiError(RuntimeError):
    pass


def _api_url(api_base_url: str, path: str) -> str:
    return f"{api_base_url.rstrip('/')}{path}"


def get_station_data(api_base_url: str, timeout: int = 5) -> dict:
    response = requests.get(_api_url(api_base_url, "/stations"), timeout=timeout)
    if response.status_code != 200:
        raise DashboardApiError(f"API returned {response.status_code}: {response.text}")
    return response.json().get("stations", {})


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
