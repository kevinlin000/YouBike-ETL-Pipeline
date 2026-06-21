import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import os
import hashlib
import json
import logging
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

logger = logging.getLogger(__name__)
MODEL_TIME_STEPS = 3
MODEL_FORECAST_HORIZON = "model_artifact_horizon"
MODEL_FORECAST_HORIZON_DESCRIPTION = (
    "Legacy response keys use next_hour naming, but current artifacts should be "
    "interpreted as the model-defined horizon. The documented local evaluation "
    "uses horizon_steps=1, meaning the next observation rather than a guaranteed "
    "one-hour forecast."
)
REQUEST_ID_HEADER = "X-Request-ID"
API_DEMO_MODE_ENV = "API_DEMO_MODE"
METRICS_PATH = "/metrics"
MODEL_METADATA_FILENAME = "model_metadata.json"
DEMO_MODEL_VERSION = "api-demo-fixtures-v1"
TRUTHY_VALUES = {"1", "true", "yes", "on"}
DEMO_STATION_FIXTURES = {
    "500101001": {"name": "捷運公館站 (大安區)", "capacity": 20, "station_bias": -1},
    "500101002": {"name": "臺大資訊大樓 (大安區)", "capacity": 20, "station_bias": -4},
    "500101003": {"name": "捷運市政府站 (信義區)", "capacity": 20, "station_bias": 3},
    "500101004": {"name": "捷運西門站 (萬華區)", "capacity": 20, "station_bias": 1},
    "500101005": {"name": "捷運士林站 (士林區)", "capacity": 20, "station_bias": 4},
}

# --- 1. 定義資料格式 ---
class RecentObservation(BaseModel):
    bikes_available: int
    temperature: float
    rain: float

    @field_validator("bikes_available")
    @classmethod
    def bikes_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("bikes_available 不可為負數")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_reasonable(cls, v: float) -> float:
        if not -50 <= v <= 60:
            raise ValueError("temperature 需介於 -50 與 60 之間")
        return v

    @field_validator("rain")
    @classmethod
    def rain_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("rain 不可為負數")
        return v

class PredictRequest(BaseModel):
    station_no: str
    bikes_available: int
    temperature: float
    rain: float
    recent_observations: list[RecentObservation] | None = None

    @field_validator("bikes_available")
    @classmethod
    def bikes_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("bikes_available 不可為負數")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_reasonable(cls, v: float) -> float:
        if not -50 <= v <= 60:
            raise ValueError("temperature 需介於 -50 與 60 之間")
        return v

    @field_validator("rain")
    @classmethod
    def rain_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("rain 不可為負數")
        return v

    @field_validator("recent_observations")
    @classmethod
    def recent_observations_match_model_window(
        cls,
        v: list[RecentObservation] | None,
    ) -> list[RecentObservation] | None:
        if v is not None and len(v) != MODEL_TIME_STEPS:
            raise ValueError(f"recent_observations 必須剛好包含 {MODEL_TIME_STEPS} 筆")
        return v

class PredictResponse(BaseModel):
    station_no: str
    predicted_bikes_next_hour: int
    forecast_horizon: str
    forecast_horizon_description: str
    model_version: str
    model_artifact_hash: str | None
    model_metadata_loaded: bool
    model_metadata_generated_at: str | None

class StationRiskInput(BaseModel):
    station_no: str
    bikes_available: int
    spaces_available: int
    recent_observations: list[RecentObservation] | None = None

    @field_validator("bikes_available", "spaces_available")
    @classmethod
    def availability_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("availability values 不可為負數")
        return v

    @field_validator("recent_observations")
    @classmethod
    def recent_observations_match_model_window(
        cls,
        v: list[RecentObservation] | None,
    ) -> list[RecentObservation] | None:
        if v is not None and len(v) != MODEL_TIME_STEPS:
            raise ValueError(f"recent_observations 必須剛好包含 {MODEL_TIME_STEPS} 筆")
        return v

class StationsRiskRequest(BaseModel):
    temperature: float
    rain: float
    stations: list[StationRiskInput]

    @field_validator("temperature")
    @classmethod
    def temperature_reasonable(cls, v: float) -> float:
        if not -50 <= v <= 60:
            raise ValueError("temperature 需介於 -50 與 60 之間")
        return v

    @field_validator("rain")
    @classmethod
    def rain_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("rain 不可為負數")
        return v

    @field_validator("stations")
    @classmethod
    def stations_non_empty(cls, v: list[StationRiskInput]) -> list[StationRiskInput]:
        if not v:
            raise ValueError("stations 不可為空")
        return v

class StationRiskResult(BaseModel):
    station_no: str
    current_bikes_available: int
    current_spaces_available: int
    predicted_bikes_next_hour: int
    predicted_spaces_next_hour: int
    forecast_horizon: str
    risk_level: str
    risk_score: int
    suggested_action: str

class StationsRiskResponse(BaseModel):
    forecast_horizon: str
    forecast_horizon_description: str
    model_version: str
    model_artifact_hash: str | None
    model_metadata_loaded: bool
    model_metadata_generated_at: str | None
    risks: list[StationRiskResult]

class StationsResponse(BaseModel):
    # 改為回傳字典：{ "station_no": "中文名稱 (行政區)", ... }
    stations: dict 

class HealthResponse(BaseModel):
    status: str
    demo_mode: bool
    model_loaded: bool
    scaler_loaded: bool
    station_mapping_loaded: bool
    station_catalog_loaded: bool
    warehouse_lookup_enabled: bool
    forecast_horizon: str
    model_version: str
    model_artifact_hash: str | None
    model_metadata_loaded: bool
    model_metadata_generated_at: str | None

class ReadinessResponse(HealthResponse):
    ready: bool

# --- 2. 全域變數 ---
model = None
scaler = None
station_mapping = None
station_info_map = None  # 新增：站點資訊對照表
db_engine: Engine | None = None
request_metrics: dict[tuple[str, str], dict[str, float | int]] = {}
request_metrics_lock = Lock()


def api_demo_mode_enabled() -> bool:
    return os.getenv(API_DEMO_MODE_ENV, "").strip().lower() in TRUTHY_VALUES


def short_sha256(data: bytes, length: int = 16) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()[:length]}"


def hash_json_payload(payload: dict) -> str:
    return short_sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


def unloaded_model_lineage() -> dict[str, str | bool | None]:
    return {
        "model_version": "unloaded",
        "model_artifact_hash": None,
        "model_metadata_loaded": False,
        "model_metadata_generated_at": None,
    }


def demo_model_lineage() -> dict[str, str | bool | None]:
    return {
        "model_version": DEMO_MODEL_VERSION,
        "model_artifact_hash": hash_json_payload(DEMO_STATION_FIXTURES),
        "model_metadata_loaded": False,
        "model_metadata_generated_at": None,
    }


model_lineage = unloaded_model_lineage()


def derive_model_version(
    metadata: dict | None,
    artifact_hash: str | None,
) -> str:
    hash_suffix = artifact_hash.split(":", maxsplit=1)[-1] if artifact_hash else "unknown"
    if metadata:
        model_type = metadata.get("model", {}).get("type", "model")
        horizon_steps = metadata.get("horizon_steps", "unknown")
        return f"{model_type}-h{horizon_steps}-{hash_suffix}"
    return f"legacy-artifact-{hash_suffix}"


def load_model_lineage(base_path: Path) -> dict[str, str | bool | None]:
    artifact_paths = {
        "model": base_path / "youbike_lstm_multistation.pth",
        "scaler": base_path / "scaler.pkl",
        "station_mapping": base_path / "station_mapping.pkl",
        "station_info_map": base_path / "station_info_map.pkl",
    }
    artifact_hashes = {
        name: hash_file(path)
        for name, path in artifact_paths.items()
        if path.exists()
    }
    artifact_hash = hash_json_payload(artifact_hashes) if artifact_hashes else None

    metadata_path = base_path / MODEL_METADATA_FILENAME
    metadata: dict | None = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Model metadata is not valid JSON: %s", exc)

    return {
        "model_version": derive_model_version(metadata, artifact_hash),
        "model_artifact_hash": artifact_hash,
        "model_metadata_loaded": metadata is not None,
        "model_metadata_generated_at": metadata.get("generated_at_utc") if metadata else None,
    }


def load_demo_resources() -> None:
    global station_mapping, station_info_map, model_lineage
    station_mapping = {station_no: index for index, station_no in enumerate(DEMO_STATION_FIXTURES)}
    station_info_map = {
        station_no: fixture["name"]
        for station_no, fixture in DEMO_STATION_FIXTURES.items()
    }
    model_lineage = demo_model_lineage()


def reset_request_metrics() -> None:
    with request_metrics_lock:
        request_metrics.clear()


def record_request_metric(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    if path == METRICS_PATH:
        return

    key = (method, path)
    with request_metrics_lock:
        metric = request_metrics.setdefault(
            key,
            {
                "requests": 0,
                "errors": 0,
                "duration_ms_sum": 0.0,
                "duration_ms_max": 0.0,
            },
        )
        metric["requests"] = int(metric["requests"]) + 1
        if status_code >= 500:
            metric["errors"] = int(metric["errors"]) + 1
        metric["duration_ms_sum"] = float(metric["duration_ms_sum"]) + duration_ms
        metric["duration_ms_max"] = max(float(metric["duration_ms_max"]), duration_ms)


def label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_metrics() -> str:
    with request_metrics_lock:
        snapshot = {
            key: dict(value)
            for key, value in sorted(request_metrics.items())
        }

    lines = [
        "# HELP youbike_api_requests_total Total HTTP requests handled by the API.",
        "# TYPE youbike_api_requests_total counter",
        "# HELP youbike_api_request_errors_total HTTP 5xx responses handled by the API.",
        "# TYPE youbike_api_request_errors_total counter",
        "# HELP youbike_api_request_duration_ms_sum Total request duration in milliseconds.",
        "# TYPE youbike_api_request_duration_ms_sum counter",
        "# HELP youbike_api_request_duration_ms_count Number of measured request durations.",
        "# TYPE youbike_api_request_duration_ms_count counter",
        "# HELP youbike_api_request_duration_ms_max Maximum observed request duration in milliseconds.",
        "# TYPE youbike_api_request_duration_ms_max gauge",
    ]
    for (method, path), metric in snapshot.items():
        labels = f'method="{label_value(method)}",path="{label_value(path)}"'
        lines.extend(
            [
                f"youbike_api_requests_total{{{labels}}} {int(metric['requests'])}",
                f"youbike_api_request_errors_total{{{labels}}} {int(metric['errors'])}",
                f"youbike_api_request_duration_ms_sum{{{labels}}} {float(metric['duration_ms_sum']):.6f}",
                f"youbike_api_request_duration_ms_count{{{labels}}} {int(metric['requests'])}",
                f"youbike_api_request_duration_ms_max{{{labels}}} {float(metric['duration_ms_max']):.6f}",
            ]
        )
    return "\n".join(lines) + "\n"

# --- 3. 定義模型架構 (必須與訓練程式碼完全同步) ---
class MultiStationLSTM(nn.Module):
    def __init__(self, num_stations, input_size=4, hidden_size=64, output_size=1, embedding_dim=5):
        super(MultiStationLSTM, self).__init__()
        
        # 1. 站點嵌入層 (ID 轉 向量)
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)
        
        # 2. LSTM 層 (輸入維度 = 4數值 + 5嵌入 = 9)
        self.lstm_input_size = input_size + embedding_dim
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, batch_first=True)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, time_steps, 5) -> 前 4 個是數值，第 5 個是 ID
        numerical_features = x[:, :, :4] 
        station_ids = x[:, :, 4].long() 
        
        # 透過 Embedding 轉換 ID
        station_embedded = self.station_embedding(station_ids)
        
        # 拼接數值特徵與站點特徵 (dim=2)
        combined_input = torch.cat((numerical_features, station_embedded), dim=2)
        
        # LSTM 運算
        out, _ = self.lstm(combined_input)
        
        # 取最後一個時間點的輸出
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out

# --- 4. 生命週期管理 (啟動時載入模型) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, station_mapping, station_info_map, db_engine, model_lineage

    if api_demo_mode_enabled():
        load_demo_resources()
        db_engine = None
        logger.info("%s is enabled; using deterministic API demo fixtures.", API_DEMO_MODE_ENV)
        yield
        model = None
        scaler = None
        station_mapping = None
        station_info_map = None
        db_engine = None
        model_lineage = unloaded_model_lineage()
        logger.info("API demo resources released.")
        return
    
    # 模型檔案路徑
    # 取得目前 main.py 的所在目錄 (api/app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = Path(current_dir) / ".." / "model_files"
    model_path = base_path / "youbike_lstm_multistation.pth"
    scaler_path = base_path / "scaler.pkl"
    mapping_path = base_path / "station_mapping.pkl"
    info_map_path = base_path / "station_info_map.pkl"

    try:
        logger.info("正在從 %s 載入資源...", base_path)
        scaler = joblib.load(scaler_path)
        station_mapping = {str(k): v for k, v in joblib.load(mapping_path).items()}
        if info_map_path.exists():
            station_info_map = joblib.load(info_map_path)
        else:
            station_info_map = {sid: sid for sid in station_mapping.keys()}
        num_stations = len(station_mapping)
        model = MultiStationLSTM(num_stations=num_stations, input_size=4)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        model_lineage = load_model_lineage(base_path)
        logger.info("所有模型資源載入成功。")
    except Exception as e:
        model_lineage = unloaded_model_lineage()
        logger.exception("模型載入失敗: %s", e)
    db_engine = create_optional_db_engine()
    yield
    model = None
    scaler = None
    station_mapping = None
    station_info_map = None
    db_engine = None
    model_lineage = unloaded_model_lineage()
    logger.info("模型資源已釋放。")

app = FastAPI(lifespan=lifespan, title="YouBike LSTM Prediction API")

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER, "").strip() or str(uuid4())
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        record_request_metric(request.method, request.url.path, 500, duration_ms)
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
            headers={REQUEST_ID_HEADER: request_id},
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    record_request_metric(request.method, request.url.path, response.status_code, duration_ms)
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

# --- 5. API 路由設定 ---

def get_rain_cat(rain: float) -> int:
    if rain == 0:
        return 0
    if rain <= 2:
        return 1
    if rain <= 10:
        return 2
    return 3

def ensure_model_ready() -> None:
    if api_demo_mode_enabled():
        return
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

def ensure_station_supported(station_no: str) -> None:
    if station_mapping is None or station_no not in station_mapping:
        raise HTTPException(status_code=404, detail="Station ID not supported by model")

def service_state() -> dict:
    demo_mode = api_demo_mode_enabled()
    model_loaded = model is not None
    scaler_loaded = scaler is not None
    station_mapping_loaded = station_mapping is not None
    station_catalog_loaded = station_info_map is not None
    return {
        "status": "online",
        "demo_mode": demo_mode,
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "station_mapping_loaded": station_mapping_loaded,
        "station_catalog_loaded": station_catalog_loaded,
        "warehouse_lookup_enabled": db_engine is not None,
        "forecast_horizon": MODEL_FORECAST_HORIZON,
        **model_lineage,
    }

def inference_ready() -> bool:
    if api_demo_mode_enabled():
        return station_mapping is not None and station_info_map is not None
    state = service_state()
    return (
        state["model_loaded"]
        and state["scaler_loaded"]
        and state["station_mapping_loaded"]
        and state["station_catalog_loaded"]
    )

def create_optional_db_engine() -> Engine | None:
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        logger.info("DB_PASSWORD is not set; warehouse lag-window lookup disabled.")
        return None

    db_user = os.getenv("DB_USER", "admin")
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "youbike_db")
    return create_engine(
        URL.create(
            "mysql+pymysql",
            username=db_user,
            password=db_password,
            host=db_host,
            port=int(db_port),
            database=db_name,
        ),
        pool_size=2,
        max_overflow=2,
        pool_recycle=1800,
        pool_pre_ping=True,
    )

def get_recent_observations_from_warehouse(
    station_no: str,
    temperature: float,
    rain: float,
) -> list[RecentObservation] | None:
    if db_engine is None:
        return None

    query = text(
        """
        SELECT bikes_available
        FROM station_status
        WHERE station_no = :station_no
        ORDER BY record_time DESC
        LIMIT :limit
        """
    )
    try:
        with db_engine.connect() as conn:
            rows = conn.execute(
                query,
                {"station_no": station_no, "limit": MODEL_TIME_STEPS},
            ).fetchall()
    except Exception as exc:
        logger.warning("Warehouse lag-window lookup failed for station %s: %s", station_no, exc)
        return None

    if len(rows) != MODEL_TIME_STEPS:
        logger.info(
            "Warehouse lag-window lookup returned %s rows for station %s; using fallback.",
            len(rows),
            station_no,
        )
        return None

    return [
        RecentObservation(
            bikes_available=int(row.bikes_available),
            temperature=temperature,
            rain=rain,
        )
        for row in reversed(rows)
    ]

def observation_features(observation: RecentObservation) -> list[float]:
    return [
        observation.bikes_available,
        observation.temperature,
        observation.rain,
        get_rain_cat(observation.rain),
    ]

def build_feature_sequence(
    station_no: str,
    bikes_available: int,
    temperature: float,
    rain: float,
    recent_observations: list[RecentObservation] | None,
) -> np.ndarray:
    if recent_observations is None:
        recent_observations = get_recent_observations_from_warehouse(
            station_no,
            temperature,
            rain,
        )

    # When real lag-window observations are unavailable, keep the old demo path.
    if recent_observations is None:
        return np.array([[
            bikes_available,
            temperature,
            rain,
            get_rain_cat(rain),
        ]] * MODEL_TIME_STEPS)

    return np.array([observation_features(observation) for observation in recent_observations])


def demo_predict_bikes(
    station_no: str,
    bikes_available: int,
    temperature: float,
    rain: float,
    recent_observations: list[RecentObservation] | None,
) -> int:
    fixture = DEMO_STATION_FIXTURES[station_no]
    capacity = int(fixture["capacity"])
    if recent_observations:
        baseline = sum(observation.bikes_available for observation in recent_observations) / len(recent_observations)
    else:
        baseline = bikes_available

    rain_penalty = 0
    if rain > 10:
        rain_penalty = 3
    elif rain > 2:
        rain_penalty = 2
    elif rain > 0:
        rain_penalty = 1

    heat_penalty = 1 if temperature >= 32 else 0
    raw_prediction = baseline + int(fixture["station_bias"]) - rain_penalty - heat_penalty
    return min(capacity, max(0, int(round(raw_prediction))))


def predict_bikes_next_hour(
    station_no: str,
    bikes_available: int,
    temperature: float,
    rain: float,
    recent_observations: list[RecentObservation] | None = None,
) -> int:
    if api_demo_mode_enabled():
        return demo_predict_bikes(
            station_no,
            bikes_available,
            temperature,
            rain,
            recent_observations,
        )

    # 特徵工程與模型輸入必須與訓練流程保持一致。
    raw_features = build_feature_sequence(
        station_no,
        bikes_available,
        temperature,
        rain,
        recent_observations,
    )
    features_scaled = scaler.transform(raw_features)

    s_idx = station_mapping[station_no]
    s_idx_seq = np.full((len(features_scaled), 1), s_idx)
    combined_input = np.hstack((features_scaled, s_idx_seq))
    input_tensor = torch.FloatTensor(combined_input).unsqueeze(0)

    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    pred_val_scaled = prediction_scaled.item()
    dummy_matrix = np.zeros((1, 4))
    dummy_matrix[0, 0] = pred_val_scaled
    real_values = scaler.inverse_transform(dummy_matrix)
    result_bikes = real_values[0][0]

    return max(0, int(round(result_bikes)))

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

@app.get("/")
def home():
    return {"status": "online", "model": "LSTM Multi-Station", "features": ["Bikes", "Temp", "Rain", "Rain_Cat"]}

@app.get("/health", response_model=HealthResponse)
def health():
    return service_state()

@app.get(METRICS_PATH)
def metrics():
    return PlainTextResponse(render_metrics(), media_type="text/plain")

@app.get("/ready", response_model=ReadinessResponse)
def readiness():
    state = service_state()
    ready = inference_ready()
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                **state,
                "ready": False,
            },
        )
    return {
        **state,
        "ready": True,
    }

@app.get("/stations", response_model=StationsResponse)
def get_stations():
    if station_info_map is None:
        raise HTTPException(status_code=503, detail="Model information not initialized")
    return {"stations": station_info_map}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    ensure_model_ready()
    ensure_station_supported(request.station_no)

    try:
        final_prediction = predict_bikes_next_hour(
            request.station_no,
            request.bikes_available,
            request.temperature,
            request.rain,
            request.recent_observations,
        )

        return {
            "station_no": request.station_no,
            "predicted_bikes_next_hour": final_prediction,
            "forecast_horizon": MODEL_FORECAST_HORIZON,
            "forecast_horizon_description": MODEL_FORECAST_HORIZON_DESCRIPTION,
            **model_lineage,
        }

    except Exception as e:
        logger.exception("Predict 執行錯誤: %s", e)
        raise HTTPException(status_code=500, detail="Internal Prediction Error")

@app.post("/stations/risk", response_model=StationsRiskResponse)
def rank_station_risks(request: StationsRiskRequest):
    ensure_model_ready()

    risks = []
    try:
        for station in request.stations:
            ensure_station_supported(station.station_no)
            predicted_bikes = predict_bikes_next_hour(
                station.station_no,
                station.bikes_available,
                request.temperature,
                request.rain,
                station.recent_observations,
            )
            observed_capacity = station.bikes_available + station.spaces_available
            predicted_spaces = max(0, observed_capacity - predicted_bikes)
            risk_level, risk_score, suggested_action = classify_station_risk(
                predicted_bikes,
                predicted_spaces,
            )
            risks.append({
                "station_no": station.station_no,
                "current_bikes_available": station.bikes_available,
                "current_spaces_available": station.spaces_available,
                "predicted_bikes_next_hour": predicted_bikes,
                "predicted_spaces_next_hour": predicted_spaces,
                "forecast_horizon": MODEL_FORECAST_HORIZON,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "suggested_action": suggested_action,
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Station risk ranking 執行錯誤: %s", e)
        raise HTTPException(status_code=500, detail="Internal Risk Ranking Error")

    risks.sort(key=lambda item: (-item["risk_score"], item["station_no"]))
    return {
        "forecast_horizon": MODEL_FORECAST_HORIZON,
        "forecast_horizon_description": MODEL_FORECAST_HORIZON_DESCRIPTION,
        **model_lineage,
        "risks": risks,
    }
