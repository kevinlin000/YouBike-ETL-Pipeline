import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
MODEL_TIME_STEPS = 3

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
    risk_level: str
    risk_score: int
    suggested_action: str

class StationsRiskResponse(BaseModel):
    risks: list[StationRiskResult]

class StationsResponse(BaseModel):
    # 改為回傳字典：{ "station_no": "中文名稱 (行政區)", ... }
    stations: dict 

# --- 2. 全域變數 ---
model = None
scaler = None
station_mapping = None
station_info_map = None  # 新增：站點資訊對照表

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
    global model, scaler, station_mapping, station_info_map
    
    # 模型檔案路徑
    # 取得目前 main.py 的所在目錄 (api/app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "model_files")
    model_path = os.path.join(base_path, "youbike_lstm_multistation.pth")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    mapping_path = os.path.join(base_path, "station_mapping.pkl")
    info_map_path = os.path.join(base_path, "station_info_map.pkl")

    try:
        logger.info("正在從 %s 載入資源...", base_path)
        scaler = joblib.load(scaler_path)
        station_mapping = {str(k): v for k, v in joblib.load(mapping_path).items()}
        if os.path.exists(info_map_path):
            station_info_map = joblib.load(info_map_path)
        else:
            station_info_map = {sid: sid for sid in station_mapping.keys()}
        num_stations = len(station_mapping)
        model = MultiStationLSTM(num_stations=num_stations, input_size=4)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        logger.info("所有模型資源載入成功。")
    except Exception as e:
        logger.exception("模型載入失敗: %s", e)
    yield
    model = None
    scaler = None
    logger.info("模型資源已釋放。")

app = FastAPI(lifespan=lifespan, title="YouBike LSTM Prediction API")

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
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

def ensure_station_supported(station_no: str) -> None:
    if station_mapping is None or station_no not in station_mapping:
        raise HTTPException(status_code=404, detail="Station ID not supported by model")

def observation_features(observation: RecentObservation) -> list[float]:
    return [
        observation.bikes_available,
        observation.temperature,
        observation.rain,
        get_rain_cat(observation.rain),
    ]

def build_feature_sequence(
    bikes_available: int,
    temperature: float,
    rain: float,
    recent_observations: list[RecentObservation] | None,
) -> np.ndarray:
    # When real lag-window observations are unavailable, keep the old demo path.
    if recent_observations is None:
        return np.array([[
            bikes_available,
            temperature,
            rain,
            get_rain_cat(rain),
        ]] * MODEL_TIME_STEPS)

    return np.array([observation_features(observation) for observation in recent_observations])

def predict_bikes_next_hour(
    station_no: str,
    bikes_available: int,
    temperature: float,
    rain: float,
    recent_observations: list[RecentObservation] | None = None,
) -> int:
    # 特徵工程與模型輸入必須與訓練流程保持一致。
    raw_features = build_feature_sequence(
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
            "predicted_bikes_next_hour": final_prediction
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
    return {"risks": risks}
