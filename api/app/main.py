from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import pandas as pd
import numpy as np
import os
from contextlib import asynccontextmanager

# --- 定義資料格式 ---
class PredictRequest(BaseModel):
    station_no: str
    bikes_available: int
    temperature: float
    rain: float

class PredictResponse(BaseModel):
    station_no: str
    predicted_bikes_next_hour: int

class StationsResponse(BaseModel):
    supported_stations: list[str]

# --- 全域變數 ---
model = None
scaler = None
station_mapping = None

# --- 生命週期管理器 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, station_mapping
    
    # 使用絕對路徑
    base_path = "/app/model_files"
    model_path = os.path.join(base_path, "youbike_lstm_multistation.pth")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    mapping_path = os.path.join(base_path, "station_mapping.pkl")

    print(f"[INFO] Loading model from: {model_path}")

    try:
        # 1. 載入對照表
        raw_mapping = joblib.load(mapping_path)
        station_mapping = {str(k): v for k, v in raw_mapping.items()}
        
        # 2. 載入 Scaler
        scaler = joblib.load(scaler_path)
        
        # 3. 載入 PyTorch 模型
        class MultiStationLSTM(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, num_stations, embedding_dim):
                super(MultiStationLSTM, self).__init__()
                self.station_embedding = torch.nn.Embedding(num_stations, embedding_dim)
                self.lstm = torch.nn.LSTM(input_size + embedding_dim, hidden_size, num_layers, batch_first=True)
                self.fc = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x, station_idx):
                station_emb = self.station_embedding(station_idx).unsqueeze(1)
                station_emb = station_emb.repeat(1, x.size(1), 1)
                combined_input = torch.cat((x, station_emb), dim=2)
                out, _ = self.lstm(combined_input)
                out = self.fc(out[:, -1, :])
                return out

        # [參數正確] 根據之前的報錯修正
        input_size = 3  
        hidden_size = 64
        num_layers = 1      # 這裡是一層
        output_size = 1
        embedding_dim = 5   # 這裡是 5
        num_stations = len(station_mapping)

        model = MultiStationLSTM(input_size, hidden_size, num_layers, output_size, num_stations, embedding_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        print("[INFO] Model loaded successfully.")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
    
    yield
    model = None
    scaler = None
    print("[INFO] Model resources released.")

app = FastAPI(lifespan=lifespan, title="YouBike Traffic Prediction API")

@app.get("/")
def home():
    return {"message": "YouBike API is running!", "docs_url": "/docs"}

@app.get("/stations", response_model=StationsResponse)
def get_stations():
    if station_mapping is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"supported_stations": list(station_mapping.keys())}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    if request.station_no not in station_mapping:
        raise HTTPException(status_code=404, detail="Station not found")

    try:
        # 1. 準備特徵 [Bikes, Temp, Rain] (這是 Scaler 規定的順序)
        features = np.array([[request.bikes_available, request.temperature, request.rain]])
        
        # 2. 標準化
        features_scaled = scaler.transform(features)
        
        # 3. 轉 Tensor
        x_input = torch.FloatTensor(features_scaled).unsqueeze(1) 
        station_idx = station_mapping[request.station_no]
        station_input = torch.LongTensor([station_idx])

        # 4. 推論
        with torch.no_grad():
            prediction_scaled = model(x_input, station_input)
            
        # 5. 反標準化 (✅ 順序修正版)
        pred_val = prediction_scaled.item()
        
        # Scaler 的欄位順序是 ['bikes', 'temp', 'rain']
        # 我們要把預測出來的車輛數 (pred_val) 放在 "第 0 個位置"
        dummy_features = np.array([[pred_val, 0, 0]]) 
        
        real_values = scaler.inverse_transform(dummy_features)
        
        # 取出第 0 個值 (也就是還原後的 bikes)
        result_value = real_values[0][0] 
        
        predicted_bikes = max(0, int(round(result_value)))

        return {
            "station_no": request.station_no,
            "predicted_bikes_next_hour": predicted_bikes
        }

    except Exception as e:
        print(f"[Error] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))