import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- 1. 定義資料格式 ---
class PredictRequest(BaseModel):
    station_no: str
    bikes_available: int
    temperature: float
    rain: float

class PredictResponse(BaseModel):
    station_no: str
    predicted_bikes_next_hour: int

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
    base_path = "/app/model_files"
    model_path = os.path.join(base_path, "youbike_lstm_multistation.pth")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    mapping_path = os.path.join(base_path, "station_mapping.pkl")
    info_map_path = os.path.join(base_path, "station_info_map.pkl")

    try:
        print(f"[INFO] 正在從 {base_path} 載入資源...")
        
        # 1. 載入必要的 Pkl 檔案
        scaler = joblib.load(scaler_path)
        station_mapping = {str(k): v for k, v in joblib.load(mapping_path).items()}
        
        # 載入站點資訊檔 (若無則使用空字典)
        if os.path.exists(info_map_path):
            station_info_map = joblib.load(info_map_path)
        else:
            station_info_map = {sid: sid for sid in station_mapping.keys()}

        # 2. 初始化模型並載入權重
        num_stations = len(station_mapping)
        model = MultiStationLSTM(num_stations=num_stations, input_size=4) # 關鍵修正: input_size=4
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        print("[INFO] 所有模型資源載入成功。")
        
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
    
    yield
    # 釋放資源
    model = None
    scaler = None
    print("[INFO] 模型資源已釋放。")

app = FastAPI(lifespan=lifespan, title="YouBike LSTM Prediction API")

# --- 5. API 路由設定 ---

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
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
        
    if request.station_no not in station_mapping:
        raise HTTPException(status_code=404, detail="Station ID not supported by model")

    try:
        # A. 特徵工程: 將原始降雨量轉換為 Rain_Cat (0, 1, 2, 3)
        # 必須與訓練時的邏輯完全一致
        def get_rain_cat(r):
            if r == 0: return 0
            elif r <= 2: return 1
            elif r <= 10: return 2
            else: return 3
        
        rain_cat = get_rain_cat(request.rain)

        # B. 準備特徵矩陣 [Bikes, Temp, Rain, Rain_Cat]
        # 注意：這裡的順序必須與 Scaler 訓練時一致
        raw_features = np.array([[
            request.bikes_available, 
            request.temperature, 
            request.rain, 
            rain_cat
        ]])
        
        # C. 數據標準化
        features_scaled = scaler.transform(raw_features)
        
        # D. 製作序列輸入 (Time Steps = 3)
        # 由於即時預測僅有當前數據，我們複製 3 份來模擬穩定狀態序列
        seq_features = np.tile(features_scaled, (3, 1))
        
        # E. 加入站點索引 ID
        s_idx = station_mapping[request.station_no]
        s_idx_seq = np.full((3, 1), s_idx)
        
        # 合併為 (3, 5) -> 4個數值特徵 + 1個 ID
        combined_input = np.hstack((seq_features, s_idx_seq))
        
        # F. 轉換為 Tensor 並增加 Batch 維度 -> (1, 3, 5)
        input_tensor = torch.FloatTensor(combined_input).unsqueeze(0)

        # G. 模型推論
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            
        # H. 反標準化 (Inverse Transform)
        pred_val_scaled = prediction_scaled.item()
        
        # 建立與 Scaler 相同的 4 欄位 dummy 矩陣
        dummy_matrix = np.zeros((1, 4))
        dummy_matrix[0, 0] = pred_val_scaled # 將預測結果放在 bikes 欄位
        
        real_values = scaler.inverse_transform(dummy_matrix)
        result_bikes = real_values[0][0]
        
        # I. 結果處理：四捨五入並確保不為負數
        final_prediction = max(0, int(round(result_bikes)))

        return {
            "station_no": request.station_no,
            "predicted_bikes_next_hour": final_prediction
        }

    except Exception as e:
        print(f"[RUNTIME ERROR] {e}")
        raise HTTPException(status_code=500, detail="Internal Prediction Error")