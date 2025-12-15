import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

# --- 1. Model Architecture Definition ---
class MultiStationLSTM(nn.Module):
    def __init__(self, num_stations, input_size=3, hidden_size=64, output_size=1, embedding_dim=5):
        super(MultiStationLSTM, self).__init__()
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)
        self.lstm_input_size = input_size + embedding_dim
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        numerical_features = x[:, :, :3]
        station_ids = x[:, :, 3].long()
        station_embedded = self.station_embedding(station_ids)
        combined_input = torch.cat((numerical_features, station_embedded), dim=2)
        out, _ = self.lstm(combined_input)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# --- 2. Global Artifacts Store ---
artifacts = {}

# --- 3. Lifespan Manager  ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    base_path = "model_files"
    model_path = os.path.join(base_path, "youbike_lstm_multistation.pth")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    mapping_path = os.path.join(base_path, "station_mapping.pkl")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
    else:
        print("[INFO] Loading model artifacts...")
        
        try:
            scaler = joblib.load(scaler_path)
            raw_mapping = joblib.load(mapping_path)
            station_mapping = {str(k): v for k, v in raw_mapping.items()}
            
            num_stations = len(station_mapping)
            model = MultiStationLSTM(num_stations=num_stations)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()

            artifacts["model"] = model
            artifacts["scaler"] = scaler
            artifacts["mapping"] = station_mapping
            print("[INFO] Model, Scaler, and Mapping loaded successfully.")
            print(f"[INFO] Supported Stations: {list(station_mapping.keys())}") # 印出來檢查一下
        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {e}")
        
    yield
    artifacts.clear()

app = FastAPI(title="YouBike Multi-Station Prediction API", lifespan=lifespan)

# --- 4. Input Schema ---
class PredictionRequest(BaseModel):
    station_no: str
    bikes_available: int
    temperature: float
    rain: float

# --- 5. API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/stations")
def get_supported_stations():
    mapping = artifacts.get("mapping")
    if not mapping:
        raise HTTPException(status_code=500, detail="Mapping not loaded")
    return {"supported_stations": list(mapping.keys())}

@app.post("/predict")
def predict_traffic(request: PredictionRequest):
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")
    mapping = artifacts.get("mapping")

    if not model or not scaler or not mapping:
        raise HTTPException(status_code=500, detail="Model artifacts not initialized")

    if request.station_no not in mapping:
        raise HTTPException(status_code=400, detail=f"Station {request.station_no} not supported.")

    try:
        station_idx = mapping[request.station_no]

        input_features = np.array([[
            request.bikes_available, 
            request.temperature, 
            request.rain
        ]]) 

        scaled_features = scaler.transform(input_features)

        # Replicate input to simulate time steps (1, 3, 3)
        seq_values = np.tile(scaled_features, (1, 3, 1))
        
        # Create station ID tensor (1, 3, 1)
        seq_ids = np.full((1, 3, 1), station_idx)
        
        # Combine features (1, 3, 4)
        final_input = np.dstack((seq_values, seq_ids))
        
        tensor_data = torch.FloatTensor(final_input)
        
        with torch.no_grad():
            prediction_scaled = model(tensor_data)
            prediction_value = prediction_scaled.item()

        # Inverse transform
        dummy_array = np.zeros((1, 3))
        dummy_array[0, 0] = prediction_value
        actual_prediction = scaler.inverse_transform(dummy_array)[0, 0]
        final_prediction = max(0.0, actual_prediction)
        return {
            "station": request.station_no,
            "inputs": request.dict(),
            "predicted_bikes_next_hour": round(final_prediction, 1)
        }

    except Exception as e:
        return {"error": str(e)}