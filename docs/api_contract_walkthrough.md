# API Contract 說明

本文件整理 FastAPI 服務的 endpoint、request/response 形狀與錯誤邊界。API 實作位於 `api/app/main.py`，dashboard client 位於 `dashboard/api_client.py`。

## 服務狀態

`GET /`

範例 response：

```json
{
  "status": "online",
  "model": "LSTM Multi-Station",
  "features": ["Bikes", "Temp", "Rain", "Rain_Cat"]
}
```

這個 endpoint 只表示 Web service 可連線，不代表模型已完成評估或具備正式預測品質。

## 站點清單

`GET /stations`

範例 response：

```json
{
  "stations": {
    "500101001": "捷運公館站 (大安區)",
    "500101002": "臺大資訊大樓 (大安區)"
  }
}
```

錯誤行為：

- `503 Model information not initialized`：站點 metadata 尚未載入。

用途：

- Dashboard 使用此 endpoint 產生站點選單。
- API 會依模型 artifact 的 station mapping 限制可支援站點。

## 單站預測

`POST /predict`

最小 request：

```json
{
  "station_no": "500101001",
  "bikes_available": 12,
  "temperature": 27.5,
  "rain": 0
}
```

含近期觀測值的 request：

```json
{
  "station_no": "500101001",
  "bikes_available": 12,
  "temperature": 27.5,
  "rain": 0,
  "recent_observations": [
    {"bikes_available": 10, "temperature": 27.0, "rain": 0},
    {"bikes_available": 11, "temperature": 27.2, "rain": 0.5},
    {"bikes_available": 12, "temperature": 27.5, "rain": 3}
  ]
}
```

範例 response：

```json
{
  "station_no": "500101001",
  "predicted_bikes_next_hour": 7,
  "forecast_horizon": "model_artifact_horizon",
  "forecast_horizon_description": "Legacy response keys use next_hour naming, but current artifacts should be interpreted as the model-defined horizon. The documented local evaluation uses horizon_steps=1, meaning the next observation rather than a guaranteed one-hour forecast."
}
```

重要行為：

- 目前模型 artifact 使用 `MODEL_TIME_STEPS = 3`，因此 `recent_observations` 必須剛好 3 筆。
- 若 request 沒有提供 `recent_observations`，且 DB credentials 存在，API 會嘗試從 MySQL `station_status` 查詢最近 3 筆 `bikes_available`。
- warehouse fallback 目前沒有查詢歷史天氣，因此會沿用 request 中的 `temperature` 與 `rain`。
- 若沒有可用 lag window，API 會沿用早期 demo 行為，將目前狀態重複 3 次後推論。
- Response 保留 `predicted_bikes_next_hour` 欄位名稱是為了相容舊 demo；實際預測時窗需看 `forecast_horizon` metadata。

錯誤行為：

- `422`：payload 不合法，例如負數車輛數、不合理氣溫、負數雨量、lag window 長度錯誤。
- `503 Model is not ready`：模型或 scaler 尚未載入。
- `404 Station ID not supported by model`：站點不在模型 artifact 支援範圍內。
- `500 Internal Prediction Error`：推論過程發生未預期錯誤。

## 多站風險排序

`POST /stations/risk`

範例 request：

```json
{
  "temperature": 27.5,
  "rain": 0,
  "stations": [
    {
      "station_no": "500101001",
      "bikes_available": 1,
      "spaces_available": 29
    },
    {
      "station_no": "500101002",
      "bikes_available": 28,
      "spaces_available": 2
    }
  ]
}
```

範例 response：

```json
{
  "forecast_horizon": "model_artifact_horizon",
  "forecast_horizon_description": "Legacy response keys use next_hour naming, but current artifacts should be interpreted as the model-defined horizon. The documented local evaluation uses horizon_steps=1, meaning the next observation rather than a guaranteed one-hour forecast.",
  "risks": [
    {
      "station_no": "500101001",
      "current_bikes_available": 1,
      "current_spaces_available": 29,
      "predicted_bikes_next_hour": 0,
      "predicted_spaces_next_hour": 30,
      "forecast_horizon": "model_artifact_horizon",
      "risk_level": "stock_out",
      "risk_score": 102,
      "suggested_action": "rebalance_in"
    }
  ]
}
```

風險規則：

| 條件 | `risk_level` | `suggested_action` |
| --- | --- | --- |
| predicted bikes <= 2 | `stock_out` | `rebalance_in` |
| predicted spaces <= 2 | `full_load` | `rebalance_out` |
| predicted bikes <= 5 | `low_supply` | `monitor_supply` |
| predicted spaces <= 5 | `low_dock` | `monitor_docks` |
| otherwise | `normal` | `monitor` |

重要行為：

- 結果會依 `risk_score` 由高到低排序，同分時再依 `station_no` 排序。
- 每個站點可各自提供 `recent_observations`。
- 預測空位數由觀測容量推估：`bikes_available + spaces_available - predicted_bikes`。
- Dashboard 可直接消費 response 中的 `risk_level`、`risk_score` 與 `suggested_action`。

錯誤行為：

- `422`：request 不合法，例如空的 `stations`、負數車輛/空位、不合法天氣值、lag window 長度錯誤。
- `503 Model is not ready`：模型或 scaler 尚未載入。
- `404 Station ID not supported by model`：至少一個站點不在模型支援範圍內。
- `500 Internal Risk Ranking Error`：推論或排序過程發生未預期錯誤。

## Demo Mode Contract

Dashboard 固定範例資料模式不會呼叫 FastAPI，而是在 `dashboard/api_client.py` 使用固定站點資料與模擬推論結果。它保留與 API 相同的主要欄位：

- `predicted_bikes_next_hour`
- `forecast_horizon`
- `risk_level`
- `risk_score`
- `suggested_action`

這個設計讓 dashboard 可以在沒有 FastAPI、模型檔、資料庫或 Docker Compose 的情況下檢視流程。固定範例資料模式只代表介面與資料形狀，不代表模型表現。

## English Summary

The FastAPI service exposes a station catalog, single-station prediction, and multi-station risk-ranking endpoint. Requests are validated with Pydantic, model readiness is handled explicitly, unsupported stations return clear errors, and response metadata clarifies the forecast horizon. The dashboard fixed-sample-data mode keeps the same response shape but uses reproducible simulated data, so it should not be treated as model-performance evidence.
