# API Contract Walkthrough

This walkthrough positions the project as a backend / AI application: the backend validates station inputs, wraps model artifacts, and turns raw predictions into an operational risk-ranking contract.

The API is implemented in `api/app/main.py`. The dashboard client in `dashboard/api_client.py` consumes the same contract, while demo mode returns deterministic mock responses with the same response shape.

## Service Readiness

`GET /`

Example response:

```json
{
  "status": "online",
  "model": "LSTM Multi-Station",
  "features": ["Bikes", "Temp", "Rain", "Rain_Cat"]
}
```

This endpoint checks that the web service is reachable. It does not prove model accuracy.

## Station Catalog

`GET /stations`

Example response:

```json
{
  "stations": {
    "500101001": "捷運公館站 (大安區)",
    "500101002": "臺大資訊大樓 (大安區)"
  }
}
```

Failure behavior:

- `503 Model information not initialized`: station metadata was not loaded.

Backend relevance:

- The dashboard uses this endpoint to build station display options.
- Station IDs are constrained to the model artifact's supported station mapping.

## Single-Station Prediction

`POST /predict`

Minimal request:

```json
{
  "station_no": "500101001",
  "bikes_available": 12,
  "temperature": 27.5,
  "rain": 0
}
```

Request with explicit lag window:

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

Example response:

```json
{
  "station_no": "500101001",
  "predicted_bikes_next_hour": 7,
  "forecast_horizon": "model_artifact_horizon",
  "forecast_horizon_description": "Legacy response keys use next_hour naming, but current artifacts should be interpreted as the model-defined horizon. The documented local evaluation uses horizon_steps=1, meaning the next observation rather than a guaranteed one-hour forecast."
}
```

Important behavior:

- `recent_observations` must contain exactly 3 rows because the current model artifacts use `MODEL_TIME_STEPS = 3`.
- If `recent_observations` is omitted and database credentials are configured, the API attempts to load the latest 3 `bikes_available` rows from `station_status`.
- The warehouse fallback does not currently load historical weather; it reuses the request's current temperature and rain for each lag row.
- If no lag window is available, the API preserves the original demo-compatible behavior by repeating the current state 3 times.

Failure behavior:

- `422`: invalid payload, such as negative `bikes_available`, unreasonable `temperature`, negative `rain`, or a lag window with the wrong length.
- `503 Model is not ready`: model or scaler was not loaded.
- `404 Station ID not supported by model`: request station is not in the artifact's station mapping.
- `500 Internal Prediction Error`: unexpected inference failure.

Backend relevance:

- The endpoint is a model-serving boundary, not a notebook shortcut.
- The request contract separates current station state from optional historical observations.
- The response includes forecast-horizon metadata so clients do not overinterpret legacy `next_hour` field names.

## Multi-Station Risk Ranking

`POST /stations/risk`

Example request:

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

Example response:

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

Risk rules:

| Condition | `risk_level` | `suggested_action` |
| --- | --- | --- |
| predicted bikes <= 2 | `stock_out` | `rebalance_in` |
| predicted spaces <= 2 | `full_load` | `rebalance_out` |
| predicted bikes <= 5 | `low_supply` | `monitor_supply` |
| predicted spaces <= 5 | `low_dock` | `monitor_docks` |
| otherwise | `normal` | `monitor` |

Important behavior:

- Results are sorted by highest `risk_score`, then by `station_no`.
- Each station may provide its own `recent_observations` lag window.
- Predicted empty docks are derived from observed capacity: `bikes_available + spaces_available - predicted_bikes`.

Failure behavior:

- `422`: invalid request, such as empty `stations`, negative availability values, invalid weather values, or wrong lag-window length.
- `503 Model is not ready`: model or scaler was not loaded.
- `404 Station ID not supported by model`: at least one station is not in the artifact's station mapping.
- `500 Internal Risk Ranking Error`: unexpected inference/ranking failure.

Backend relevance:

- This endpoint turns model output into an application workflow.
- The API response is directly consumable by dashboard cards, labels, and action text.
- This is the strongest AI-application story: prediction is not the product; decision support is the product.

## Demo Mode Contract

Dashboard demo mode does not call FastAPI. It uses deterministic mock station data and mock predictions in `dashboard/api_client.py`, but keeps the same API-shaped fields:

- `predicted_bikes_next_hour`
- `forecast_horizon`
- `risk_level`
- `risk_score`
- `suggested_action`

This makes interviews reliable without Docker Compose, a running FastAPI server, database credentials, or model artifacts. Demo output must not be described as model-performance evidence.

## Interview Summary

Use this framing:

> The backend exposes validated model-serving contracts, not just a notebook prediction. `/predict` handles single-station inference and lag-window inputs; `/stations/risk` turns model outputs into ranked operational actions. The dashboard consumes the same response shape, and demo mode preserves that contract without depending on live infrastructure.
