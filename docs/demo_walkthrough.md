# Demo Walkthrough

這份 walkthrough 是面試展示用腳本，以中文說明為主。目標不是證明 LSTM 很準，而是展示這個專案如何把資料、模型 artifact、FastAPI contract 和 dashboard decision workflow 串成一個後端 / AI 應用作品。

English short version: use the Streamlit demo to show the product flow, then map each screen back to the FastAPI model-serving contract. Demo mode is deterministic and is not model-performance evidence.

## 展示前定位

建議先用 20 秒定調：

> 這個專案我會定位成後端 / AI 應用作品。資料管線負責收集 YouBike 站點狀態；後端重點是 FastAPI 如何包裝模型 artifact、驗證 request、提供單站預測和多站風險排序；dashboard 則是 client/demo layer，用來展示 API workflow。Demo mode 是 deterministic mock，方便面試展示，不代表模型準確率。

這段很重要，因為它先避免兩個誤解：

- Streamlit 不是 production frontend claim。
- Demo output 不是 LSTM evaluation result。

## 開啟 Demo

指令：

```bash
make dashboard-demo
```

預設開啟：

```text
http://localhost:8501
```

如果面試時不能跑本機，改用 README 的 GIF 和截圖：

- `docs/images/dashboard_demo_walkthrough.gif`
- `docs/images/dashboard_demo_risk_ranking.png`

## Step 1: 先看 Demo Mode

展示 sidebar 的 demo mode 狀態。

說法：

> 這裡我刻意做了 demo mode。面試時不用啟動 Docker Compose、FastAPI、MySQL 或模型檔，也可以展示完整使用流程。這是 deterministic mock，用來展示 API response shape 和 decision workflow，不拿來宣稱模型表現。

對應後端能力：

- dashboard client 有 live API path 和 demo path。
- demo path 保留 `predicted_bikes_next_hour`、`forecast_horizon`、`risk_level`、`risk_score`、`suggested_action` 等 API-shaped fields。
- 這讓展示穩定，但仍跟後端 contract 對齊。

## Step 2: 單站預測

在單站預測頁籤：

1. 選一個站點。
2. 輸入目前可借車數、溫度、雨量。
3. 執行預測。

說法：

> 這個畫面對應 FastAPI 的 `POST /predict`。後端會驗證 station ID、車輛數、溫度、雨量，也支援傳入 3 筆 `recent_observations` 作為 lag window。現在 response 保留早期 demo 的 `next_hour` 欄位名稱，但我後來補了 `forecast_horizon` metadata，避免誤導成已驗證的一小時預測。

對應 API：

```http
POST /predict
```

核心 request：

```json
{
  "station_no": "500101001",
  "bikes_available": 12,
  "temperature": 27.5,
  "rain": 0
}
```

核心 response：

```json
{
  "station_no": "500101001",
  "predicted_bikes_next_hour": 7,
  "forecast_horizon": "model_artifact_horizon"
}
```

後端重點：

- `Pydantic` schema validation。
- `503 Model is not ready` readiness boundary。
- `404 Station ID not supported by model` station support boundary。
- optional `recent_observations` lag-window input。
- response metadata 說清楚模型 horizon。

## Step 3: 多站風險排序

在多站風險排序頁籤：

1. 保留多個站點列。
2. 調整其中幾站的可借車數或空位數，刻意製造缺車或滿站情境。
3. 執行排序。
4. 指出 priority cards 或排序表。

說法：

> 這個畫面對應 `POST /stations/risk`。它不是只回傳模型數字，而是把預測結果轉成調度可以理解的 risk label 和 suggested action，例如 `stock_out` 對應補車、`full_load` 對應移出車輛。這就是我想強調的 AI application 部分：prediction 不是產品，decision workflow 才是產品。

對應 API：

```http
POST /stations/risk
```

核心 request：

```json
{
  "temperature": 27.5,
  "rain": 0,
  "stations": [
    {
      "station_no": "500101001",
      "bikes_available": 1,
      "spaces_available": 29
    }
  ]
}
```

核心 response：

```json
{
  "forecast_horizon": "model_artifact_horizon",
  "risks": [
    {
      "station_no": "500101001",
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

後端重點：

- 一個 endpoint 處理多站 request。
- 每站可以有自己的 lag window。
- response 依 `risk_score` 排序。
- dashboard 可以直接消費 response，不需要再猜業務規則。

## Step 4: 回到架構圖

打開 README 的 backend / AI application architecture diagram。

說法：

> 這張圖是我現在對這個專案的定位。FastAPI 是核心 backend，負責 API contract、model readiness、validation、forecast metadata 和 risk translation。Streamlit 是 client/demo layer。Airflow 和 MySQL 是支撐資料來源，說明模型和分析不是憑空來的。

對應文件：

- `docs/backend_ai_architecture.md`
- `docs/backend_ai_positioning.md`
- `docs/api_contract_walkthrough.md`

## Step 5: 誠實說明 ML 邊界

最後補一句模型評估邊界。

說法：

> 我後來把 notebook training flow 整理成 CLI，補了 current-value、rolling mean、same-time previous-day 和 Ridge lag-regression baselines。結果是目前 LSTM 在 next-observation 和近似一小時 horizon 都沒有打敗最強 baseline。所以我不會說這是 production forecasting model；我會把它定位成 model-serving prototype 和 AI application workflow。

這句話的效果是把弱點轉成成熟度：

- 你知道 baseline 的重要性。
- 你不亂吹模型。
- 你知道 model-serving 和 model-accuracy 是不同能力。

## 90 秒版本

如果只有很短時間，照這個順序講：

1. 這是後端 / AI 應用作品，不是 production service。
2. Dashboard demo mode 可以穩定展示，不依賴 live infra。
3. 單站預測對應 `POST /predict`，展示 FastAPI validation 和 model-serving contract。
4. 多站排序對應 `POST /stations/risk`，展示 prediction 到 action 的 decision workflow。
5. 架構圖說明 FastAPI 是核心，Streamlit 是 client/demo，Airflow/MySQL 是資料支撐。
6. LSTM 沒打敗最強 baseline，所以不宣稱 production forecasting accuracy。

## 不要這樣講

- 不要說 Streamlit 是正式 production frontend。
- 不要說 demo mode 是真實模型準確率。
- 不要說 LSTM 已經證明很準。
- 不要把重點放在「我用了很多工具」；重點要放在 API contract 和 decision workflow。

## English Backup Pitch

Use this if the interviewer wants an English summary:

> I position this as a backend / AI application project. FastAPI is the model-serving boundary: it validates prediction requests, loads model artifacts, exposes single-station prediction and multi-station risk-ranking endpoints, and returns forecast-horizon metadata. Streamlit is a demo client for the same API-shaped workflow. The data pipeline with Airflow and MySQL supports the application, but the main story is turning model output into a tested decision-support backend.
