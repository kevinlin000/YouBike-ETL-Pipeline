# 系統設計說明

這份文件整理 YouBike 專案的後端與 AI 應用設計。它的目的不是把作品包裝成正在營運的大型 production service，而是清楚說明：資料如何進來、怎麼建模、模型如何被 API 包裝、dashboard 如何消費 API，以及目前還有哪些工程邊界。

## 設計目標

本專案要解決的問題是：在站點層級判斷 YouBike 可借車輛與可停空位的供需狀態，並把模型或規則輸出轉成可展示的調度輔助流程。

設計上刻意把責任拆開：

- Airflow / ETL 負責資料擷取與資料品質檢查。
- MySQL 保存站點維度資料與高頻狀態事實資料。
- dbt scaffold 描述 staging / mart 分析層。
- PyTorch LSTM artifact 提供模型服務化的載入目標。
- FastAPI 負責 API contract、request validation、readiness、request tracing、模型推論與風險排序。
- Streamlit dashboard 負責展示單站預測與多站風險排序流程。

這個切法讓專案可以用後端 / AI 應用工程角度討論，而不是只停留在 notebook 或 dashboard。

## 非目標

目前不主張這個 repo 已經完成下列能力：

- 高流量 production service。
- 真正線上營運的即時調度系統。
- 完整 MLOps，例如 model registry、自動 retraining、線上 drift monitoring。
- 大數據平台，例如 Spark、Kafka、Data Lake 或 Kubernetes。
- 已驗證準確率優於 baseline 的 production forecasting model。

這些不是逃避限制，而是避免把展示專案講成超出證據的系統。面試時可以把重點放在：資料建模、API 邊界、模型服務化、測試與可展示流程。

## 高層架構

資料與服務流程如下：

1. Airflow DAG 定期呼叫 YouBike 2.0 開放資料。
2. ETL 將站點靜態資訊與站點即時狀態拆成不同資料表。
3. MySQL 保存 `station_info` 與 `station_status`。
4. dbt staging / marts 整理成站點小時與行政區小時的分析資料。
5. Notebook 與訓練程式使用歷史資料建立 LSTM artifact 與 baseline 評估。
6. FastAPI 啟動時載入模型權重、scaler、station mapping 與 station metadata。
7. Dashboard 呼叫 FastAPI 或 demo client，展示單站預測與多站風險排序。

對外展示時有兩種模式：

- 正式服務路徑：FastAPI 讀取模型 artifact，必要時從 MySQL 查詢近期 lag window。
- Demo 路徑：FastAPI 或 dashboard 使用固定範例資料，方便面試或本機檢視，不依賴模型檔、MySQL 或 Docker Compose。

## 資料模型

核心資料模型刻意保持簡潔：

### `station_info`

站點維度表，保存低變動資料：

- `station_no`：站點主鍵。
- `name_tw`：中文站名。
- `district`：行政區。
- `lat` / `lng`：座標。
- `total_spaces`：站點容量。

### `station_status`

站點狀態事實表，保存高頻時間序列資料：

- `id`：流水號主鍵。
- `station_no`：外鍵，對應 `station_info.station_no`。
- `bikes_available`：目前可借車輛。
- `spaces_available`：目前可停空位。
- `record_time`：觀測時間。

`station_status` 使用 `(station_no, record_time)` 作為 unique key，避免同一站點同一時間重複寫入。這個設計可以清楚回答面試常見問題：資料粒度是一站一時間點，站點 metadata 與狀態紀錄分開，避免在高頻資料中重複保存站點靜態欄位。

### dbt analytics layer

dbt scaffold 目前提供：

- `stg_station_info`：清理後的站點維度模型。
- `stg_station_status`：清理後的狀態事實模型，加入缺車與滿站 flags。
- `mart_station_hourly_health`：站點小時層級營運健康指標。
- `mart_district_peak_hour_health`：行政區小時層級營運健康指標，加入尖峰 / 離峰標籤。

這層不是完整企業資料倉儲，而是用來展示分析模型邊界、資料測試與 downstream dashboard / model feature 檢查。

## API 設計

FastAPI 是本專案的後端核心。主要 endpoint：

| Method | Path | 責任 |
| --- | --- | --- |
| GET | `/health` | liveness check，確認 API process 可回應，並回傳資源載入狀態 |
| GET | `/ready` | inference readiness，確認是否具備推論所需資源 |
| GET | `/metrics` | Prometheus-style 文字指標，輸出 request count、5xx error count 與 latency summary |
| GET | `/stations` | 回傳模型支援的站點 catalog |
| POST | `/predict` | 單站模型時窗預測 |
| POST | `/stations/risk` | 多站缺車 / 滿站風險排序 |

API request 使用 Pydantic 驗證，包含：

- 車輛數與空位不可為負。
- 溫度需在合理範圍。
- 雨量不可為負。
- `recent_observations` 若存在，必須剛好符合模型需要的 3 筆 lag window。
- 不在 station mapping 內的站點回 `404`。
- 模型未就緒時回 `503`。

這些邊界讓 dashboard 不需要猜後端狀態，也讓 API contract 可被測試。

## Readiness 與 liveness

`/health` 與 `/ready` 分開處理：

- `/health`：只要 API process 活著就回 `200`，並列出 `model_loaded`、`scaler_loaded`、`station_mapping_loaded`、`station_catalog_loaded`、`warehouse_lookup_enabled` 與 `demo_mode`。
- `/ready`：正式模式下，只有模型、scaler、station mapping 與 station metadata 都載入時才回 `200`。任一必要資源缺失時回 `503`。

這個切法可以避免常見誤判：服務 process 還活著，不代表模型推論已經可用。

## Request tracing

API middleware 會為每個 response 加上 `X-Request-ID`：

- 呼叫端有傳入 `X-Request-ID` 時，API 會沿用。
- 呼叫端沒有傳入時，API 會產生新的 UUID。
- log 會記錄 request id、method、path、status code 與 duration。
- 未處理例外會收斂成標準 500 JSON，並保留 request id。

這讓 dashboard 或 API client 發生問題時，可以用同一組 request id 回到後端 log 追查單次請求。這不是完整 observability 平台，但已經具備後端服務最基本的 correlation 能力。

## Metrics endpoint

`GET /metrics` 會輸出輕量 Prometheus-style text metrics。目前追蹤的指標包含：

- `youbike_api_requests_total`
- `youbike_api_request_errors_total`
- `youbike_api_request_duration_ms_sum`
- `youbike_api_request_duration_ms_count`
- `youbike_api_request_duration_ms_max`

指標以 HTTP method 和 path 作為 label。`/metrics` 本身不會被納入統計，避免 scrape 行為讓 request count 自我膨脹。

這個 endpoint 的定位是基礎可觀測性：讓本機 demo、面試展示或簡單監控可以看到 API 是否有 request、5xx error 和 latency 變化。它不是完整 production monitoring；若要上 production，仍應補 histogram buckets、process metrics、dashboard、alert rules 和 log aggregation。

## Demo mode 設計

專案現在有兩個 demo mode：

### API demo mode

啟動方式：

```bash
make api-demo
```

等同於：

```bash
API_DEMO_MODE=true python -m uvicorn api.app.main:app --reload --port 8000
```

這個模式會：

- 載入固定站點 catalog。
- 不讀取模型檔。
- 不連 MySQL。
- `/ready` 回 `200`，並標示 `demo_mode: true`。
- `/predict` 與 `/stations/risk` 回傳 deterministic simulated responses。
- 保留 request validation 與 `X-Request-ID` tracing。

它的目的不是假裝模型準確，而是讓面試時可以穩定展示後端 API contract。

### Dashboard fixed-sample-data mode

啟動方式：

```bash
make dashboard-demo
```

等同於：

```bash
DASHBOARD_DEMO_MODE=true streamlit run dashboard/app.py
```

這個模式讓 dashboard 不依賴 live FastAPI、模型檔或 Docker Compose，也能展示單站預測與多站風險排序流程。它展示的是 UI flow 與 API-shaped response，不是模型評估結果。

## 模型服務化邊界

FastAPI 啟動時會嘗試載入：

- PyTorch LSTM 權重。
- scaler。
- station mapping。
- station info map。

`/predict` 支援兩種 lag window 來源：

- request 直接傳入 `recent_observations`。
- 若 request 未傳入且 DB credentials 存在，API 嘗試從 MySQL `station_status` 查最近 3 筆 `bikes_available`。

目前限制是 warehouse 沒有保存對齊時間的 weather history，因此自動查詢 lag window 時只能沿用 request 中的 temperature / rain。這也是為什麼 README 與模型評估文件都明確說明：目前 LSTM 是 model-serving prototype，不是已驗證的高準確率 production forecasting model。

## 風險排序設計

`/stations/risk` 的價值不只是批次推論，而是把 prediction 轉成調度可讀的風險結果：

| 條件 | risk level | suggested action |
| --- | --- | --- |
| predicted bikes <= 2 | `stock_out` | `rebalance_in` |
| predicted spaces <= 2 | `full_load` | `rebalance_out` |
| predicted bikes <= 5 | `low_supply` | `monitor_supply` |
| predicted spaces <= 5 | `low_dock` | `monitor_docks` |
| otherwise | `normal` | `monitor` |

API 回傳排序後的風險清單，dashboard 可以直接呈現調度順位，不需要把風險規則散落在前端。

## Failure modes

目前已明確處理的錯誤邊界：

- 模型或 scaler 未載入：`503 Model is not ready`。
- station catalog 未載入：`503 Model information not initialized`。
- 不支援的站點：`404 Station ID not supported by model`。
- request payload 不合法：`422`。
- warehouse lag-window 查詢失敗：記錄 warning，回到 demo-compatible fallback。
- 未處理例外：回標準 500 JSON，保留 request id。

仍可改進的地方：

- 將 log 改成結構化 JSON，方便接 log aggregator。
- 將目前 `/metrics` 擴充為 histogram buckets、process metrics 與 prediction error count。
- 將 demo mode 與正式模式的設定集中到 config module。
- 為模型 artifact 加 checksum / version metadata。

## 測試策略

目前測試覆蓋：

- ETL transform 與資料品質邊界。
- FastAPI health/readiness。
- request tracing。
- `/metrics` request count、error count 與 latency summary。
- API demo mode。
- `/predict` payload validation、unknown station、lag window、warehouse fallback。
- `/stations/risk` 批次排序、風險分類、per-station lag window。
- dashboard client live/demo path。
- LSTM training script 的小型 fixture、baseline suite 與 artifact 輸出。
- dbt seed / source / model tests。
- API concurrency benchmark 的 latency、error rate 與 throughput 摘要輸出。

CI 會在 push / pull request 時跑 Python tests 與 dbt seed/build。這讓作品不是只靠手動展示，也能用自動化測試證明主要 contract 沒有破掉。

## 效能 benchmark

專案提供一個零新增依賴的本機 API concurrency benchmark：

```bash
make api-demo
```

另開一個 terminal：

```bash
make api-benchmark
```

benchmark 會以 5 個 worker 併發呼叫：

- `GET /ready`
- `POST /predict`
- `POST /stations/risk`

輸出欄位包含：

- requests
- errors
- error rate
- throughput
- min / avg / p50 / p95 / p99 / max latency

本機 API demo mode 範例輸出如下；這只是開發機上的 local benchmark，不代表 production SLO：

| endpoint | requests | errors | error % | rps | min ms | avg ms | p50 ms | p95 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| predict | 30 | 0 | 0.00 | 102.68 | 12.92 | 16.02 | 15.54 | 19.24 | 19.66 | 19.68 |
| ready | 30 | 0 | 0.00 | 102.68 | 8.65 | 13.45 | 13.36 | 16.07 | 16.32 | 16.38 |
| risk | 30 | 0 | 0.00 | 102.68 | 13.82 | 16.07 | 15.42 | 18.96 | 20.28 | 20.64 |

這個 benchmark 不是正式 production load test。它的定位是檢查本機 demo API 的基本可用性、latency 量級、錯誤率與簡單併發行為；若要討論高併發或正式 SLO，仍應補 k6 / Locust、ramp-up、固定硬體規格、監控 dashboard 與多輪測試紀錄。

## 擴展方向

如果要把這個作品往更高階後端 / AI 應用作品推進，優先順序如下：

1. **正式負載測試**：在目前本機 concurrency benchmark 之外，用 k6 或 Locust 測 `/predict`、`/stations/risk` 的 ramp-up、長時間穩定性與錯誤率。
2. **結構化 observability**：將目前 `/metrics` 擴充為 histogram buckets，輸出 JSON log，並建立簡單 dashboard / alert rules。
3. **模型版本管理**：在 response 加 model version / artifact hash，讓 prediction 可追溯。
4. **Feature store-lite**：補 weather history table，讓 warehouse fallback 能查對齊時間的天氣特徵。
5. **背景工作與快取**：對熱門站點或批次風險排序加入 cache / scheduled precompute。
6. **部署文件**：整理 Docker Compose profile、環境變數、secret handling、rolling restart 與 rollback 策略。

這些方向比繼續美化 dashboard 更能提升後端 / AI 應用工程的面試說服力。

## 面試時的講法

可以用這段總結：

> 這個專案我會定位成後端 / AI 應用作品。資料工程負責提供可靠資料來源，FastAPI 是模型服務邊界，處理 request validation、readiness、request tracing、單站預測與多站風險排序。模型目前是 model-serving prototype，我不會誇大準確率；我更想展示的是如何把資料、模型 artifact、API contract、測試和 dashboard 串成可檢視的應用流程。

## English Summary

This project is best positioned as a backend / AI application engineering portfolio project. FastAPI is the model-serving boundary: it validates requests, exposes liveness/readiness endpoints, attaches request IDs, serves single-station prediction and multi-station risk-ranking APIs, and supports a deterministic API demo mode for interview walkthroughs. Airflow and MySQL provide the data foundation, while Streamlit is the client/demo layer. The LSTM should be described as a model-serving prototype rather than a proven production forecasting model.
