# Operations Runbook

這份文件整理專案的啟動、驗證、觀測與故障處理方式。它的定位是作品維護與面試說明用 runbook，不代表目前有一套持續營運中的 production 環境。

## 執行模式

專案目前有三種常用執行模式：

| 模式 | 指令 | 適用情境 | 依賴 |
| --- | --- | --- | --- |
| Dashboard demo | `make dashboard-demo` | 單獨展示 Streamlit 流程 | Python app dependencies |
| API demo | `make api-demo` | 展示 FastAPI contract、request validation、readiness、metrics、JSON log | Python app dependencies |
| Full stack | `make up` | 啟動 Airflow、MySQL、FastAPI、dashboard | Docker Compose、`.env` |

Demo mode 使用固定資料與可重現 mock prediction。它適合展示 API / dashboard 流程，但不應被描述成模型評估結果。

## 本機 Demo 操作

### Dashboard demo

```bash
make install-app
make dashboard-demo
```

開啟：

```text
http://localhost:8501
```

確認項目：

- sidebar 顯示 demo mode。
- 單站預測可以回傳 `forecast_horizon`。
- 多站風險排序可以產生 `risk_level`、`risk_score` 與 `suggested_action`。

### API demo

```bash
make install-app
make api-demo
```

開啟：

```text
http://localhost:8000/docs
```

基本檢查：

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/metrics
```

重新產生 API contract artifacts：

```bash
make api-contract
```

輸出：

- `docs/openapi.json`：FastAPI OpenAPI schema。
- `docs/api_examples.http`：本機 request examples，可用 REST Client 類工具執行。

範例 prediction：

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-check' \
  -d '{"station_no":"500101001","bikes_available":15,"temperature":25,"rain":0}'
```

API demo mode 的 `/ready` 應回 `200`，並包含：

- `demo_mode: true`
- `model_version: api-demo-fixtures-v1`
- `model_metadata_loaded: false`

## Full Stack 啟動

### 環境變數

先建立 `.env`：

```bash
cp .env.example .env
```

必要設定：

```env
MYSQL_ROOT_PASSWORD=your_root_password_here
MYSQL_DATABASE=youbike_db
MYSQL_USER=youbike
MYSQL_PASSWORD=your_app_password_here
```

可選設定：

```env
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_WEBSERVER_SECRET_KEY=your_random_secret_key
API_LOG_LEVEL=INFO
API_KEY=your_api_key_here
API_REQUIRE_API_KEY=false
API_RATE_LIMIT_PER_MINUTE=0
ETL_VALIDATION_MODE=strict
```

公開 repo 不應提交真實密碼、GCP VM 資訊或 Secret Manager secret value。

完整 env var inventory、secret handling、demo / production-like 設定界線與 API threat model 見 [`docs/configuration_security.md`](configuration_security.md)。

### 啟動服務

```bash
make up
```

預設服務：

| 服務 | URL / Port |
| --- | --- |
| Airflow UI | `http://localhost:8080` |
| FastAPI docs | `http://localhost:8000/docs` |
| Streamlit dashboard | `http://localhost:8501` |
| MySQL | `localhost:3306` |

檢查容器：

```bash
make ps
make logs
```

停止服務：

```bash
make down
```

## Health、Readiness 與 Metrics

`/health` 是 liveness check。API process 可回應時應回 `200`，並列出資源載入狀態、model lineage 與 warehouse lookup 狀態。

`/ready` 是 inference readiness。正式模式下，模型、scaler、station mapping 與 station metadata 都載入時才應回 `200`。缺任何必要資源時應回 `503`。

`/metrics` 會輸出 Prometheus-style text metrics：

- `youbike_api_requests_total`
- `youbike_api_request_errors_total`
- `youbike_api_request_duration_ms_sum`
- `youbike_api_request_duration_ms_count`
- `youbike_api_request_duration_ms_max`
- `youbike_api_request_duration_seconds_bucket`
- `youbike_api_request_duration_seconds_sum`
- `youbike_api_request_duration_seconds_count`

`/metrics` 本身不納入統計，避免 scrape 行為讓 request count 自我膨脹。Prometheus scrape、PromQL、dashboard panels 與 alert rule 草案見 [`docs/observability.md`](observability.md)。

## API Access Controls

API key 與 rate limit 預設關閉，避免本機 demo 因環境變數不完整而中斷。若要用 production-like 模式展示受保護的 inference API，可設定：

```env
API_KEY=your_api_key_here
API_REQUIRE_API_KEY=true
API_RATE_LIMIT_PER_MINUTE=60
```

受保護 endpoint：

- `GET /stations`
- `POST /predict`
- `POST /stations/risk`

公開檢查 endpoint：

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /metrics`

呼叫受保護 endpoint 時需帶：

```bash
curl -s http://127.0.0.1:8000/stations \
  -H "X-API-Key: your_api_key_here"
```

Streamlit dashboard 的 live mode 也會讀取同一個 `API_KEY`，並在呼叫 FastAPI 時帶 `X-API-Key` header。

錯誤邊界：

- 缺少 API key：`401`
- API key 錯誤：`403`
- `API_REQUIRE_API_KEY=true` 但未設定 `API_KEY`：`503`
- 超過 `API_RATE_LIMIT_PER_MINUTE`：`429`，並回 `Retry-After`

這是單一 API process 內的輕量 guardrail。若要正式公開服務，應改由 API gateway、集中式 rate limiter、正式身份系統與 WAF 承接。

## JSON Log

API log 使用 JSON event 格式。request completion 事件包含：

- `timestamp`
- `service`
- `event`
- `request_id`
- `method`
- `path`
- `status_code`
- `duration_ms`
- `model_version`

範例：

```json
{
  "duration_ms": 1.42,
  "event": "request_completed",
  "method": "GET",
  "model_version": "api-demo-fixtures-v1",
  "path": "/health",
  "request_id": "demo-check",
  "service": "youbike-prediction-api",
  "status_code": 200,
  "timestamp": "2026-06-21T07:46:28.438759+00:00"
}
```

`X-Request-ID` 可由呼叫端指定。Dashboard 或外部 client 回報錯誤時，應先用這個 id 對應 API log。

## Model Artifact Lineage

API response 會回傳：

- `model_version`
- `model_artifact_hash`
- `model_metadata_loaded`
- `model_metadata_generated_at`

目前 served artifact 沒有 `model_metadata.json`，因此 API 會回傳 `legacy-artifact-...` 與 artifact hash，並標示 `model_metadata_loaded: false`。這代表目前已能追到同一組服務資源，但還不是正式 model registry。

如果要替換服務中的模型：

1. 先用 `make train-lstm` 將候選 artifact 輸出到 `.scratch/model_training`。
2. 檢查 `model_metadata.json` 的 train / validation / test 指標與 `model_selection.recommendation`。
3. 只有在明確接受候選模型時，才指定 `--output-dir api/model_files`。
4. 替換後重新啟動 API，檢查 `/health` 與 `/predict` 的 `model_version` / `model_artifact_hash` 是否改變。
5. 保留前一版 artifact，以便手動 rollback。

## API Benchmark

啟動 API demo 後可執行短測：

```bash
make api-benchmark
```

此指令使用 `demo` profile，以 5 個 worker 混合呼叫：

- `GET /ready`
- `POST /predict`
- `POST /stations/risk`

輸出包含 request count、error rate、throughput、p50、p95、p99、max latency 與 pass/watch/fail 判讀。

若要保留較長的容量探測結果：

```bash
make api-load-test
```

此指令使用 `capacity` profile，並輸出 `.scratch/benchmarks/api-capacity.json`。這是本機 benchmark，不代表 production SLO。詳細 profile 與判讀方式見 [`docs/performance_load_test.md`](performance_load_test.md)。

## 常見故障處理

| 現象 | 可能原因 | 檢查方式 | 處理 |
| --- | --- | --- | --- |
| `/health` 200 但 `/ready` 503 | 模型、scaler、station mapping 或 station metadata 未載入 | `curl /ready` 查看 detail | 檢查 `api/model_files` 是否完整，或改用 `make api-demo` 展示 contract |
| `/stations` 回 503 | station metadata 未載入 | `curl /health` 查看 `station_catalog_loaded` | 檢查 `station_info_map.pkl` 或 demo resources |
| `/predict` 回 404 | station 不在 mapping 內 | 檢查 request `station_no` | 用 `/stations` 取得支援站點 |
| `/predict` 回 422 | request payload 不合法 | 查看 response detail | 修正負數車輛、不合理溫度、雨量或 lag window 長度 |
| warehouse fallback 沒有使用 DB | `DB_PASSWORD` 未設定或查不到完整 lag window | 查看 JSON log event | 設定 DB env，或手動提供 `recent_observations` |
| dashboard 無法連 API | `API_BASE_URL` 錯誤或 API 未 ready | 檢查 dashboard env 與 `/ready` | 修正 `API_BASE_URL`，或先用 dashboard demo mode |
| dbt parse/build 失敗 | profiles 或 DB env 不完整 | `analytics/dbt/profiles.yml`、CI log | 複製 profiles example，確認 DB env |

## Rollback

這個 repo 目前沒有正式部署 pipeline；rollback 是手動流程：

1. 確認目前 commit 與上一個可用 commit。
2. 若只是文件或 API demo 問題，優先用新 commit 修復，不做 destructive reset。
3. 若模型 artifact 替換造成問題，還原上一版 `api/model_files` artifact，重啟 API，確認 `model_artifact_hash` 回到預期值。
4. 若 Docker Compose 服務異常，先 `make down`，再 `make up`。
5. Push 前執行 `make validate-config` 與 `make test`；涉及 dbt 時再執行 `DB_PASSWORD=dummy make dbt-parse`。

## CI 驗證

GitHub Actions 會在 push / pull request 時執行：

- Config/security validation：`make validate-config`
- Python tests：`python -m pytest tests/ -v`
- dbt dependencies install
- dbt seed/build，使用一次性 MySQL service

CI 能驗證主要 contract 沒破，但不代表：

- 真實 GCP VM 正在營運。
- 完整歷史資料評估可在 fresh clone 重現。
- 本機 benchmark 是 production SLO。

## English Summary

This runbook documents local demo, full-stack Docker Compose startup, environment variables, health/readiness checks, metrics, JSON request logging, model artifact lineage, benchmark usage, troubleshooting, rollback, and CI validation. The project is a maintained portfolio showcase, not a currently operated production service.
