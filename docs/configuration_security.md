# 設定與安全邊界

這份文件整理 YouBike 專案的環境變數、secret handling、demo / local full stack / production-like 設定差異，以及 API 服務的基本 threat model。

本專案是作品展示，不是正在營運的 production service。這份文件的目標是把安全與設定邊界說清楚，避免把 demo mode、真實資料庫、GCP Secret Manager、Airflow 預設帳密和公開作品集敘事混在一起。

## 執行模式

| 模式 | 啟動方式 | 依賴 | 用途 | 安全注意事項 |
| --- | --- | --- | --- | --- |
| Dashboard demo | `make dashboard-demo` | Python app dependencies | 單獨展示 Streamlit 流程 | 使用固定範例資料，不代表模型結果 |
| API demo | `make api-demo` | Python app dependencies | 展示 FastAPI contract、validation、readiness、metrics、logs | `API_DEMO_MODE=true` 不讀模型檔、不連 DB |
| Local full stack | `make up` | Docker Compose、`.env` | 啟動 Airflow、MySQL、FastAPI、dashboard | 需要本機 `.env`，不可提交真實密碼 |
| Production-like | 手動部署 Docker Compose / VM | MySQL、Secret Manager、network policy | 展示過去部署架構與可維運設計 | 需替換預設密碼、限制 ports、設定 secret 與 log/metrics 收集 |

## Environment Variables

| 變數 | 使用者 | 是否必要 | 是否 secret | 預設 / 範例 | 說明 |
| --- | --- | --- | --- | --- | --- |
| `MYSQL_ROOT_PASSWORD` | Docker Compose MySQL | full stack 必要 | 是 | 無預設 | MySQL root 密碼，只能放本機 `.env` 或 secret store |
| `MYSQL_DATABASE` | Docker Compose / API / Airflow | full stack 必要 | 否 | `youbike_db` | MySQL database 名稱 |
| `MYSQL_USER` | Docker Compose / API / Airflow | full stack 必要 | 否 | `youbike` | App 使用者；正式環境應限制權限 |
| `MYSQL_PASSWORD` | Docker Compose / API / Airflow | full stack 必要 | 是 | 無預設 | App DB 密碼，只能放本機 `.env` 或 secret store |
| `DB_HOST` | ETL / API / dbt | standalone / dbt 可選 | 否 | `127.0.0.1` 或 `mysql-db` | DB host；容器內通常是 `mysql-db` |
| `DB_PORT` | ETL / API / dbt | 可選 | 否 | `3306` | DB port |
| `DB_USER` | ETL / API / dbt | 可選 | 否 | `admin` / `youbike` | standalone job 或 dbt 使用者 |
| `DB_PASSWORD` | ETL / API / dbt | warehouse fallback / dbt 必要 | 是 | 無預設 | ETL standalone、API warehouse fallback、dbt 使用 |
| `DB_NAME` | ETL / API / dbt | 可選 | 否 | `youbike_db` | DB schema/database |
| `GCP_PROJECT_ID` | ETL / Airflow DAG | GCP 部署時可選 | 否 | `youbike-airflow-server` | Secret Manager project id；公開文件不應放真實部署細節 |
| `GCP_SECRET_ID` | ETL / Airflow DAG | GCP 部署時可選 | 否 | `mysql_password` | Secret Manager secret id，不是 secret value |
| `API_TIMEOUT` | ETL / Airflow DAG | 可選 | 否 | `30` | YouBike Open Data API timeout 秒數 |
| `API_RETRIES` | ETL / Airflow DAG | 可選 | 否 | `3` | Extract retry 次數 |
| `API_RETRY_BACKOFF` | ETL / Airflow DAG | 可選 | 否 | `2.0` | Retry backoff 基準秒數 |
| `ETL_VALIDATION_MODE` | ETL / Airflow DAG | 可選 | 否 | `strict` | `strict` 會中止不合法資料；`warn` 只記錄警告 |
| `API_DEMO_MODE` | FastAPI | demo 可選 | 否 | `false` | `true` 時使用固定範例站點與 mock prediction |
| `API_LOG_LEVEL` | FastAPI | 可選 | 否 | `INFO` | JSON log level |
| `DASHBOARD_DEMO_MODE` | Streamlit | demo 可選 | 否 | `false` | `true` 時 dashboard 使用固定範例資料 |
| `API_BASE_URL` | Streamlit | dashboard live mode 必要 | 否 | `http://api:8000` | Dashboard 呼叫 FastAPI 的 base URL |
| `AIRFLOW_USER` | Docker Compose Airflow | full stack 可選 | 否 | `admin` | 本機預設帳號；正式環境不可沿用 |
| `AIRFLOW_PASSWORD` | Docker Compose Airflow | full stack 可選 | 是 | `admin` | 本機預設密碼；正式環境必須替換 |
| `AIRFLOW_WEBSERVER_SECRET_KEY` | Docker Compose Airflow | full stack 可選 | 是 | `secret_key_change_in_production` | Airflow webserver secret key，正式環境必須替換 |
| `DBT_PYTHON` | Makefile | dbt install 可選 | 否 | `python3.11` | 指定安裝 dbt dependencies 的 Python |

## Secret Handling

目前 repo 的 secret handling 原則：

- `.env` 已在 `.gitignore`，真實密碼不應提交。
- `.env.example` 只放 placeholder 與非敏感預設。
- `make validate-config` 會檢查追蹤檔案中的設定與 secret 邊界，CI 也會執行同一個檢查。
- MySQL 密碼在 Docker Compose local full stack 由 `.env` 注入。
- ETL / Airflow DAG 會先看 `DB_PASSWORD`，若未設定則嘗試從 GCP Secret Manager 讀取。
- API warehouse fallback 只有在 `DB_PASSWORD` 存在時才建立 DB engine；缺 DB credentials 時會記錄 `warehouse_lookup_disabled`，不阻擋 API demo。
- README 與 docs 不應揭露真實 GCP VM IP、Secret Manager secret value、真實 DB 密碼或個人 credential。

目前仍未做到：

- 沒有完整 secret rotation 流程。
- 沒有集中式 config module。
- 沒有 Kubernetes / cloud-native secret mount。
- 沒有完整企業級 secret scanning 服務；目前提供的是 repo 內的輕量 config / secret-boundary validation。

這些限制可以在面試時主動說明，不要把作品包裝成完整 production security implementation。

## 自動化檢查

本專案提供一個零新增依賴的檢查：

```bash
make validate-config
```

檢查項目：

- `.env`、`analytics/dbt/profiles.yml`、`analytics/dbt/.user.yml` 不可被 Git 追蹤。
- `.gitignore` 必須保護本機 secret、dbt artifacts 與 local profiles。
- `.env.example` 中列出的環境變數必須在本文件中有說明。
- 追蹤檔案中不可出現常見 private key、GitHub token、AWS access key、Google API key、GCP service account JSON 等高風險 pattern。

這不是完整 secret scanning 產品。它的定位是 portfolio 專案中的 CI guardrail，避免明顯的 secret / config 邊界回歸。

## Demo 與正式設定的界線

Demo mode 的目的：

- 穩定展示 API contract、dashboard flow、request validation、observability。
- 不依賴模型檔、MySQL、Docker Compose 或 GCP。
- 用固定範例資料讓面試展示可重現。

Demo mode 不能主張：

- 真實模型準確率。
- production latency。
- 真實 warehouse fallback 行為。
- 真實調度策略成效。

正式或 production-like 設定應該：

- `API_DEMO_MODE=false`
- `DASHBOARD_DEMO_MODE=false`
- 使用完整 `api/model_files`
- 設定 DB credentials 或明確關閉 warehouse fallback
- 替換 Airflow 預設帳密
- 限制 MySQL、Airflow、API、dashboard 對外暴露的 ports
- 接上 log aggregation、metrics dashboard 與 alert routing

## Trust Boundaries

| 邊界 | 資產 | 主要風險 | 目前控制 | 缺口 |
| --- | --- | --- | --- | --- |
| Public internet -> FastAPI | API availability、模型 artifact、站點 catalog | 無限制流量、payload abuse、error probing | Pydantic validation、readiness、request id、metrics | 無 auth、無 rate limit、無 WAF |
| FastAPI -> MySQL | station status history、DB credentials | credential leak、過度查詢、DB unavailable | DB engine only when `DB_PASSWORD` exists、pool limits、fallback path | 無 query-level timeout 設定、無 least-privilege 文件 |
| Dashboard -> FastAPI | API base URL、使用者輸入 | 指向錯誤 backend、API unavailable | `API_BASE_URL` env、dashboard demo mode、client error handling | 無 authentication、無 CSRF/session model |
| ETL / Airflow -> Open Data API | Raw station data、pipeline availability | 外部 API timeout、schema drift、duplicate data | timeout/retry、strict validation、unique key handling | 無 upstream contract monitoring |
| Airflow / ETL -> MySQL | warehouse tables、DB credentials | bad load、duplicate rows、secret exposure | transform validation、unique constraints、Secret Manager fallback | standalone job 與 DAG config 還未集中 |
| Repo -> Public portfolio | source code、docs、screenshots | accidental secret commit、misleading claims | `.env` ignored、docs 明確 demo/mock limitation、`make validate-config` | 無完整企業級 secret scanning |

## API Threat Model

| Threat | Impact | Current Mitigation | Recommended Next Step |
| --- | --- | --- | --- |
| Invalid payload causes runtime errors | API 500、demo 中斷 | Pydantic validators、422 response tests | 保留 contract tests，新增 fuzz/property tests |
| Unknown station id probes model support | 站點支援範圍外錯誤 | `ensure_station_supported()` 回 404 | 若公開服務需加 auth 或 rate limit |
| High request volume increases latency | API latency 上升、CPU 壓力 | Local benchmark profiles、metrics histogram | 加 rate limiting、process metrics、load test baseline |
| Model artifact mismatch | 推論結果不可追蹤 | `model_version`、artifact hash、metadata loaded flag | 正式 model registry 與 release notes |
| DB credential leak | Warehouse exposure | `.env` ignored、Secret Manager fallback | Secret scanning、least-privilege DB user、rotation runbook |
| Airflow default credentials used outside local demo | Airflow UI 被登入 | `.env.example` 明確提醒替換 | 在 production-like compose profile 禁止預設值 |
| Logs expose sensitive values | Secret leakage | 現有 JSON request logs 不記錄 DB password | 增加 log field allowlist 與 secret redaction test |

## 面試時的說法

可以這樣講：

> 這個作品不是完整 production security project，但我有把設定邊界補清楚。Demo mode 完全不依賴 DB 或模型檔，適合展示 API contract；local full stack 透過 `.env` 注入 MySQL 和 Airflow 設定；ETL / Airflow 可用 GCP Secret Manager 讀 DB password。API 端有 Pydantic validation、readiness、request id、JSON logs、Prometheus metrics 和 benchmark profiles；CI 也會跑 `make validate-config`，避免 `.env`、dbt local profile 或常見 secret pattern 被提交。若要真的上線，我會先補 auth/rate limit、企業級 secret scanning、least-privilege DB user、正式 alert routing 和 secret rotation。

## English Summary

This document defines the configuration and security boundaries for the portfolio project. It lists environment variables, distinguishes demo mode from local full-stack and production-like settings, documents secret-handling rules, and provides a lightweight API threat model. The project has basic validation, readiness, request tracing, metrics, and Secret Manager fallback, but it should not be described as a complete production security implementation.
