# Reviewer Quickstart

這份文件是給面試官、招聘方或技術 reviewer 的快速閱讀路線。目標是在很短時間內看懂這個 repo 想展示什麼，而不是要求 reviewer 從頭讀完整份 README。

English short version: start with the backend / AI application story, then inspect the FastAPI contracts, dashboard demo flow, and ML claim boundaries. This is a portfolio showcase, not a currently operated production service.

## 30 秒定位

這是一個以 YouBike 站點資料為題材的後端 / AI 應用作品。

主要展示的是：

- 用 Airflow 和 MySQL 建立高頻站點資料基礎。
- 用 PyTorch LSTM prototype 產生模型 artifact。
- 用 FastAPI 包裝 model-serving boundary、request validation 和 risk-ranking API。
- 用 Streamlit demo 展示單站預測與多站調度風險排序。
- 用 baseline evaluation 誠實界定目前 LSTM 還不是 production forecasting model。

一句話版本：

> 這個專案的重點不是炫耀模型準確率，而是展示我能把資料、模型 artifact、API contract 和 decision workflow 串成可測試、可展示的 AI application backend。

## 3 分鐘閱讀路線

如果只想快速判斷作品集價值，照這個順序看：

1. 先看 [`README.md`](../README.md) 的專案摘要、核心成果和 backend / AI 架構圖。
2. 看 dashboard GIF 和 [`docs/demo_walkthrough.md`](demo_walkthrough.md)，理解 demo 如何對應 `POST /predict` 和 `POST /stations/risk`。
3. 看 [`docs/api_contract_walkthrough.md`](api_contract_walkthrough.md)，確認 API validation、readiness boundary、forecast horizon metadata 和 risk-ranking response shape。
4. 看 [`docs/ml_modeling_audit.md`](ml_modeling_audit.md) 和 [`docs/lstm_evaluation_report.md`](lstm_evaluation_report.md)，確認模型 claim 沒有過度包裝。

這條路線最適合 Backend Engineer 或 AI Application Engineer 職缺。

## 10 分鐘深讀路線

如果要更完整評估工程能力，建議依序看：

| 想評估的能力 | 建議閱讀 |
| --- | --- |
| 專案主線 | [`README.md`](../README.md), [`docs/project_story.md`](project_story.md) |
| Backend / AI application framing | [`docs/backend_ai_positioning.md`](backend_ai_positioning.md), [`docs/backend_ai_architecture.md`](backend_ai_architecture.md) |
| API contract | [`api/app/main.py`](../api/app/main.py), [`tests/test_api.py`](../tests/test_api.py), [`docs/api_contract_walkthrough.md`](api_contract_walkthrough.md) |
| Dashboard client logic | [`dashboard/api_client.py`](../dashboard/api_client.py), [`dashboard/app.py`](../dashboard/app.py), [`tests/test_dashboard_client.py`](../tests/test_dashboard_client.py) |
| ETL and warehouse modeling | [`dags/youbike_transform.py`](../dags/youbike_transform.py), [`sql/init_schema.sql`](../sql/init_schema.sql), [`analytics/dbt`](../analytics/dbt) |
| ML evaluation boundary | [`scripts/train_multistation_lstm.py`](../scripts/train_multistation_lstm.py), [`docs/lstm_evaluation_report.md`](lstm_evaluation_report.md) |

## 這個專案可以主張什麼

可以主張：

- 能設計 FastAPI model-serving endpoints，並處理 validation、readiness 和 unsupported station boundary。
- 能把 prediction output 轉成 operational risk ranking，而不是只停在模型數字。
- 能用 deterministic demo mode 讓作品穩定展示，同時保留 API-shaped response。
- 能補 baseline evaluation，誠實判斷 LSTM prototype 是否值得升級成正式模型。
- 能用 CI 覆蓋 ETL、API、dashboard client、dbt scaffold 和 model-training metadata。

不應該主張：

- 這是正在營運的 production service。
- LSTM 已經被證明是高準確率 production forecaster。
- Streamlit 是正式 production frontend。
- Demo mode output 是模型評估結果。

## 面試展示順序

建議順序：

1. 用 20 秒說明這是 backend / AI application portfolio。
2. 開 README 的 backend / AI architecture diagram。
3. 展示 dashboard demo mode 的單站預測。
4. 展示多站風險排序，強調 prediction 到 suggested action。
5. 回到 FastAPI endpoint 和 tests，說明 API contract。
6. 最後說明 LSTM 沒有打敗最強 baseline，所以目前定位是 model-serving prototype。

完整講稿在 [`docs/demo_walkthrough.md`](demo_walkthrough.md)。

## 中文與英文文件分工

中文是主要敘事語言，因為這個 repo 主要用於中文面試和口頭展示。英文保留為快速掃描、履歷關鍵字和非中文 reviewer 的輔助材料。

文件策略詳見 [`docs/documentation_language_strategy.md`](documentation_language_strategy.md)。
