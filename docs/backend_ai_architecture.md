# 後端與模型服務架構

本文件說明本專案中資料管線、模型 artifact、FastAPI 服務與 Streamlit dashboard 的關係。

![Backend / AI Application Architecture](images/backend_ai_architecture.svg)

## 架構重點

- **FastAPI 是模型服務核心。** API 負責 request validation、health/readiness 檢查、request id tracing、模型載入狀態、站點支援範圍檢查、forecast horizon metadata，以及單站預測與多站風險排序 endpoint。
- **模型 artifact 是 runtime dependency。** PyTorch 權重、scaler、站點 mapping 與站點 metadata 由 API service 啟動時載入。
- **Streamlit 是資料應用介面。** Dashboard 消費 API 形狀一致的 response，顯示單站預測與多站缺車、滿站風險排序。
- **Airflow 與 MySQL 是資料基礎。** Airflow 負責定期擷取站點狀態，MySQL 保存站點維度表與狀態事實表，提供分析與模型訓練資料來源。
- **固定範例資料模式是展示與本地檢視用途。** 它使用固定站點資料與可重現的模擬推論結果，不代表模型評估結果。
- **CI 覆蓋主要邊界。** 測試涵蓋 ETL transform、API contract、request id response tracing、dashboard client、training metadata 與 dbt scaffold。

## 資料流

1. Airflow 呼叫 YouBike 開放資料 API。
2. ETL 將站點靜態資訊寫入 `station_info`，將站點狀態寫入 `station_status`。
3. Notebook 與訓練程式使用歷史站點狀態和天氣資料進行分析與模型訓練。
4. FastAPI 載入模型 artifact，提供 `/health`、`/ready`、`/predict` 與 `/stations/risk`。
5. Streamlit dashboard 呼叫 API 或 demo client，呈現預測與風險排序結果。

## 邊界說明

- Dashboard 固定範例資料模式只保證介面流程與 response shape 可檢視。
- 模型準確率需以 [`lstm_evaluation_report.md`](lstm_evaluation_report.md) 的 baseline 評估為準。
- 目前 API 的 warehouse fallback 可查近期可借車數，但尚未查詢對齊時間的 weather history。

## English Summary

FastAPI is the model-serving boundary. It loads model artifacts, validates station inputs, exposes health/readiness, prediction, and risk-ranking endpoints, returns forecast-horizon metadata, and attaches request ids for basic tracing. Streamlit consumes the same response shape as a dashboard layer, while Airflow and MySQL provide the data foundation for analysis and model training.
