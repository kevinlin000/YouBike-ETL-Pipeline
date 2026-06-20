# 專案脈絡

這份文件整理 YouBike 專案從問題定義、資料管線、統計分析、模型 prototype 到 API 與 dashboard 的完整脈絡。

## 核心問題

YouBike 的營運問題不只是全市車輛總數，而是站點與時間之間的供需錯置。使用者實際遇到的痛點通常是：

- 出發站沒有車可借。
- 目的地沒有空位可還。
- 尖峰時間部分站點快速耗盡，其他站點仍有餘裕。

因此，本專案以站點層級的高頻狀態資料為基礎，分析缺車與滿站風險，並建立一個可由 API 與 dashboard 呈現的調度輔助流程。

## 工程流程

| 階段 | 問題 | 專案做法 | 主要檔案 |
| --- | --- | --- | --- |
| 資料擷取 | 如何取得足夠細的站點狀態？ | Airflow 每 10 分鐘擷取 YouBike 開放資料。 | `dags/youbike_dag.py`, `etl_job.py` |
| 資料建模 | 如何區分站點靜態資訊與動態狀態？ | MySQL 使用 `station_info` 維度表與 `station_status` 事實表。 | `sql/init_schema.sql`, `dags/youbike_transform.py` |
| 統計分析 | 哪些站點或區域有供需失衡？ | 使用變異係數、t 檢定、ANOVA、卡方檢定、分群與迴歸分析。 | `notebooks/01_youbike_analysis.ipynb` |
| 預測 prototype | 如何驗證近期狀態是否能形成預測特徵？ | 使用多站點 LSTM，整合近期車輛數、天氣與站點 embedding。 | `notebooks/05_multistation_lstm.ipynb`, `scripts/train_multistation_lstm.py` |
| 服務化 | 如何讓模型輸出被應用端使用？ | FastAPI 載入模型 artifact，提供單站預測與多站風險排序 API。 | `api/app/main.py`, `tests/test_api.py` |
| 操作介面 | 如何把預測轉成可讀的決策資訊？ | Streamlit dashboard 顯示單站預測與缺車、滿站風險排序。 | `dashboard/app.py`, `dashboard/api_client.py` |

## 統計分析的意義

原始報告中的統計分析主要支撐兩件事：

1. 站點層級資料比全市平均更能描述使用者痛點。
2. 近期站點狀態具有預測訊號，因此高頻資料擷取有工程價值。

其中迴歸分析曾顯示加入 lag 類特徵後，解釋力明顯提高。這個結果應解讀為「近期站點狀態是重要特徵」，不是 LSTM 模型準確率。

## 模型部分的定位

目前機器學習部分應定位為模型服務化 prototype。專案已完成：

- PyTorch 多站點 LSTM prototype。
- 模型權重、scaler、站點 mapping 等 artifact 輸出。
- 可重現訓練程式。
- current-value、rolling mean、same-time previous-day、Ridge lag-regression baseline。
- FastAPI 推論 endpoint 與 dashboard 消費模型輸出。

目前不應主張 LSTM 已經是正式可用的高準確度預測模型。本地評估顯示，LSTM 在下一筆 observation 與近似一小時 horizon 下，都沒有打敗最強 baseline。細節見 [`ml_modeling_audit.md`](ml_modeling_audit.md) 與 [`lstm_evaluation_report.md`](lstm_evaluation_report.md)。

## API 與 Dashboard 的價值

本專案較完整的工程價值在於服務化流程：

- FastAPI 負責 request validation、模型載入狀態、站點支援範圍與錯誤回應。
- `/predict` 提供單站預測。
- `/stations/risk` 將模型輸出轉成缺車、滿站、車輛偏低、空位偏低等風險標籤。
- Dashboard 以相同 response shape 呈現單站預測與多站風險排序。

這讓模型輸出不只停在數字，而是能轉成可排序、可閱讀的調度輔助資訊。

## 目前限制

| 限制 | 影響 | 後續方向 |
| --- | --- | --- |
| LSTM 未打敗最強 baseline | 模型可作為服務化 prototype，不適合作為準確率主張。 | 先補齊特徵與 baseline，再考慮替換服務中的 artifact。 |
| warehouse 查詢缺少 weather history | API 自動查詢近期狀態時，只能取得車輛數，無法取得對齊時間的天氣特徵。 | 增加天氣觀測表或 feature join。 |
| Dashboard 固定範例資料模式 | 可穩定檢視介面流程，但不是模型評估結果。 | 在介面與文件中清楚標示用途，模型表現仍以 evaluation report 為準。 |
| dbt layer 仍是輕量 scaffold | 可驗證 analytics model 形狀，但不是完整企業資料倉儲。 | 若目標轉向 analytics engineering，再擴充 marts 與資料品質規則。 |

## English Summary

This project studies station-level YouBike imbalance using high-frequency station data. It includes Airflow ingestion, MySQL warehouse tables, statistical analysis, a PyTorch multi-station LSTM prototype, FastAPI model-serving endpoints, and a Streamlit decision-support dashboard. The LSTM is currently best described as a model-serving prototype because local evaluations do not beat the strongest baseline. The strongest engineering contribution is the path from station data and model artifacts to validated APIs and operational risk ranking.
