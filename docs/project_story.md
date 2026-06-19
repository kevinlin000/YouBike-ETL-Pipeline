# Project Story

This document is the shortest end-to-end explanation of the project. Use it when a reviewer asks, "What was this project actually trying to prove?"

## One-Line Thesis

Taipei YouBike imbalance is a station-level, time-dependent operations problem, so the project collects high-frequency station data, analyzes imbalance patterns, and demonstrates how those signals can be served through an API and dashboard.

Chinese version:

> 台北 YouBike 的問題不是全市車輛總數，而是不同站點在不同時間的供需錯置；因此這個專案用高頻站點資料建立分析、預測 prototype、API 與 dashboard，展示如何把資料轉成調度輔助流程。

## Evidence Chain

| Step | Question | Project answer | Evidence in repo |
| --- | --- | --- | --- |
| 1. Operational problem | What problem matters to users? | Some stations run out of bikes or docks even when city-wide averages look acceptable. | `README.md`, `notebooks/01_youbike_analysis.ipynb` |
| 2. Data foundation | What data is needed? | Station status must be captured frequently and stored separately from station metadata. | `dags/youbike_dag.py`, `dags/youbike_transform.py`, `sql/init_schema.sql` |
| 3. Statistical signal | Is recent station state useful? | Regression and lag-style analysis support that recent availability carries predictive signal. | `notebooks/01_youbike_analysis.ipynb`, `README.md` |
| 4. Modeling prototype | How was prediction explored? | A PyTorch Multi-Station LSTM uses recent bike counts, weather inputs, and station embeddings. | `notebooks/05_multistation_lstm.ipynb`, `scripts/train_multistation_lstm.py` |
| 5. Serving layer | How does the model become usable? | FastAPI loads model artifacts, validates requests, and returns station predictions or risk rankings. | `api/app/main.py`, `tests/test_api.py` |
| 6. Decision workflow | How does output become action? | Streamlit turns predictions into single-station checks and multi-station stock-out/full-load ranking. | `dashboard/app.py`, `dashboard/api_client.py`, `docs/images/dashboard_demo_walkthrough.gif` |

## What The ML Part Means

The ML work should be described as a model-serving prototype, not as a finished production forecasting system.

Defensible claim:

> The project packages a PyTorch Multi-Station LSTM prototype into FastAPI artifacts and demonstrates how prediction output can support station-level operational decisions.

Not defensible yet:

> The LSTM is production-grade or has proven strong out-of-sample accuracy on the full historical dataset.

The reported R-squared improvement from about `0.02` to `0.92` belongs to the statistical regression analysis. It supports the decision to collect lag features, but it is not the LSTM test metric.

## Why The Current API Change Matters

The original training flow used sliding windows from historical observations. A serving endpoint that only repeated the current value would not match that shape. The API now supports a 3-row `recent_observations` window and can optionally read the latest three bike counts from MySQL `station_status` when database credentials are configured.

This is a useful engineering improvement because the request contract now matches the model's sequence-input shape. The API response also includes `forecast_horizon` metadata because the legacy `next_hour` field names came from the original demo, while the documented local evaluation uses one next observation as the target horizon. The remaining gap is weather alignment: the warehouse lookup can fetch recent bike counts, but the current schema does not store historical weather features, so automatic lookup still reuses the request's temperature and rain values.

## Interview Explanation

Use this version when you need to explain the project quickly:

> I started from a station-level imbalance problem: users care about whether their origin station has bikes and whether their destination has docks. I built an Airflow and MySQL pipeline to capture station status every 10 minutes, then used statistical analysis to show that recent station state is a strong signal. The LSTM part is a prototype that turns those lag features, weather inputs, and station identity into a forecast. The strongest engineering part is the serving workflow: model artifacts are loaded by FastAPI, requests are validated, Streamlit converts predictions into stock-out and full-load risk ranking, and the API now exposes forecast-horizon metadata instead of relying only on legacy `next_hour` field names. I would not claim the LSTM is production-grade yet; documented local evaluations show the current LSTM does not beat the strongest baseline on either next-observation or approximate one-hour forecasting.

Chinese version:

> 這個專案一開始是在處理站點層級的供需失衡：使用者在意的是出發站有沒有車、目的地有沒有位，而不是全市平均還剩多少車。所以我用 Airflow 和 MySQL 每 10 分鐘收集站點狀態，再用統計分析確認近期站點狀態有預測訊號。LSTM 這段是 prototype，用 lag features、天氣和站點 embedding 做預測；比較值得強調的是後面的服務化流程，FastAPI 載入模型 artifact、驗證 request，Streamlit 再把預測轉成缺車或滿站風險排序，而且 API 現在會回傳 forecast-horizon metadata，不只依賴早期 demo 留下的 `next_hour` 欄位名稱。我不會把它說成 production-grade ML；現在本地評估顯示 LSTM 在下一筆 observation 和近似一小時 horizon 都沒有打敗最強 baseline。

## Report Structure To Use

If this project needs to be presented as a written report, use this order:

1. Problem: station-level imbalance and peak-hour user pain.
2. Data: YouBike station metadata and high-frequency status facts.
3. Pipeline: Airflow ingestion, MySQL schema, validation, and dbt scaffold.
4. Analysis: variance, campus effect, land-use clusters, tail-risk hotspots, and lag-feature signal.
5. Modeling: Multi-Station LSTM prototype and reproducible training script.
6. Serving: FastAPI request contract, artifact loading, and warehouse lag-window lookup.
7. Dashboard: demo mode, single-station prediction, and multi-station risk ranking.
8. Limitations: not production service, not complete LSTM evaluation, weather history not yet aligned.
9. Next step: add aligned weather history and richer lag features before making model-accuracy claims.

## Open Gaps

| Gap | Why it matters | Best next action |
| --- | --- | --- |
| LSTM does not beat the strongest baseline in local next-observation or approximate one-hour runs | The model is useful for serving demonstration, but not yet as an accuracy claim. | Add aligned weather history, richer lag features, and simpler baselines before tuning or replacing the model. |
| Weather history is not in warehouse lookup | Automatic inference only fetches bike-count history, not aligned weather history. | Add a weather observation table or a documented feature-store-style join before claiming production forecasting. |
| Dashboard demo is mocked | It is useful for interviews but not model evidence. | Keep demo mode clearly labeled and link it to the real API contract. |
| dbt layer is lightweight | It validates analytics shape but is not a complete warehouse product. | Expand marts only if targeting analytics engineering roles. |
