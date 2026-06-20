# 機器學習脈絡與限制

這份文件整理本專案機器學習部分目前完成的內容、可支持的技術結論，以及尚未完成的限制。

## 現階段定位

本專案已完成 PyTorch 多站點 LSTM prototype，並將模型 artifact 接到 FastAPI 與 Streamlit dashboard。這代表專案具備模型服務化流程，但不代表目前 LSTM 已經是正式可用的高準確度預測模型。

目前較準確的定位是：

- 統計分析顯示近期站點狀態具有預測訊號。
- LSTM prototype 驗證了多站點序列模型的實作與 artifact 輸出流程。
- FastAPI 驗證了模型載入、request validation、推論 endpoint 與風險排序 response。
- 本地 baseline 評估顯示，目前 LSTM 沒有打敗最強 baseline，因此不能把模型準確率作為主要結論。

## Notebook 與 Script 關係

| 檔案 | 角色 | 說明 |
| --- | --- | --- |
| `notebooks/01_youbike_analysis.ipynb` | 統計分析 | 使用描述統計、t 檢定、ANOVA、卡方檢定、分群與迴歸分析站點失衡。 |
| `notebooks/03_data_merge.ipynb` | 特徵資料整理 | 將 YouBike 站點資料與天氣資料合併成訓練資料。 |
| `notebooks/04_lstm_prediction.ipynb` | 單站 LSTM prototype | 先以單一站點測試短序列與天氣特徵。 |
| `notebooks/05_multistation_lstm.ipynb` | 多站點 LSTM lineage | 建立目前 API 使用的多站點模型 artifact lineage。 |
| `scripts/train_multistation_lstm.py` | 可重現訓練程式 | 將 notebook 流程整理成 CLI，輸出 artifact、metadata 與 baseline 指標。 |

這些檔案的證據層級不同。統計分析支持特徵方向；LSTM notebook 支持 prototype 訓練；訓練程式支持可重現評估；FastAPI 與 dashboard 支持模型服務化流程。

## 目前服務中的模型

`api/app/main.py` 會從 `api/model_files/` 載入以下 artifact：

- `youbike_lstm_multistation.pth`
- `scaler.pkl`
- `station_mapping.pkl`
- `station_info_map.pkl`

模型架構為 `MultiStationLSTM`：

- 數值特徵：`bikes_available`、`temperature`、`rain`、`Rain_Cat`
- 站點表示：station ID embedding，維度為 5
- LSTM hidden size：64
- dropout：0.2
- output：預測未來 `bikes_available`

原 notebook 選出 13 個代表站點，並使用 `TIME_STEPS = 3` 建立 sliding window。這裡的 3 代表 3 筆 observation，不等於 3 小時；若原始資料約每 10 分鐘一筆，3 筆 observation 約為 30 分鐘。

## 現有證據支持的結論

目前 repo 可以支持以下結論：

- 高頻站點狀態資料有分析價值，因為 lag 類特徵在統計分析中具有明顯訊號。
- 專案已建立 PyTorch LSTM prototype，並能輸出 FastAPI 可載入的模型 artifact。
- 多站點 LSTM 訓練流程已整理成 CLI，包含 train/validation/test split、metadata 輸出與 baseline suite。
- Baseline 包含 current-value、rolling mean、same-time previous-day 與 Ridge lag-regression。
- 本地 checkpoint-data 評估顯示，目前 LSTM 應定位為服務化 prototype，而不是優於 baseline 的正式預測模型。
- API 與 dashboard 能將模型輸出轉成 `stock_out`、`full_load`、`low_supply`、`low_dock` 等風險標籤。

## 現有證據不支持的結論

目前 repo 不應主張：

- LSTM 已具備正式 production forecasting 品質。
- LSTM 已在完整公開資料上被 fresh clone 重現評估。
- LSTM 已打敗最強 baseline。
- API 會自動查詢完整的歷史特徵視窗與歷史天氣。
- 模型支援台北所有 YouBike 站點。

原報告中的 R-squared 提升屬於統計迴歸分析，應解讀為 lag features 有價值，不應拿來當作 LSTM test performance。

## API 推論缺口

訓練資料使用歷史 sliding window。FastAPI `/predict` 現在支援手動傳入 `recent_observations`，讓 request 形狀能對齊模型需要的 3 筆近期觀測。

若 request 未提供 `recent_observations`，且 DB credentials 存在，API 會嘗試從 MySQL `station_status` 讀取最近 3 筆 `bikes_available`。目前缺口是 warehouse 沒有儲存對齊時間的 weather history，因此自動查詢路徑只能沿用 request 中的當前 `temperature` 與 `rain`。

API 仍保留 `predicted_bikes_next_hour` 與 `predicted_spaces_next_hour` 欄位名稱，原因是相容早期 dashboard demo。新的 response 另外加入 `forecast_horizon` metadata，避免將欄位名稱誤解成已完成驗證的一小時預測。

## 後續建議

1. 保留 [`lstm_evaluation_report.md`](lstm_evaluation_report.md) 作為目前模型評估邊界。
2. 補齊 weather history，讓訓練與推論都能使用對齊時間的天氣特徵。
3. 擴充 lag features，並先改善簡單 baseline。
4. 只有在新模型於 test split 同時打敗最佳 baseline 的 MAE 與 RMSE 後，才替換 `api/model_files/` 中的服務 artifact。
5. 未來若保存新的 `model_metadata.json` 結果，需同時記錄資料來源、指令、split 設定與 baseline 結果。

## English Summary

The machine-learning component is best understood as a model-serving prototype. The project includes statistical feature evidence, a PyTorch multi-station LSTM lineage, a reproducible training script, baseline metrics, FastAPI inference endpoints, and a dashboard workflow. Local evaluations show that the current LSTM does not beat the strongest baseline on the tested horizons, so model accuracy should not be the main claim until weather history, lag features, and baseline comparisons are improved.
