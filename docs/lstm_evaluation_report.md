# LSTM 評估報告

本文件記錄 `scripts/train_multistation_lstm.py` 在本機 processed YouBike/weather CSV 上的評估結果。這些結果用來界定目前模型證據，不會直接替換 `api/model_files/` 中的服務 artifact。

訓練程式會在 `model_metadata.json` 輸出 `model_selection` 摘要。替換服務 artifact 的標準採保守規則：候選 LSTM 需要在 test split 同時打敗最佳 baseline 的 MAE 與 RMSE。

## 執行背景

| 項目 | 值 |
| --- | --- |
| 產生時間 | `2026-06-19T19:23:29Z` 與 `2026-06-19T19:24:56Z` |
| 下一筆 observation 指令 | `python scripts/train_multistation_lstm.py --data-path data/processed/.ipynb_checkpoints/youbike_weather_merged-checkpoint.csv --output-dir .scratch/model_training_baseline_suite_with_ridge` |
| 近似一小時指令 | `python scripts/train_multistation_lstm.py --data-path data/processed/.ipynb_checkpoints/youbike_weather_merged-checkpoint.csv --output-dir .scratch/model_training_one_hour_horizon_with_ridge --horizon-steps 6` |
| 資料來源 | 本機 notebook checkpoint CSV，未提交到公開 repo |
| Notebook lineage | `notebooks/05_multistation_lstm.ipynb` |
| 站點選擇 | `district-representative` |
| 站點數 | 13 |
| Time steps | 3 observations |
| 評估 horizon | 1 observation 與 6 observations |
| Split | 70% train / 15% validation / 15% test |
| Epochs | 100 |
| Seed | 42 |

本機來源 CSV 有 947,940 筆資料列。訓練程式會從各 district-like group 選一個代表站點。下一筆 observation 實驗產生 5,005 筆 training sequences、1,079 筆 validation sequences、1,092 筆 test sequences；六步 horizon 實驗產生 4,940 筆 training sequences、1,079 筆 validation sequences、1,092 筆 test sequences。

選出的站點資料列中位數取樣間隔為 10 分鐘，25th 到 75th percentile 約為 9.98 到 10.02 分鐘。因此在這份本機資料上，`horizon_steps=6` 可視為近似一小時目標。

## 下一筆 Observation 評估

數值越低越好。

| Split | Model / baseline | N | MAE | RMSE | LSTM MAE delta | LSTM RMSE delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | LSTM | 5,005 | 1.001 | 2.105 |  |  |
| Train | Current value | 5,005 | 0.777 | 2.173 | -0.224 | 0.068 |
| Train | Rolling mean | 5,005 | 1.265 | 2.686 | 0.264 | 0.581 |
| Train | Same-time previous-day | 3,211 | 5.108 | 7.159 | 4.107 | 5.054 |
| Train | Ridge lag regression | 5,005 | 0.986 | 2.143 | -0.014 | 0.038 |
| Validation | LSTM | 1,079 | 1.119 | 2.275 |  |  |
| Validation | Current value | 1,079 | 0.742 | 2.246 | -0.377 | -0.029 |
| Validation | Rolling mean | 1,079 | 1.244 | 2.775 | 0.125 | 0.500 |
| Validation | Same-time previous-day | 1,079 | 5.604 | 7.695 | 4.485 | 5.420 |
| Validation | Ridge lag regression | 1,079 | 1.034 | 2.256 | -0.085 | -0.019 |
| Test | LSTM | 1,092 | 1.021 | 1.902 |  |  |
| Test | Current value | 1,092 | 0.811 | 1.870 | -0.210 | -0.032 |
| Test | Rolling mean | 1,092 | 1.269 | 2.333 | 0.248 | 0.432 |
| Test | Same-time previous-day | 1,092 | 7.261 | 9.347 | 6.240 | 7.445 |
| Test | Ridge lag regression | 1,092 | 0.992 | 1.871 | -0.029 | -0.031 |

`LSTM MAE delta` 與 `LSTM RMSE delta` 的計算方式是 baseline metric 減 LSTM metric。正值代表 LSTM 優於該 baseline；負值代表 baseline 較好。

## 近似一小時 Horizon 實驗

此實驗使用 `--horizon-steps 6`，在本機 10 分鐘間隔資料上約等於一小時。

| Split | Model / baseline | N | MAE | RMSE | LSTM MAE delta | LSTM RMSE delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | LSTM | 4,940 | 2.814 | 3.994 |  |  |
| Train | Current value | 4,940 | 2.869 | 4.970 | 0.056 | 0.976 |
| Train | Rolling mean | 4,940 | 3.055 | 5.094 | 0.241 | 1.099 |
| Train | Same-time previous-day | 3,211 | 5.108 | 7.159 | 2.294 | 3.165 |
| Train | Ridge lag regression | 4,940 | 3.117 | 4.586 | 0.303 | 0.592 |
| Validation | LSTM | 1,079 | 4.101 | 5.787 |  |  |
| Validation | Current value | 1,079 | 3.233 | 5.443 | -0.868 | -0.344 |
| Validation | Rolling mean | 1,079 | 3.528 | 5.621 | -0.573 | -0.166 |
| Validation | Same-time previous-day | 1,079 | 5.604 | 7.695 | 1.503 | 1.908 |
| Validation | Ridge lag regression | 1,079 | 3.832 | 5.365 | -0.269 | -0.422 |
| Test | LSTM | 1,092 | 3.244 | 4.514 |  |  |
| Test | Current value | 1,092 | 2.897 | 4.404 | -0.347 | -0.110 |
| Test | Rolling mean | 1,092 | 3.046 | 4.482 | -0.198 | -0.032 |
| Test | Same-time previous-day | 1,092 | 7.261 | 9.347 | 4.017 | 4.833 |
| Test | Ridge lag regression | 1,092 | 3.034 | 4.280 | -0.210 | -0.234 |

## 結果解讀

短 horizon 任務中，current-value baseline 很強。當相鄰資料列之間變化不大時，直接用最新可借車數預測下一筆狀態，是不容易被複雜模型打敗的基準。

這次評估結果顯示：

- LSTM 可以完成訓練並輸出 API 可載入的 artifact。
- 目前 LSTM 架構可作為模型服務化 prototype。
- 在下一筆 observation 目標下，LSTM 優於 rolling mean 與 same-time previous-day，但未打敗 current-value 或 Ridge lag-regression。
- 在近似一小時目標下，LSTM 仍未打敗 test split 中最強 baseline：MAE 最佳為 current-value，RMSE 最佳為 Ridge lag-regression。
- 原分析中的 R-squared 結果屬於迴歸與 lag-feature 證據，不是 LSTM test performance。

## 目前結論

本專案已具備可重現 LSTM 訓練流程與 baseline suite，但目前 LSTM prototype 沒有打敗最強 baseline。因此，合理結論是「已完成模型 artifact 與服務化流程」，而不是「已得到高準確度預測模型」。

## 後續模型工作

1. 在 `model_selection.recommendation` 顯示候選模型同時打敗最佳 test baseline 的 MAE 與 RMSE 前，不替換服務中的 artifact。
2. 在調整 LSTM 前，先補齊 aligned weather history 與更完整的 lag features。
3. 持續強化簡單 tabular/time-series baseline，避免只增加神經網路複雜度。
4. 將 aligned weather history 補到 API inference path，再討論 production-style forecasting。
5. 未來若保存新的 `model_metadata.json`，需明確記錄資料來源、指令、split 與 baseline 結果。

## English Summary

The local evaluation shows that the LSTM training pipeline is reproducible and can produce API-compatible artifacts, but the current LSTM does not beat the strongest baseline on either the next-observation or approximate one-hour horizon. The model should therefore be treated as model-serving evidence, not as proof of production forecasting accuracy.
