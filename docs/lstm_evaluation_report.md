# LSTM Evaluation Report

This note records local evaluation runs of `scripts/train_multistation_lstm.py` against the processed YouBike/weather CSV available in the maintainer workspace.

The results are intentionally documented as model evidence, not promoted as new production artifacts. The served API artifacts were not replaced.

New training runs write a `model_selection` summary into `model_metadata.json`. The promotion rule is intentionally conservative: a candidate LSTM should beat the best available baseline on the test split for both MAE and RMSE before replacing served artifacts.

## Run Context

| Item | Value |
| --- | --- |
| Generated at | `2026-06-19T19:23:29Z` and `2026-06-19T19:24:56Z` |
| Command | `python scripts/train_multistation_lstm.py --data-path data/processed/.ipynb_checkpoints/youbike_weather_merged-checkpoint.csv --output-dir .scratch/model_training_baseline_suite_with_ridge` |
| One-hour command | `python scripts/train_multistation_lstm.py --data-path data/processed/.ipynb_checkpoints/youbike_weather_merged-checkpoint.csv --output-dir .scratch/model_training_one_hour_horizon_with_ridge --horizon-steps 6` |
| Source data | Local notebook checkpoint CSV, not committed to the public repo |
| Notebook lineage | `notebooks/05_multistation_lstm.ipynb` |
| Station selection | `district-representative` |
| Selected stations | 13 |
| Time steps | 3 observations |
| Forecast horizons evaluated | 1 observation and 6 observations |
| Split | 70% train / 15% validation / 15% test |
| Epochs | 100 |
| Seed | 42 |

The local source CSV contained 947,940 data rows. The training script selected one representative station per district-like grouping. The next-observation run produced 5,005 training sequences, 1,079 validation sequences, and 1,092 test sequences; the six-step run produced 4,940 training sequences, 1,079 validation sequences, and 1,092 test sequences.

The selected-station rows have a median sampling interval of 10 minutes, with the 25th to 75th percentile ranging from roughly 9.98 to 10.02 minutes. For that local dataset, `horizon_steps=6` is a reasonable approximation of a one-hour target.

## Next-Observation Metrics

Lower is better.

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

`LSTM MAE delta` and `LSTM RMSE delta` are computed as baseline metric minus LSTM metric. Positive means the LSTM improved over that baseline. Negative means the baseline was better.

## One-Hour Horizon Experiment

This run used `--horizon-steps 6`, which is approximately one hour on the local 10-minute data.

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

## Interpretation

The current-value baseline is strong for this short-horizon task: predicting the next observation as the latest observed bike count is hard to beat when station status changes gradually between adjacent records.

This run shows:

- The LSTM trains and produces stable, API-compatible artifacts.
- The served model architecture is useful as a model-serving prototype.
- For the next-observation target, the LSTM beats rolling mean and same-time previous-day baselines, but not current value or Ridge lag regression on the test split.
- For the approximate one-hour target, the LSTM still does not beat the best test baselines: current value is best by MAE, and Ridge lag regression is best by RMSE.
- The R-squared result in the original analysis still belongs to regression/lag-feature evidence, not LSTM test performance.

## Portfolio Claim Boundary

Safe claim:

> The project includes a reproducible LSTM training pipeline and a documented baseline suite that includes naive and Ridge lag-regression baselines. In the current local evaluation, the LSTM prototype does not beat the strongest baseline on either the next-observation run or the approximate one-hour run, so I treat it as model-serving evidence rather than a proven production forecasting model.

Avoid:

> The LSTM achieved strong production forecasting performance.

## Next Modeling Work

The next modeling iteration should focus on improving feature history and baselines before replacing the served artifacts:

1. Keep the served artifacts unchanged until `model_selection.recommendation` says the candidate beats the best test baseline on both MAE and RMSE.
2. Add aligned weather history and richer lag features to the training/evaluation dataset before tuning the LSTM further.
3. Improve or expand the simple tabular/time-series baselines before adding more neural-network complexity.
4. Add aligned weather history to the inference path before production-style claims.
5. Preserve future `model_metadata.json` outputs under `docs/` only when the source data and command are clearly documented.
