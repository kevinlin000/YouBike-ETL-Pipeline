# LSTM Evaluation Report

This note records a local evaluation run of `scripts/train_multistation_lstm.py` against the processed YouBike/weather CSV available in the maintainer workspace.

The result is intentionally documented as model evidence, not promoted as a new production artifact. In this run, the Multi-Station LSTM did not beat the current-value naive baseline on the test split.

## Run Context

| Item | Value |
| --- | --- |
| Generated at | `2026-06-19T18:19:01Z` |
| Command | `python scripts/train_multistation_lstm.py --data-path data/processed/.ipynb_checkpoints/youbike_weather_merged-checkpoint.csv --output-dir .scratch/model_training_baseline_suite` |
| Source data | Local notebook checkpoint CSV, not committed to the public repo |
| Notebook lineage | `notebooks/05_multistation_lstm.ipynb` |
| Station selection | `district-representative` |
| Selected stations | 13 |
| Time steps | 3 observations |
| Forecast horizon | 1 observation |
| Split | 70% train / 15% validation / 15% test |
| Epochs | 100 |
| Seed | 42 |

The local source CSV contained 947,940 data rows. The training script selected one representative station per district-like grouping, which produced 5,005 training sequences, 1,079 validation sequences, and 1,092 test sequences.

## Metrics

Lower is better.

| Split | Model / baseline | N | MAE | RMSE | LSTM MAE delta | LSTM RMSE delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | LSTM | 5,005 | 1.001 | 2.105 |  |  |
| Train | Current value | 5,005 | 0.777 | 2.173 | -0.224 | 0.068 |
| Train | Rolling mean | 5,005 | 1.265 | 2.686 | 0.264 | 0.581 |
| Train | Same-time previous-day | 3,211 | 5.108 | 7.159 | 4.107 | 5.054 |
| Validation | LSTM | 1,079 | 1.119 | 2.275 |  |  |
| Validation | Current value | 1,079 | 0.742 | 2.246 | -0.377 | -0.029 |
| Validation | Rolling mean | 1,079 | 1.244 | 2.775 | 0.125 | 0.500 |
| Validation | Same-time previous-day | 1,079 | 5.604 | 7.695 | 4.485 | 5.420 |
| Test | LSTM | 1,092 | 1.021 | 1.902 |  |  |
| Test | Current value | 1,092 | 0.811 | 1.870 | -0.210 | -0.032 |
| Test | Rolling mean | 1,092 | 1.269 | 2.333 | 0.248 | 0.432 |
| Test | Same-time previous-day | 1,092 | 7.261 | 9.347 | 6.240 | 7.445 |

`LSTM MAE delta` and `LSTM RMSE delta` are computed as baseline metric minus LSTM metric. Positive means the LSTM improved over that baseline. Negative means the baseline was better.

## Interpretation

The current-value baseline is strong for this short-horizon task: predicting the next observation as the latest observed bike count is hard to beat when station status changes gradually between adjacent records.

This run shows:

- The LSTM trains and produces stable, API-compatible artifacts.
- The served model architecture is useful as a model-serving prototype.
- The current LSTM beats rolling mean and same-time previous-day baselines.
- The current LSTM still does not beat the strongest current-value baseline on the test split.
- The R-squared result in the original analysis still belongs to regression/lag-feature evidence, not LSTM test performance.

## Portfolio Claim Boundary

Safe claim:

> The project includes a reproducible LSTM training pipeline and a documented baseline suite. In the current local evaluation, the LSTM prototype beats rolling mean and same-time previous-day baselines, but it does not beat the strongest current-value baseline, so I treat it as model-serving evidence rather than a proven production forecasting model.

Avoid:

> The LSTM achieved strong production forecasting performance.

## Next Modeling Work

The next modeling iteration should focus on reframing the forecasting target before replacing the served artifacts:

1. Verify the exact sampling interval and target horizon.
2. Decide whether the operational target should be next observation, next hour, or a dispatch-specific window.
3. Re-evaluate the baseline suite against that horizon before tuning the LSTM.
4. Add aligned weather history to the inference path before production-style claims.
5. Preserve future `model_metadata.json` outputs under `docs/` only when the source data and command are clearly documented.
