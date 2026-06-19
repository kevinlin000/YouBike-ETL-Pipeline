# LSTM Evaluation Report

This note records a local evaluation run of `scripts/train_multistation_lstm.py` against the processed YouBike/weather CSV available in the maintainer workspace.

The result is intentionally documented as model evidence, not promoted as a new production artifact. In this run, the Multi-Station LSTM did not beat the current-value naive baseline on the test split.

## Run Context

| Item | Value |
| --- | --- |
| Generated at | `2026-06-19T18:19:01Z` |
| Command | `python scripts/train_multistation_lstm.py --data-path data/processed/.ipynb_checkpoints/youbike_weather_merged-checkpoint.csv --output-dir .scratch/model_training_checkpoint` |
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

| Split | LSTM MAE | Baseline MAE | MAE delta | LSTM RMSE | Baseline RMSE | RMSE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1.001 | 0.777 | -0.224 | 2.105 | 2.173 | 0.068 |
| Validation | 1.119 | 0.742 | -0.377 | 2.275 | 2.246 | -0.029 |
| Test | 1.021 | 0.811 | -0.210 | 1.902 | 1.870 | -0.032 |

`MAE delta` and `RMSE delta` are computed as baseline metric minus LSTM metric. Positive means the LSTM improved over the baseline. Negative means the baseline was better.

## Interpretation

The current-value baseline is strong for this short-horizon task: predicting the next observation as the latest observed bike count is hard to beat when station status changes gradually between adjacent records.

This run shows:

- The LSTM trains and produces stable, API-compatible artifacts.
- The served model architecture is useful as a model-serving prototype.
- The current LSTM configuration should not be presented as outperforming a simple baseline.
- The R-squared result in the original analysis still belongs to regression/lag-feature evidence, not LSTM test performance.

## Portfolio Claim Boundary

Safe claim:

> The project includes a reproducible LSTM training pipeline and a documented baseline comparison. In the current local evaluation, the LSTM prototype did not beat the current-value baseline, so I treat it as model-serving evidence rather than a proven production forecasting model.

Avoid:

> The LSTM achieved strong production forecasting performance.

## Next Modeling Work

The next modeling iteration should focus on improving or reframing the forecasting approach before replacing the served artifacts:

1. Verify the exact sampling interval and target horizon.
2. Add stronger baselines such as seasonal same-time previous-day and rolling averages.
3. Tune the LSTM only after the baseline suite is fixed.
4. Add aligned weather history to the inference path before production-style claims.
5. Preserve future `model_metadata.json` outputs under `docs/` only when the source data and command are clearly documented.

