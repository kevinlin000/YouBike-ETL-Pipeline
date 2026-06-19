# ML Modeling Audit

This note clarifies what the machine-learning part of the project currently does, what evidence it provides, and what should not be claimed from it.

## Bottom Line

The strongest defensible claim is:

> The project includes a PyTorch Multi-Station LSTM prototype, trained from notebook-generated YouBike and weather features, then packaged as FastAPI inference artifacts and surfaced through a Streamlit decision-support dashboard.

The current repository does not yet support a stronger claim such as "the LSTM is a production-grade, rigorously evaluated forecasting model." The notebooks show training loss and model-serving integration, and `scripts/train_multistation_lstm.py` now provides a reproducible training CLI. Full metrics still require rerunning the script where the complete processed CSV is available.

## Mental Model

The ML storyline has four separate parts:

1. Statistical analysis asks whether station availability has time-dependent signal.
2. The single-station LSTM notebook tests the modeling idea on one station.
3. The multi-station LSTM notebook generalizes the prototype with station embeddings and weather features.
4. `scripts/train_multistation_lstm.py` turns that notebook flow into a reproducible training command with artifact metadata.
5. FastAPI and Streamlit turn the model artifact into a product-like workflow.

These parts are related, but they are not the same evidence. Regression supports the feature idea. The LSTM notebook supports prototype training and artifact generation. The API and dashboard support model-serving and decision-support workflow.

## Notebook Lineage

| Notebook | Role | What it contributes |
| --- | --- | --- |
| `notebooks/01_youbike_analysis.ipynb` | Statistical analysis | Explains station imbalance with descriptive statistics, t-tests, ANOVA, chi-square tests, clustering, and regression. |
| `notebooks/03_data_merge.ipynb` | Feature dataset preparation | Merges YouBike station records with weather data into the processed training dataset. |
| `notebooks/04_lstm_prediction.ipynb` | Single-station LSTM prototype | Tests whether short availability sequences and weather features can be used in a PyTorch LSTM for one station. |
| `notebooks/05_multistation_lstm.ipynb` | Current served model lineage | Builds the original multi-station model artifacts used by `api/app/main.py`. |
| `scripts/train_multistation_lstm.py` | Reproducible training CLI | Recreates the multi-station training flow, writes artifacts, and saves `model_metadata.json`. |

## Current Served Model

`api/app/main.py` loads these artifacts from `api/model_files/`:

- `youbike_lstm_multistation.pth`
- `scaler.pkl`
- `station_mapping.pkl`
- `station_info_map.pkl`

The served architecture is `MultiStationLSTM`:

- numeric input features: `bikes_available`, `temperature`, `rain`, `Rain_Cat`
- station representation: station ID embedding with dimension 5
- LSTM hidden size: 64
- dropout: 0.2
- output: predicted future `bikes_available`

The training notebook selects 13 representative stations, one per district-like grouping, and creates sliding windows with `TIME_STEPS = 3`. The target is the next row's scaled `bikes_available` value.

Important nuance: the notebook comments describe this as "past 3 hours", but the implementation is actually "past 3 observations." If the source records are sampled every 10 minutes, three observations represent roughly 30 minutes, not three hours. This should be corrected before making time-horizon claims.

## What The Existing Evidence Supports

The repository supports these claims:

- High-frequency station data has value because the statistical analysis shows strong temporal dependence through lag-style regression features.
- A PyTorch LSTM prototype was built for station-level bike availability forecasting.
- The model was packaged into FastAPI-compatible artifacts.
- The multi-station training flow now has a CLI with time-based train / validation / test splits, current-value naive baseline metrics, and metadata output.
- The API and dashboard demonstrate how model output can be converted into operational risk labels such as `stock_out`, `full_load`, `low_supply`, and `low_dock`.
- The dashboard demo mode is useful for interview presentation because it shows the workflow without requiring live model services.

## What The Existing Evidence Does Not Prove

The repository does not currently prove:

- Out-of-sample LSTM accuracy on the full historical dataset unless `scripts/train_multistation_lstm.py` is rerun with that data and the generated metadata is preserved.
- Production forecasting quality.
- That the LSTM beats the current-value baseline on the full historical dataset unless `scripts/train_multistation_lstm.py` is rerun with that data and `lstm_vs_baseline` is reviewed.
- That the API endpoint automatically queries a complete historical feature window with weather history.
- That all Taipei YouBike stations are supported by the trained model.

The reported R-squared improvement from roughly `0.02` to `0.92` belongs to the regression analysis in the statistical notebook. It should be described as evidence that lag features are valuable, not as LSTM performance.

## API Inference Gap

The training notebook creates real sliding windows from historical rows. The FastAPI `/predict` path accepts optional `recent_observations`, so callers can provide the same 3-row lag-window shape used by the training flow. When request history is omitted and DB credentials are available, the API attempts to load the latest three `bikes_available` rows from MySQL `station_status`.

This closes the biggest serving-shape gap, but it is still not complete production forecasting. The current warehouse table does not store weather history, so the automatic lookup uses historical bike counts with the request's current temperature and rain values. A production-grade endpoint should query both recent station observations and aligned weather features before running inference.

## Interview-Safe Explanation

Use this phrasing:

> I treated the ML part as a prototype model-serving layer. The analysis showed that recent station state is important, so I built a PyTorch LSTM with station embeddings and weather features, saved the artifacts, and served them through FastAPI. I later converted the notebook flow into a reproducible training script with metadata output and a current-value naive baseline. The API can now accept a 3-row recent-observation window and can read recent bike counts from the warehouse when DB credentials are configured. The remaining limitation is that I still need to rerun training on the full dataset and add weather-history alignment before I would call it production forecasting.

Chinese version:

> 我當時 ML 這段不是在做完整 production ML，而是先用統計分析確認近期站點狀態有預測訊號，再用 PyTorch LSTM 做一個多站點預測 prototype。後面把模型權重、scaler、站點 mapping 存成 artifact，讓 FastAPI 可以載入推論，Streamlit 再把預測結果轉成缺車或滿站風險排序。現在 repo 已經補了可重現 training script、metadata 輸出、current-value naive baseline，也讓 API 可以接 3 筆近期觀測值；如果有 DB credentials，API 也會查 warehouse 最近 3 筆可借車數。剩下限制是 weather history 還沒進 warehouse lookup，所以還不能說是完整 production forecasting。

Avoid this phrasing:

> The LSTM achieved R-squared 0.92.

Also avoid:

> The API predicts all Taipei stations accurately in production.

## Recommended Upgrade Path

1. Rerun `make train-lstm` where `data/processed/youbike_weather_merged.csv` is available.
2. Preserve the generated `model_metadata.json` and summarize the out-of-sample metrics in docs.
3. Use `metrics.lstm_vs_baseline` to decide whether the model improves on the current-value baseline.
4. Add weather-history alignment to the warehouse-backed inference path.
5. Add a small evaluation report under `docs/` so the README can link to real model metrics.

## Current Portfolio Positioning

For this portfolio, the ML story is still useful, but the emphasis should be:

- data product thinking,
- feature engineering from high-frequency station data,
- model artifact packaging,
- FastAPI inference integration,
- dashboard translation from predictions to operational actions.

Do not make model-accuracy the center of the story until the evaluation path above is implemented.
