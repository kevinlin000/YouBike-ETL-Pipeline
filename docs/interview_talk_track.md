# Interview Talk Track

Use this as a concise script for explaining the project in interviews. The goal is to present the repo as a data engineering and analytics portfolio project, not as a currently operated production service.

## 30-Second Version

This project analyzes Taipei YouBike 2.0 station imbalance and turns it into an end-to-end data application. I built scheduled station-status ingestion, normalized MySQL tables, statistical analysis notebooks, a PyTorch LSTM prediction service behind FastAPI, and a Streamlit dashboard for single-station prediction and multi-station risk ranking. The current dashboard has a deterministic demo mode, so I can show the product flow without needing Docker Compose, model files, or live cloud services.

## 2-3 Minute Version

Start with the operational problem: YouBike does not only have a total supply problem. The harder problem is that bikes and empty docks are unevenly distributed across stations and time windows. A city-wide average can look healthy while individual stations are empty or full during commute peaks.

The data engineering layer ingests YouBike station status on a fixed schedule through Airflow and stores the result in MySQL. Static station metadata is separated from high-frequency station status facts, so the warehouse can represent both station identity and time-series availability.

The analysis layer uses descriptive statistics, t-tests, ANOVA, clustering, chi-square testing, and regression to explain why station-level monitoring matters. One of the key findings is that lag-style temporal features are much more useful than static location-only features, which supports the decision to collect higher-frequency status data.

The application layer wraps the LSTM inference path with FastAPI. The dashboard turns raw predictions into a decision-support workflow: a user can inspect a single station or rank multiple stations by stock-out and full-load risk. For interviews, demo mode uses deterministic mock data to show the product flow safely. It should be described as a workflow demo, not as model-performance evidence.

When discussing the machine-learning portion, be precise: the regression R-squared improvement shows that recent station state has predictive signal, while the LSTM is a served prototype that still needs reproducible training and out-of-sample evaluation before it should be called production forecasting.

## Demo Flow

1. Run `make dashboard-demo`.
2. Open the Streamlit URL, normally `http://localhost:8501`.
3. Point out the sidebar demo-mode state. Say that the dashboard is not depending on FastAPI, model files, or Docker Compose in this mode.
4. On the single-station tab, run one prediction and explain that the UI converts current station state plus weather inputs into a one-hour availability estimate.
5. On the multi-station tab, run risk ranking and focus on the priority cards. Explain that this is the operational translation layer: predictions become ranked actions such as replenishing bikes or moving bikes out.
6. Close with the architecture: Airflow and MySQL build the data foundation, notebooks establish analytical evidence, FastAPI serves inference, and Streamlit demonstrates the decision workflow.

## Good Claims

- "This is a portfolio showcase of the full path from ingestion to decision-support UI."
- "The dashboard demo mode is deterministic and intentionally independent from live services."
- "The model-serving interface is represented through FastAPI endpoints and request validation."
- "The project preserves historical deployment evidence through GCP, Docker, Airflow, and monitoring screenshots."
- "The strongest engineering story is connecting high-frequency station data to station-level operational decisions."
- "The LSTM portion is best described as a prototype model-serving layer, not a completed model-evaluation claim."

## Claims To Avoid

- Do not say the service is currently running in production.
- Do not present demo-mode output as real model accuracy.
- Do not say the LSTM achieved R-squared 0.92; that number belongs to the regression analysis.
- Do not imply `/predict` currently queries a real historical lag window from the warehouse.
- Do not imply the current dbt layer is a full enterprise warehouse.
- Do not claim complete MLOps coverage.
- Do not oversell Spark, Kafka, Kubernetes, or data lake experience from this repo.

## Likely Follow-Up Questions

### Why Airflow?

The ingestion problem is scheduled, repeatable, and operationally observable. Airflow provides scheduling, retries, task logs, and a familiar orchestration model for data pipelines.

### Why separate `station_info` and `station_status`?

Station metadata changes slowly, while availability changes frequently. Separating dimension-like station data from status facts avoids duplicating station metadata on every observation and makes time-series analysis cleaner.

### What does the LSTM add?

The analysis suggested that recent station availability carries predictive signal. The LSTM extends that idea into a PyTorch sequence-model prototype with station embeddings and weather features. The strongest thing to emphasize is the model-serving workflow: notebook artifacts are packaged, loaded by FastAPI, validated through API contracts, and translated by the dashboard into operational risk labels.

The careful limitation is that this is not yet a rigorously evaluated forecasting system. The repo now includes a reproducible training script with time-based train / validation / test splitting and metadata output, but full metrics still need to be generated from the complete processed dataset and compared against a naive baseline.

### What would you improve next?

For ML engineering quality, I would rerun the training script on the full processed dataset, preserve the generated `model_metadata.json`, add a naive baseline, and then replace the API shortcut with a real warehouse-backed lag window. For analytics engineering roles, I would expand the dbt marts and connect the warehouse-backed metrics to the dashboard. The shared transform module and pre-load validation gate are already in place, so the next step depends on which job family I want to target.

### What is the biggest limitation?

The biggest limitation is the ML evaluation path. The dashboard demo mode is intentionally mocked for presentation, and the live `/predict` path currently builds a short sequence from the current state instead of querying a real recent-history window. For production forecasting, the API should query real lag windows from the warehouse, and the training script should be rerun on the full dataset with documented out-of-sample metrics and a baseline comparison.

For deeper ML context, use [`docs/ml_modeling_audit.md`](ml_modeling_audit.md).
