# Backend / AI Architecture

This diagram reframes the project for backend and AI application engineering roles.

![Backend / AI Application Architecture](images/backend_ai_architecture.svg)

## How To Read It

- **FastAPI is the backend core.** It owns request validation, model readiness checks, station support checks, forecast-horizon metadata, and the prediction/risk-ranking API contracts.
- **Streamlit is a client/demo layer.** It consumes API-shaped responses and demonstrates the workflow; it is not the production frontend claim.
- **Model artifacts are a runtime dependency.** PyTorch weights, scaler, station mapping, and metadata are loaded by the API service.
- **Airflow and MySQL are supporting data infrastructure.** They explain where the historical station data comes from, but they are not the main story for backend / AI application roles.
- **Demo mode and CI make the project interview-safe.** The dashboard can be shown without live infrastructure, while tests cover API behavior, dashboard client behavior, ETL transforms, model-training metadata, and dbt parsing/building.

## Interview Framing

Use this phrasing:

> The center of the application is the FastAPI model-serving boundary. The API validates station inputs, loads model artifacts, returns forecast-horizon metadata, and exposes both single-station prediction and multi-station risk ranking. Streamlit is a demo client for the same workflow, while Airflow and MySQL provide the data foundation behind the model.

Avoid this phrasing:

> Streamlit is the production frontend.

> The LSTM is already a proven production forecaster.
