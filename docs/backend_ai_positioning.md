# Backend / AI Application Positioning

Use this note when positioning the project for backend or AI application engineering roles.

## Target Role Fit

This project is strongest as a backend / AI application portfolio piece with a data-heavy domain:

- FastAPI service that loads model artifacts and exposes prediction endpoints.
- Pydantic request validation for station predictions and risk-ranking inputs.
- Model-serving boundary between historical PyTorch artifacts and API responses.
- Streamlit dashboard that consumes API-shaped prediction/risk data.
- Deterministic demo mode for showing the application flow without live infrastructure.
- CI coverage for API behavior, dashboard client behavior, ETL transforms, and model-training metadata.

The data engineering stack supports the application story. Airflow, MySQL, Docker Compose, dbt, and GCP evidence show that the application is grounded in a real data pipeline, but the primary job-market positioning should be:

> Backend / AI application engineer who can turn data and model artifacts into tested APIs, decision workflows, and demoable products.

## What To Emphasize

Lead with application behavior:

1. A user or internal operator needs to know which YouBike stations are likely to run out of bikes or docks.
2. The backend exposes prediction and risk-ranking APIs around that workflow.
3. The API validates inputs, handles model readiness, supports lag-window inputs, and returns forecast-horizon metadata.
4. The dashboard converts API responses into operational actions.
5. The ML layer is evaluated conservatively against baselines, so the project does not overclaim model accuracy.

This makes the project more relevant to AI application roles than a notebook-only ML project.

For a visual version of the same positioning, use [`backend_ai_architecture.md`](backend_ai_architecture.md).

## Interview Pitch

Short version:

> I position this as a backend / AI application project. The data pipeline collects station status, but the main engineering story is how I package a PyTorch prototype behind FastAPI, validate inference requests, expose single-station and multi-station risk-ranking endpoints, and build a dashboard demo around those API contracts. I also added baseline evaluation so I can be honest that the LSTM is a model-serving prototype, not a proven production forecaster.

Chinese version:

> 我會把這個專案定位成後端 / AI 應用作品。資料管線負責收集站點狀態，但主要工程重點是把 PyTorch prototype 包成 FastAPI 推論服務，做 request validation，提供單站預測和多站風險排序 endpoint，再用 dashboard 展示 API workflow。我也補了 baseline evaluation，所以可以誠實說 LSTM 是 model-serving prototype，不是已證明準確的 production forecasting model。

## Resume Framing

Use bullets like these:

- Built a FastAPI model-serving layer for YouBike station availability prediction, with validated request schemas, model readiness handling, and forecast-horizon metadata.
- Implemented a decision-support dashboard that turns prediction outputs into stock-out and full-load risk rankings for station operations.
- Added deterministic demo mode so the AI application flow can be demonstrated without live Docker, FastAPI, or model infrastructure.
- Converted notebook LSTM training into a reproducible CLI with model metadata, baseline metrics, Ridge lag-regression comparison, and promotion guidance.
- Maintained CI coverage across API contracts, dashboard client logic, ETL validation, and model-training metadata.

Avoid bullets like:

- Built a production-grade LSTM forecaster.
- Achieved high forecasting accuracy with deep learning.
- Operated a live AI system in production.

## Next Best Improvements

For backend / AI application roles, the highest-value next improvements are:

1. Add a short demo walkthrough video or GIF that starts from the dashboard and maps back to API contracts.
2. Add weather-history alignment only after deciding to deepen the model-serving story.

Do not lead with larger infrastructure rewrites unless the target role is data platform engineering.
