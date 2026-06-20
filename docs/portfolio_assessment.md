# Portfolio Assessment

This repository is best positioned as a backend / AI application portfolio project with strong data-engineering support, not as a currently operated production service. The strongest story is the path from real station data to tested FastAPI model-serving contracts, risk-ranking workflow, and demoable dashboard behavior.

## Current Strengths

- Clear end-to-end architecture: Airflow ingestion, MySQL warehouse tables, analysis notebooks, LSTM model serving, FastAPI API, and Streamlit dashboard.
- Strong backend / AI application story: validated inference requests, forecast-horizon metadata, risk-ranking responses, demo-mode client behavior, and tests around API/dashboard contracts.
- Portfolio evidence is preserved: GCP, Docker Compose, Airflow, monitoring, and data-volume screenshots.
- Demo mode makes the dashboard interview-ready without requiring Docker Compose, model artifacts, or live API services.
- CI covers ETL behavior, API validation, dashboard client logic, demo mode, and dbt scaffold parsing/build behavior.
- README now separates current runnable demo material from historical deployment evidence.
- `docs/reviewer_quickstart.md` gives interviewers and reviewers a 3-minute and 10-minute reading path through the repo.
- `docs/interview_talk_track.md` provides a concise interview script, demo flow, and claim boundaries.
- `docs/backend_ai_positioning.md` frames the project for backend / AI application engineering roles.
- `docs/backend_ai_architecture.md` provides a backend-centered architecture diagram for reviewers.
- `docs/demo_walkthrough.md` provides a Chinese-first interview demo script that maps dashboard behavior back to API contracts.
- `docs/api_contract_walkthrough.md` documents the model-serving API contract, validation behavior, risk-ranking response shape, and demo-mode contract.
- `docs/documentation_language_strategy.md` documents why Chinese is the primary explanation language and English is supporting material.
- `docs/project_story.md` provides a report-style storyline that connects the operational problem, data pipeline, statistical evidence, LSTM prototype, API, dashboard, limitations, and next steps.
- `docs/ml_modeling_audit.md` clarifies the ML notebook lineage, served LSTM artifacts, evaluation gaps, and interview-safe claims.
- `docs/lstm_evaluation_report.md` records local checkpoint-data training runs and documents that the current LSTM does not beat the strongest baseline on either next-observation or approximate one-hour forecasting.
- `docs/adr/0002-dashboard-demo-mode.md` records why demo mode is deterministic and what it should not be used to claim.
- `docs/images/dashboard_demo_walkthrough.gif` provides a compact visual walkthrough for README scanning.
- `docs/videos/youbike_backend_ai_demo_preview.mp4` provides a silent preview video for quick portfolio review.
- `dags/youbike_transform.py` centralizes transform behavior used by both the standalone ETL job and the Airflow DAG.
- `validate_transformed_data_for_load()` wires tested data-quality checks into the pre-load ETL path, with strict and warn modes.
- `mart_district_peak_hour_health` adds district-hour analytics with peak/off-peak labeling to the dbt layer.
- `scripts/train_multistation_lstm.py` converts the multi-station LSTM notebook flow into a reproducible training CLI with artifact metadata and current-value, rolling-mean, same-time previous-day, and Ridge lag-regression baselines.

## Highest-Value Improvements

1. Record a short demo walkthrough video.
   - Value: shows the backend / AI application flow quickly in a recruiter-friendly format after the reviewer quickstart has established what to look for.
   - Scope: use `docs/demo_walkthrough.md` as the script, start from dashboard demo mode, then map single-station prediction and risk ranking back to API contracts. The existing silent preview video is a draft asset, not a substitute for a voiced walkthrough.
   - Risk: low. Keep demo-mode limitations explicit.

2. Improve model features before more LSTM tuning.
   - Value: strengthens ML engineering credibility after the application story is clear.
   - Scope: keep the current evaluation report as the boundary, then add aligned weather history and richer lag features before replacing served artifacts.
   - Risk: medium. It depends on preserving the processed dataset lineage and matching features to real operations.

3. Add weather-history alignment to inference.
   - Value: closes the gap between notebook training and served inference.
   - Scope: extend the warehouse-backed path beyond recent bike counts so it can join aligned weather history before inference.
   - Risk: medium. It touches API behavior and depends on warehouse availability.

## Lower-Priority Improvements

- Rehosting the full stack on cloud: useful only if there is a concrete reason to show live production behavior. Otherwise, it adds cost and operational burden.
- Adding Kubernetes, Kafka, Spark, or a full MLOps stack: not recommended for this repo unless the project is being revived. It would dilute the existing portfolio narrative.
- Replacing the Streamlit dashboard with a custom frontend: visually possible, but not the best return for a data engineering portfolio unless applying for frontend-heavy roles.

## Interview Positioning

For backend / AI application roles, lead with the user-facing workflow: a rider or operator needs to identify stations at risk of stock-out or full-load. Then show how the project maps that workflow into backend contracts:

1. FastAPI request validation for prediction and risk-ranking payloads,
2. model artifact loading and readiness boundaries,
3. forecast-horizon metadata to avoid misleading API semantics,
4. dashboard demo that converts predictions into ranked operational actions,
5. Airflow/MySQL data pipeline as the foundation behind the model-serving story.

Be explicit that dashboard demo mode uses deterministic mock data. The correct claim is that demo mode shows the product and API workflow; it is not evidence of model accuracy.

Also be explicit that the regression R-squared improvement is not the LSTM evaluation metric. It supports the data-engineering decision to collect high-frequency lag features, while the LSTM currently demonstrates a prototype model-serving path.

## Recommendation

For the next engineering iteration, record a short demo walkthrough video if the target is backend / AI application engineering. Improve model features only after the application story is easy to review. The API now exposes forecast-horizon metadata, and documented local runs show the current LSTM does not beat the strongest baseline on tested horizons, so model-accuracy claims should remain conservative.
