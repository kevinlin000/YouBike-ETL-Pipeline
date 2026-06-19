# Portfolio Assessment

This repository is best positioned as a maintained data engineering portfolio project, not as a currently operated production service. The strongest story is the breadth across ingestion, orchestration, relational modeling, statistical analysis, model serving, and dashboarding.

## Current Strengths

- Clear end-to-end architecture: Airflow ingestion, MySQL warehouse tables, analysis notebooks, LSTM model serving, FastAPI API, and Streamlit dashboard.
- Portfolio evidence is preserved: GCP, Docker Compose, Airflow, monitoring, and data-volume screenshots.
- Demo mode makes the dashboard interview-ready without requiring Docker Compose, model artifacts, or live API services.
- CI covers ETL behavior, API validation, dashboard client logic, demo mode, and dbt scaffold parsing/build behavior.
- README now separates current runnable demo material from historical deployment evidence.
- `docs/interview_talk_track.md` provides a concise interview script, demo flow, and claim boundaries.
- `docs/adr/0002-dashboard-demo-mode.md` records why demo mode is deterministic and what it should not be used to claim.
- `docs/images/dashboard_demo_walkthrough.gif` provides a compact visual walkthrough for README scanning.
- `dags/youbike_transform.py` centralizes transform behavior used by both the standalone ETL job and the Airflow DAG.
- `validate_transformed_data_for_load()` wires tested data-quality checks into the pre-load ETL path, with strict and warn modes.
- `mart_district_peak_hour_health` adds district-hour analytics with peak/off-peak labeling to the dbt layer.

## Highest-Value Improvements

1. Convert notebook training into a reproducible script.
   - Value: strengthens ML engineering credibility.
   - Scope: scripted training entrypoint, config, fixed output artifacts, documented metrics.
   - Risk: medium to high. Model artifacts and data availability need careful handling.

2. Add a concise architecture diagram image.
   - Value: helps non-technical reviewers understand the system faster than Mermaid alone.
   - Scope: render the existing Airflow/MySQL/FastAPI/Streamlit flow into a static image.
   - Risk: low. Keep it aligned with the README architecture.

3. Add validation observability examples.
   - Value: shows how strict/warn validation affects ETL operations.
   - Scope: document expected log messages and when to use `ETL_VALIDATION_MODE=warn`.
   - Risk: low. Keep it documentation-only unless the project is revived.

## Lower-Priority Improvements

- Rehosting the full stack on cloud: useful only if there is a concrete reason to show live production behavior. Otherwise, it adds cost and operational burden.
- Adding Kubernetes, Kafka, Spark, or a full MLOps stack: not recommended for this repo unless the project is being revived. It would dilute the existing portfolio narrative.
- Replacing the Streamlit dashboard with a custom frontend: visually possible, but not the best return for a data engineering portfolio unless applying for frontend-heavy roles.

## Interview Positioning

Lead with the operational problem: YouBike imbalance is station-level and time-dependent, not just a city-wide bike-count problem. Then show how the project maps that problem into a data system:

1. high-frequency ingestion for station status,
2. normalized warehouse tables,
3. statistical analysis to justify station-level monitoring,
4. LSTM inference served through FastAPI,
5. dashboard demo that converts predictions into ranked operational actions.

Be explicit that dashboard demo mode uses deterministic mock data. The correct claim is that demo mode shows the product and API workflow; it is not evidence of model accuracy.

## Recommendation

For the next engineering iteration, convert notebook training into a reproducible script if the goal is to strengthen the ML engineering story. If the goal is recruiter-facing polish, add a static architecture diagram or short deployment walkthrough.
