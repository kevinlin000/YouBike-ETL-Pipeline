# ADR 0002: Keep Dashboard Demo Mode Deterministic

## Status

Accepted

## Context

The Streamlit dashboard is useful in interviews, but requiring FastAPI, model artifacts, Docker Compose, and local service orchestration makes the demo fragile. This repository is a portfolio showcase, so the dashboard should be easy to run when the goal is to demonstrate the workflow.

## Decision

Keep a deterministic dashboard demo mode enabled by `DASHBOARD_DEMO_MODE=true` or `make dashboard-demo`.

Demo mode uses fixed station fixtures and mock predictions to show:

- single-station prediction flow,
- multi-station risk ranking,
- stock-out and full-load action labels,
- the dashboard's decision-support interaction model.

Demo mode must remain clearly documented as a presentation workflow, not as model-performance evidence.

## Consequences

- The dashboard can be shown without live API services, model files, Docker Compose, or cloud resources.
- README screenshots and interview demos can be reproduced locally with stable data.
- Any future dashboard changes should preserve the distinction between deterministic demo output and real FastAPI/model output.
- Model evaluation claims must come from training/evaluation artifacts, not from demo mode.
