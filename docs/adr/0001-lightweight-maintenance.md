# ADR 0001: Keep YouBike ETL as a Lightweight Portfolio Showcase

## Status

Accepted

## Context

This repo is an older data engineering portfolio project. It already demonstrates a broad stack: GCP, Airflow, Docker, MySQL, FastAPI, Streamlit, LSTM modeling, and statistical analysis.

The project is useful as a resume supplement, but it is not currently the main active engineering project.

## Decision

Use lightweight maintenance defaults:

- Local planning under `.scratch/`
- Small documentation and reproducibility improvements
- No heavy issue workflow, public automation, or major architecture changes by default

## Consequences

- Agent work should focus on preserving and clarifying the existing portfolio value.
- Major upgrades should only happen if the user explicitly decides to revive this repo.
