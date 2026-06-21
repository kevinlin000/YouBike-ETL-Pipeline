DBT_PYTHON ?= python3.11
DBT_VENV ?= .venv-dbt
DBT_BIN := $(DBT_VENV)/bin/dbt

.PHONY: help install-dev install-test install-app install-dbt test dbt-parse dbt-build train-lstm api-demo api-contract api-benchmark api-load-test dashboard-demo up down logs ps

help:
	@echo "Available commands:"
	@echo "  make install-dev  Install local development dependencies"
	@echo "  make install-test Install minimal ETL test dependencies"
	@echo "  make install-app  Install FastAPI/Streamlit app dependencies"
	@echo "  make install-dbt  Install optional dbt analytics dependencies"
	@echo "  make test         Run ETL unit tests"
	@echo "  make dbt-parse    Parse the dbt analytics project"
	@echo "  make dbt-build    Run dbt build for the analytics project"
	@echo "  make train-lstm   Train LSTM artifacts into .scratch/model_training"
	@echo "  make api-demo     Run FastAPI with deterministic demo responses"
	@echo "  make api-contract Export OpenAPI schema and local request examples"
	@echo "  make api-benchmark Run the short local API benchmark profile"
	@echo "  make api-load-test Run the longer local API capacity profile"
	@echo "  make dashboard-demo Run Streamlit dashboard in deterministic demo mode"
	@echo "  make up           Build and start Docker Compose services"
	@echo "  make down         Stop Docker Compose services"
	@echo "  make logs         Follow Docker Compose logs"
	@echo "  make ps           List Docker Compose services"

install-dev:
	python -m pip install -r requirements-dev.txt

install-test:
	python -m pip install -r requirements-test.txt

install-app:
	python -m pip install -r requirements_app.txt

install-dbt:
	$(DBT_PYTHON) -m venv $(DBT_VENV)
	$(DBT_VENV)/bin/python -m pip install --upgrade pip
	$(DBT_VENV)/bin/python -m pip install -r requirements-dbt.txt

test:
	python -m pytest tests/ -v

dbt-parse:
	test -f analytics/dbt/profiles.yml || cp analytics/dbt/profiles.example.yml analytics/dbt/profiles.yml
	cd analytics/dbt && ../../$(DBT_BIN) parse --profiles-dir .

dbt-build:
	test -f analytics/dbt/profiles.yml || cp analytics/dbt/profiles.example.yml analytics/dbt/profiles.yml
	cd analytics/dbt && ../../$(DBT_BIN) seed --profiles-dir .
	cd analytics/dbt && ../../$(DBT_BIN) build --profiles-dir .

train-lstm:
	python scripts/train_multistation_lstm.py --data-path data/processed/youbike_weather_merged.csv --output-dir .scratch/model_training

api-demo:
	API_DEMO_MODE=true python -m uvicorn api.app.main:app --reload --port 8000

api-contract:
	python scripts/export_api_contract.py --output-dir docs

api-benchmark:
	python scripts/benchmark_api.py --base-url http://127.0.0.1:8000 --profile demo

api-load-test:
	python scripts/benchmark_api.py --base-url http://127.0.0.1:8000 --profile capacity --json-output .scratch/benchmarks/api-capacity.json

dashboard-demo:
	DASHBOARD_DEMO_MODE=true streamlit run dashboard/app.py

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f

ps:
	docker-compose ps
