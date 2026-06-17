.PHONY: help install-dev install-test install-dbt test dbt-parse dbt-build up down logs ps

help:
	@echo "Available commands:"
	@echo "  make install-dev  Install local development dependencies"
	@echo "  make install-test Install minimal ETL test dependencies"
	@echo "  make install-dbt  Install optional dbt analytics dependencies"
	@echo "  make test         Run ETL unit tests"
	@echo "  make dbt-parse    Parse the dbt analytics project"
	@echo "  make dbt-build    Run dbt build for the analytics project"
	@echo "  make up           Build and start Docker Compose services"
	@echo "  make down         Stop Docker Compose services"
	@echo "  make logs         Follow Docker Compose logs"
	@echo "  make ps           List Docker Compose services"

install-dev:
	python -m pip install -r requirements-dev.txt

install-test:
	python -m pip install -r requirements-test.txt

install-dbt:
	python -m pip install -r requirements-dbt.txt

test:
	python -m pytest tests/ -v

dbt-parse:
	cd analytics/dbt && dbt parse --profiles-dir .

dbt-build:
	cd analytics/dbt && dbt build --profiles-dir .

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f

ps:
	docker-compose ps
