.PHONY: help install-dev test up down logs ps

help:
	@echo "Available commands:"
	@echo "  make install-dev  Install local development dependencies"
	@echo "  make install-test Install minimal ETL test dependencies"
	@echo "  make test         Run ETL unit tests"
	@echo "  make up           Build and start Docker Compose services"
	@echo "  make down         Stop Docker Compose services"
	@echo "  make logs         Follow Docker Compose logs"
	@echo "  make ps           List Docker Compose services"

install-dev:
	python -m pip install -r requirements-dev.txt

install-test:
	python -m pip install -r requirements-test.txt

test:
	python -m pytest tests/ -v

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f

ps:
	docker-compose ps
