PYTHON ?= python

.PHONY: install test lint train api loadtest up

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests locust scripts
	ruff format --check src tests locust scripts

train:
	$(PYTHON) -m churn.train --data data/Telco-Customer-Churn.csv --n-trials 25

api:
	uvicorn churn.api:app --host 0.0.0.0 --port 8000

loadtest:
	locust -f locust/locustfile.py --host http://127.0.0.1:8000

up:
	docker compose up --build
