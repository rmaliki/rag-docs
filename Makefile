PYTHON ?= .venv/bin/python

.PHONY: install test test-integration lint run migrate ingest-demo tokens eval up down monitoring

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

# Needs `make up` (or any PostgreSQL with pgvector and the rag_app role).
test-integration:
	RAG_TEST_ADMIN_DATABASE_URL=postgresql+psycopg://rag_admin:rag_admin@localhost:5432/rag \
	RAG_TEST_APP_DATABASE_URL=postgresql+psycopg://rag_app:rag_app@localhost:5432/rag \
	$(PYTHON) -m pytest -q -m integration

lint:
	$(PYTHON) -m ruff check .

run:
	$(PYTHON) -m uvicorn app.main:create_app --factory --reload --port 8000

migrate:
	$(PYTHON) -m alembic upgrade head

ingest-demo:
	$(PYTHON) -m scripts.ingest_documents data/demo

tokens:
	@$(PYTHON) -m scripts.create_demo_users

eval:
	$(PYTHON) -m scripts.evaluate_retrieval eval/retrieval_eval.jsonl

up:
	docker compose up -d --build db migrate api

down:
	docker compose down

monitoring:
	docker compose --profile monitoring up -d prometheus grafana
