# Adaptive Agentic RAG V1

This repository contains a standalone Adaptive Agentic RAG service. The local stack uses Docker Compose for the API, PostgreSQL with pgvector, OpenSearch, and Phoenix. It does not require the original sibling repository or live model credentials for its deterministic tests.

## Start locally

```bash
cp .env.example .env
docker compose up --build
```

The API is available at [http://localhost:8000](http://localhost:8000). PostgreSQL is on port 5432, OpenSearch on 9200, and Phoenix on [http://localhost:6006](http://localhost:6006). The API waits for healthy dependencies, applies Alembic migrations, and then starts Uvicorn. Stop the stack with `docker compose down`; add `-v` only when you intentionally want to remove local data volumes.

## Ingest and query

The API accepts tenant-scoped ingestion and query requests. A basic local flow is:

```bash
curl -X POST http://localhost:8000/v1/ingestion \
  -H 'content-type: application/json' \
  -d '{"filename":"notes.md","content":"Adaptive RAG uses hybrid retrieval.","source_type":"upload"}'

curl -X POST http://localhost:8000/v1/queries \
  -H 'content-type: application/json' \
  -d '{"query":"What does Adaptive RAG use?"}'
```

Provider settings, authentication mode, budgets, cache policy, web allowlist, SQL policy, and telemetry redaction are configured through `.env`. Keep credentials out of committed files; supply them through the environment or a local `.env` file.

## Phoenix

Open Phoenix at [http://localhost:6006](http://localhost:6006) to inspect traces. The Compose API sends OTLP traces to `http://phoenix:6006/v1/traces` inside the network. Prompts, retrieved documents, and credentials are redacted by default.

## Checks

Run the deterministic checks without starting external services:

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy adaptive
uv run pytest tests -q
docker compose config
uv run pytest tests/smoke -q
docker build --target runtime -t adaptive-rag:local .
```

Integration tests use the Compose dependencies when present. The CI workflow runs lock validation, formatting, linting, type checking, unit and integration tests, Compose smoke validation, and the image build.

## Troubleshooting

- If a dependency is still starting, the API retries migrations and exits with a clear error after the configured attempt limit.
- If ports are already in use, change `APP_PORT`, `POSTGRES_PORT`, `OPENSEARCH_PORT`, `PHOENIX_PORT`, or `PHOENIX_GRPC_PORT` in `.env`.
- To inspect service readiness, run `docker compose ps` and `docker compose logs api`.
- To recreate a single service after configuration changes, run `docker compose up --build api`.
