from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def load_compose() -> dict:
    with (ROOT / "docker-compose.yml").open() as compose_file:
        return yaml.safe_load(compose_file)


def test_compose_declares_required_services_health_checks_and_persistence() -> None:
    compose = load_compose()
    services = compose["services"]

    assert {"api", "postgres", "opensearch", "phoenix"} <= services.keys()
    assert {"postgres_data", "opensearch_data", "phoenix_data"} <= compose["volumes"].keys()

    for service_name in ("postgres", "opensearch", "phoenix"):
        assert "healthcheck" in services[service_name]

    assert services["api"]["ports"] == ["${APP_PORT:-8000}:8000"]
    assert services["api"]["volumes"] == ["uploads:/tmp/adaptive_rag_ingestion"]
    assert services["postgres"]["volumes"] == ["postgres_data:/var/lib/postgresql/data"]
    assert services["opensearch"]["volumes"] == ["opensearch_data:/usr/share/opensearch/data"]
    assert services["phoenix"]["volumes"] == ["phoenix_data:/mnt/data"]


def test_api_depends_on_healthy_infrastructure_and_has_required_environment() -> None:
    compose = load_compose()
    api = compose["services"]["api"]

    assert api["depends_on"] == {
        "postgres": {"condition": "service_healthy"},
        "opensearch": {"condition": "service_healthy"},
        "phoenix": {"condition": "service_healthy"},
    }
    assert {
        "DATABASE_URL",
        "OPENSEARCH_URL",
        "PHOENIX_ENDPOINT",
        "APP_HOST",
        "APP_PORT",
    } <= api["environment"].keys()


def test_dockerfile_and_entrypoint_define_locked_non_root_startup() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text()

    assert "FROM" in dockerfile
    assert "AS builder" in dockerfile
    assert "uv.lock" in dockerfile
    assert "USER app" in dockerfile
    assert "alembic upgrade head" in entrypoint
    assert "exec uvicorn" in entrypoint
