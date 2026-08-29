from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from adaptive.interfaces import RequestContext
from adaptive.observability.attributes import TelemetryConfig
from adaptive.observability.tracing import TraceManager


def _manager(exporter: InMemorySpanExporter) -> TraceManager:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TraceManager(provider=provider, config=TelemetryConfig())


def test_query_trace_contains_required_tree_and_attributes() -> None:
    exporter = InMemorySpanExporter()
    manager = _manager(exporter)
    context = RequestContext(tenant_id="acme", subject_id="user-1", acl=frozenset({"support"}))

    with manager.start_query(context, query="secret customer query", model_profile="test") as trace:
        with trace.span("cache.lookup", {"cache.status": "miss"}):
            pass
        with trace.span("router.decide", {"route.depth": "single_hop", "route.tool": "vector"}):
            pass
        with trace.span("retrieval.hybrid"):
            for name in (
                "retrieval.bm25",
                "retrieval.dense",
                "retrieval.fusion",
                "retrieval.rerank",
            ):
                with trace.span(name):
                    pass
        with (
            trace.span("agent.run"),
            trace.span(  # noqa: SIM117
                "agent.step", {"agent.step_count": 1}
            ),
        ):
            with trace.span("tool.vector"):
                pass
            with trace.span("agent.evidence_check"):
                pass
        with trace.span("answer.synthesize"):
            pass
        with trace.span("guardrail.output", {"guardrail.status": "allow"}):
            pass
        with trace.span("evaluation.record"):
            pass

    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} >= {"rag.request", "cache.lookup"}
    assert {span.name for span in spans} >= {
        "rag.request",
        "cache.lookup",
        "router.decide",
        "retrieval.hybrid",
        "retrieval.bm25",
        "retrieval.dense",
        "retrieval.fusion",
        "retrieval.rerank",
        "agent.run",
        "agent.step",
        "tool.vector",
        "agent.evidence_check",
        "answer.synthesize",
        "guardrail.output",
        "evaluation.record",
    }
    root = next(span for span in spans if span.name == "rag.request")
    retrieval = next(span for span in spans if span.name == "retrieval.hybrid")
    tool = next(span for span in spans if span.name == "tool.vector")
    assert retrieval.parent.span_id == root.context.span_id
    assert tool.parent.span_id != root.context.span_id
    assert root.attributes["request.id"] == context.request_id
    assert root.attributes["trace.id"] == context.trace_id
    assert root.attributes["tenant.scope_id"] != context.tenant_id
    assert root.attributes["query.hash"]
    assert all("secret customer query" not in str(span.attributes) for span in spans)
    assert all("api_key" not in str(span.attributes).lower() for span in spans)


def test_sensitive_attributes_are_redacted_or_omitted() -> None:
    exporter = InMemorySpanExporter()
    manager = _manager(exporter)
    context = RequestContext(tenant_id="acme", subject_id="user-1")

    with (
        manager.start_query(context, query="do not record this") as trace,
        trace.span(  # noqa: SIM117
            "tool.sql",
            {
                "prompt": "do not record this",
                "retrieved.document": "private text",
                "api_key": "super-secret",
                "safe.value": "ok",
            },
        ),
    ):
        pass

    attrs = next(
        span.attributes for span in exporter.get_finished_spans() if span.name == "tool.sql"
    )
    assert "prompt" not in attrs
    assert "retrieved.document" not in attrs
    assert "api_key" not in attrs
    assert attrs["safe.value"] == "ok"


def test_metrics_expose_required_instruments() -> None:
    manager = TraceManager(config=TelemetryConfig())
    metrics = manager.metrics
    for method in (
        "request",
        "cache_hit",
        "route",
        "retrieval_latency",
        "agent_step",
        "tokens",
        "cost",
        "guardrail",
        "failure",
    ):
        getattr(metrics, method)()
    assert set(metrics.instrument_names) >= {
        "rag.requests",
        "rag.cache_hits",
        "rag.routes",
        "rag.retrieval.latency",
        "rag.agent.steps",
        "rag.tokens",
        "rag.cost",
        "rag.guardrail.outcomes",
        "rag.failures",
    }
