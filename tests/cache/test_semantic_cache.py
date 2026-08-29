from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from adaptive.cache.postgres import PostgresCacheStore
from adaptive.cache.service import CacheMetrics, SemanticCacheService
from adaptive.db.models import Base, Document, DocumentVersion
from adaptive.interfaces import CacheEntry, CacheLookup, RequestContext, RouteDepth, RouteTool

SIMILARITY_THRESHOLD = 0.95


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _lookup(
    *,
    tenant: str = "acme",
    acl: frozenset[str] = frozenset({"support"}),
    embedding: list[float] | None = None,
    model: str = "gpt-test",
    mode: str = "answer",
) -> CacheLookup:
    return CacheLookup(
        query_text="refund policy",
        query_embedding=embedding or [1.0, 0.0],
        context=RequestContext(tenant_id=tenant, subject_id="user-1", acl=acl),
        route_depth=RouteDepth.SINGLE_HOP,
        route_tool=RouteTool.VECTOR,
        model_profile=model,
        response_mode=mode,
    )


def _entry(  # noqa: PLR0913
    *,
    tenant: str = "acme",
    acl: frozenset[str] = frozenset({"support"}),
    embedding: list[float] | None = None,
    expires_at: datetime | None = None,
    model: str = "gpt-test",
    mode: str = "answer",
    references: dict[str, int] | None = None,
) -> CacheEntry:
    return CacheEntry(
        query_embedding=embedding or [1.0, 0.0],
        tenant_id=tenant,
        acl=acl,
        answer_text="approved",
        citations=[{"document_id": "doc-1", "version": 1}],
        model_profile=model,
        response_mode=mode,
        referenced_document_versions=references or {},
        cost_usd=0.01,
        latency_ms=25,
        expires_at=expires_at or datetime.now(UTC) + timedelta(hours=1),
    )


async def _put(store: PostgresCacheStore, entry: CacheEntry) -> None:
    await store.put(entry)


def _active_document(session: Session, document_id: str = "doc-1", version: int = 1) -> None:
    session.add(
        Document(
            id=document_id,
            tenant_id="acme",
            filename="policy.md",
            content_hash=f"hash-{version}",
            version=version,
            canonical_text="refund policy",
            acl=["support"],
            status="active",
        )
    )
    session.add(
        DocumentVersion(
            document_id=document_id,
            tenant_id="acme",
            version=version,
            content_hash=f"hash-{version}",
            canonical_text="refund policy",
            acl=["support"],
            status="active",
        )
    )
    session.flush()


@pytest.mark.asyncio
async def test_cache_hit_requires_same_tenant_and_acl(session: Session) -> None:
    store = PostgresCacheStore(session, similarity_threshold=0.9)
    await _put(store, _entry())

    assert await store.lookup(_lookup(tenant="globex")) is None
    assert await store.lookup(_lookup(acl=frozenset({"finance"}))) is None
    assert (await store.lookup(_lookup())).entry.answer_text == "approved"


@pytest.mark.asyncio
async def test_cache_lookup_requires_cosine_similarity_threshold(session: Session) -> None:
    store = PostgresCacheStore(session, similarity_threshold=SIMILARITY_THRESHOLD)
    await _put(store, _entry())

    assert await store.lookup(_lookup(embedding=[0.7, 0.714])) is None
    hit = await store.lookup(_lookup(embedding=[0.999, 0.01]))
    assert hit is not None
    assert hit.similarity > SIMILARITY_THRESHOLD - 0.01


@pytest.mark.asyncio
async def test_cache_lookup_rejects_expired_entries(session: Session) -> None:
    store = PostgresCacheStore(session)
    await _put(store, _entry(expires_at=datetime.now(UTC) - timedelta(seconds=1)))

    assert await store.lookup(_lookup()) is None
    assert store.last_status == "stale"


@pytest.mark.asyncio
async def test_cache_lookup_requires_model_and_response_mode_compatibility(
    session: Session,
) -> None:
    store = PostgresCacheStore(session)
    await _put(store, _entry())

    assert await store.lookup(_lookup(model="other-model")) is None
    assert await store.lookup(_lookup(mode="json")) is None
    assert await store.lookup(_lookup()) is not None


@pytest.mark.asyncio
async def test_cache_lookup_validates_referenced_document_versions(session: Session) -> None:
    _active_document(session)
    store = PostgresCacheStore(session)
    await _put(store, _entry(references={"doc-1": 1}))
    assert await store.lookup(_lookup()) is not None

    session.query(DocumentVersion).filter_by(document_id="doc-1", version=1).update(
        {"status": "replaced"}
    )
    session.flush()
    assert await store.lookup(_lookup()) is None
    assert store.last_status == "stale"


@pytest.mark.asyncio
async def test_document_version_invalidation_expires_matching_entries(session: Session) -> None:
    store = PostgresCacheStore(session)
    await _put(store, _entry(references={"doc-1": 1}))
    await _put(store, _entry(references={"doc-2": 1}))

    assert await store.invalidate_document("doc-1", 1) == 1
    assert await store.lookup(_lookup()) is None
    assert store.last_status == "stale"


@pytest.mark.asyncio
async def test_cache_service_emits_safe_status_metrics(session: Session) -> None:
    metrics = CacheMetrics()
    events: list[dict[str, object]] = []
    service = SemanticCacheService(
        PostgresCacheStore(session), enabled=True, metrics=metrics, event_sink=events.append
    )
    await service.put(_entry())
    assert await service.lookup(_lookup()) is not None
    assert await service.lookup(_lookup(model="other-model")) is None

    assert metrics.counts["hit"] == 1
    assert metrics.counts["miss"] == 1
    assert {event["status"] for event in events} == {"hit", "miss"}
    assert all("refund policy" not in str(event) for event in events)


@pytest.mark.asyncio
async def test_disabled_cache_reports_bypass_without_query_payload(session: Session) -> None:
    metrics = CacheMetrics()
    events: list[dict[str, object]] = []
    service = SemanticCacheService(
        PostgresCacheStore(session), enabled=False, metrics=metrics, event_sink=events.append
    )

    assert await service.lookup(_lookup()) is None
    assert metrics.counts["bypass"] == 1
    assert events[0]["status"] == "bypass"
    assert "refund policy" not in str(events[0])
