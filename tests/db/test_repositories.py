from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptive.db.models import Base
from adaptive.db.repositories import (
    ChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
    EvaluationAnnotationRepository,
    IngestionJobRepository,
    QueryRunRepository,
    SemanticCacheEntryRepository,
    TenantRepository,
)
from adaptive.interfaces import (
    ChunkDraft,
    DocumentCreateParams,
    IngestionJob,
    IngestionStatus,
    RequestContext,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def test_document_repository_cannot_read_another_tenant(session: Session):
    repo = DocumentRepository(session)
    repo.create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )
    repo.create(
        DocumentCreateParams(
            tenant_id="globex",
            filename="b.md",
            content_hash="hash-b",
            title=None,
            canonical_text="B",
            metadata={},
            acl=frozenset(),
        )
    )

    documents = repo.list_active(RequestContext(tenant_id="acme", subject_id="user-1"))

    assert [document.tenant_id for document in documents] == ["acme"]
    assert (
        repo.get(documents[0].id, RequestContext(tenant_id="globex", subject_id="user-2")) is None
    )


def test_active_version_is_unique_per_document(session: Session):
    repo = DocumentRepository(session)
    document = repo.create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )

    repo.activate_version(document.id, version=1)
    with pytest.raises(IntegrityError):
        repo.activate_version(document.id, version=1)


def test_document_version_repository_is_tenant_scoped(session: Session):
    document = DocumentRepository(session).create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )
    versions = DocumentVersionRepository(session)
    context = RequestContext(tenant_id="acme", subject_id="user-1")
    other = RequestContext(tenant_id="globex", subject_id="user-2")

    assert versions.get(document.id, 1, context).document_id == document.id
    assert versions.get(document.id, 1, other) is None


def test_all_repositories_apply_request_context_tenant_filter(session: Session):
    context = RequestContext(tenant_id="acme", subject_id="user-1")
    other = RequestContext(tenant_id="globex", subject_id="user-2")
    document_repo = DocumentRepository(session)
    document = document_repo.create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )
    ChunkRepository(session).bulk_create(
        [
            ChunkDraft(
                id="chunk-a",
                document_id=document.id,
                tenant_id="acme",
                version=1,
                ordinal=0,
                text="A",
                heading_path=[],
                page_number=None,
                token_count=1,
                metadata={},
                acl=frozenset(),
            )
        ]
    )
    IngestionJobRepository(session).create(
        IngestionJob(
            id="job-a",
            tenant_id="acme",
            document_id=document.id,
            status=IngestionStatus.RECEIVED,
            current_stage="received",
        )
    )
    assert ChunkRepository(session).get_by_document(document.id, 1, context) != []
    assert ChunkRepository(session).get_by_document(document.id, 1, other) == []
    assert IngestionJobRepository(session).get("job-a", context) is not None
    assert IngestionJobRepository(session).get("job-a", other) is None


def test_tenant_repository_and_other_control_plane_repositories_round_trip(session: Session):
    tenant = TenantRepository(session).create(slug="acme", name="Acme")
    context = RequestContext(tenant_id=tenant.id, subject_id="user-1")
    query = QueryRunRepository(session).create(
        context, query_text="hello", status="completed", response={"text": "hi"}
    )
    cache = SemanticCacheEntryRepository(session).create(
        context,
        query_embedding=[1.0, 0.0],
        answer_text="hi",
        citations=[],
        model_profile="fake",
        response_mode="answer",
        referenced_document_versions={},
        expires_at=None,
    )
    annotation = EvaluationAnnotationRepository(session).create(
        context,
        query_run_id=query.id,
        evaluator="deterministic",
        score=1.0,
        explanation="exact",
        dataset_item_id="item-1",
        label_source="reference",
    )

    assert TenantRepository(session).get(tenant.id).slug == "acme"
    assert QueryRunRepository(session).get(query.id, context).id == query.id
    assert SemanticCacheEntryRepository(session).get(cache.id, context).answer_text == "hi"
    assert EvaluationAnnotationRepository(session).get(annotation.id, context).score == 1.0
