"""Hybrid retrieval and ingestion integration tests."""

# ruff: noqa: ANN001, E501, E702, PLR2004

from dataclasses import dataclass

import pytest

from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.service import IngestionService
from adaptive.interfaces import Chunk, IngestionStatus, RequestContext, RetrievalQuery
from adaptive.retrieval.contracts import HashEmbeddingProvider, ScoredChunk
from adaptive.retrieval.hybrid import HybridRetrieverService


def chunk(chunk_id: str, tenant: str = "acme", acl: frozenset[str] = frozenset()) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        tenant_id=tenant,
        version=1,
        ordinal=0,
        text=f"text for {chunk_id}",
        token_count=3,
        acl=acl,
    )


@dataclass
class FakeStage:
    hits: list[ScoredChunk]

    async def search(self, query: RetrievalQuery, limit: int) -> list[ScoredChunk]:
        return self.hits[:limit]


@pytest.mark.asyncio
async def test_hybrid_retriever_fuses_and_acl_filters_results():
    context = RequestContext(tenant_id="acme", subject_id="u1", acl=frozenset({"support"}))
    visible = chunk("acme-visible", acl=frozenset({"support"}))
    hidden = chunk("acme-hidden", acl=frozenset({"finance"}))
    other_tenant = chunk("globex", tenant="globex")
    retriever = HybridRetrieverService(
        bm25=FakeStage(
            [
                ScoredChunk(chunk=visible, score=10),
                ScoredChunk(chunk=hidden, score=9),
            ]
        ),
        dense=FakeStage(
            [
                ScoredChunk(chunk=other_tenant, score=10),
                ScoredChunk(chunk=visible, score=9),
            ]
        ),
        rrf_k=60,
    )

    result = await retriever.retrieve(RetrievalQuery(text="refund policy", context=context))

    assert [item.id for item in result.chunks] == ["acme-visible"]
    assert result.diagnostics.bm25_count == 2
    assert result.diagnostics.dense_count == 2
    assert result.diagnostics.fusion == "rrf"
    assert result.chunks[0].score_breakdown["rrf"] == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.asyncio
async def test_hybrid_retriever_deduplicates_and_respects_context_token_budget():
    context = RequestContext(tenant_id="acme", subject_id="u1")
    first = chunk("first")
    second = chunk("second")
    retriever = HybridRetrieverService(
        bm25=FakeStage([ScoredChunk(chunk=first, score=2), ScoredChunk(chunk=second, score=1)]),
        dense=FakeStage([ScoredChunk(chunk=first, score=3), ScoredChunk(chunk=second, score=2)]),
        context_token_budget=3,
    )

    result = await retriever.retrieve(RetrievalQuery(text="query", context=context, limit=10))

    assert [item.id for item in result.chunks] == ["first"]
    assert result.diagnostics.final_count == 1


@pytest.mark.asyncio
async def test_hybrid_retriever_applies_optional_reranker():
    context = RequestContext(tenant_id="acme", subject_id="u1")
    first, second = chunk("first"), chunk("second")

    class Reranker:
        async def rerank(self, query: RetrievalQuery, candidates: list[ScoredChunk], limit: int):
            return list(reversed(candidates))[:limit]

    retriever = HybridRetrieverService(
        bm25=FakeStage([ScoredChunk(chunk=first, score=2), ScoredChunk(chunk=second, score=1)]),
        dense=FakeStage([]),
        reranker=Reranker(),
    )

    result = await retriever.retrieve(RetrievalQuery(text="query", context=context, limit=2))

    assert [item.id for item in result.chunks] == ["second", "first"]
    assert result.diagnostics.reranked is True


@pytest.mark.asyncio
async def test_ingestion_indexes_embeddings_and_removes_replaced_version():
    class Source:
        filename = "guide.md"
        mime_type = "text/markdown"

        async def read(self):
            return SourcePayload(data=self.data, filename=self.filename, mime_type=self.mime_type)

    class Indexer:
        def __init__(self):
            self.upserts, self.deleted = [], []

        def upsert_chunks(self, chunks, embeddings=None):
            self.upserts.append((chunks, embeddings))

        def delete_document_version(self, document_id, version):
            self.deleted.append((document_id, version))

    indexer = Indexer()
    service = IngestionService(indexer=indexer, embedding_provider=HashEmbeddingProvider(4))
    context = RequestContext(tenant_id="acme", subject_id="u1")
    first = Source()
    first.data = b"# Guide\n\nfirst"
    second = Source()
    second.data = b"# Guide\n\nsecond"

    first_job = await service.process((await service.submit(first, context)).id)
    second_job = await service.process((await service.submit(second, context)).id)

    assert first_job.status == IngestionStatus.ACTIVE
    assert second_job.status == IngestionStatus.ACTIVE
    assert len(indexer.upserts[0][1]) == 1
    assert indexer.deleted == [(first_job.document_id, 1)]
