"""OpenSearch indexing adapter tests."""

# ruff: noqa: ANN001, E501, PLR2004

from dataclasses import dataclass, field

import pytest

from adaptive.interfaces import ChunkDraft, RequestContext, RetrievalQuery
from adaptive.retrieval.contracts import HashEmbeddingProvider
from adaptive.retrieval.opensearch import OpenSearchIndexWriter, OpenSearchStage


def draft(chunk_id: str = "chunk-1", version: int = 1) -> ChunkDraft:
    return ChunkDraft(
        id=chunk_id,
        document_id="doc-1",
        tenant_id="acme",
        version=version,
        ordinal=0,
        text="refund policy details",
        heading_path=["Refunds"],
        page_number=2,
        token_count=3,
        metadata={"source": "guide.md"},
        acl=frozenset({"support"}),
    )


@dataclass
class FakeClient:
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def indices(self):
        return self

    def exists(self, index: str):
        return False

    def create(self, index: str, body: dict):
        self.calls.append(("create", {"index": index, "body": body}))

    def bulk(self, body: list[dict], index: str, refresh: bool):
        self.calls.append(("bulk", {"index": index, "body": body, "refresh": refresh}))
        return {"errors": False, "items": [{"index": {"status": 201}}]}

    def delete(self, index: str, id: str, refresh: bool):
        self.calls.append(("delete", {"index": index, "id": id, "refresh": refresh}))


def test_index_writer_uses_deterministic_ids_and_shared_embedding():
    client = FakeClient()
    writer = OpenSearchIndexWriter(
        client, index_name="chunks-v1", embedding_provider=HashEmbeddingProvider(8)
    )

    first = writer.upsert_chunks([draft()])
    second = writer.upsert_chunks([draft()])

    assert first.indexed_ids == second.indexed_ids
    assert len(first.indexed_ids) == 1
    body = next(payload for kind, payload in client.calls if kind == "bulk")["body"]
    assert body[1]["doc"]["tenant_id"] == "acme"
    assert len(body[1]["doc"]["embedding"]) == 8


def test_index_writer_replaces_versions_and_deletes_by_version_with_scope():
    client = FakeClient()
    writer = OpenSearchIndexWriter(
        client, index_name="chunks-v1", embedding_provider=HashEmbeddingProvider(4)
    )

    writer.upsert_chunks([draft(version=1)])
    writer.upsert_chunks([draft(version=2)])
    writer.delete_document_version("doc-1", 1)

    delete_call = next(payload for kind, payload in client.calls if kind == "delete")
    assert delete_call["id"] == writer.chunk_id(draft(version=1))


def test_hash_embedding_provider_is_deterministic_and_normalized():
    provider = HashEmbeddingProvider(6)
    assert provider.embed(["same text"]) == provider.embed(["same text"])
    assert sum(value * value for value in provider.embed(["same text"])[0]) ** 0.5 == pytest.approx(
        1
    )


@pytest.mark.asyncio
async def test_opensearch_stage_sends_tenant_acl_and_document_filters_server_side():
    class SearchClient:
        def __init__(self):
            self.body = None

        def search(self, index, body):
            self.body = body
            return {"hits": {"hits": []}}

    client = SearchClient()
    stage = OpenSearchStage(client, index_name="chunks-v1", kind="bm25")
    context = RequestContext(tenant_id="acme", subject_id="u1", acl=frozenset({"support"}))

    await stage.search(
        RetrievalQuery(text="refund", context=context, document_ids=["doc-1"]), limit=5
    )

    filters = client.body["query"]["bool"]["filter"]
    assert {"term": {"tenant_id": "acme"}} in filters
    assert {"term": {"acl": "support"}} in filters
    assert {"terms": {"document_id": ["doc-1"]}} in filters
