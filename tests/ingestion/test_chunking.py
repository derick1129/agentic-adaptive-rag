from __future__ import annotations

import pytest

from adaptive.ingestion.chunking import chunk_document
from adaptive.interfaces import CanonicalDocument, ChunkPolicy


def _document(text: str) -> CanonicalDocument:
    return CanonicalDocument(
        content_hash="a" * 64,
        title="Guide",
        canonical_text=text,
        metadata={"document_id": "doc-1", "tenant_id": "tenant-1", "version": 3},
        acl=frozenset({"support"}),
        pages=[{"page_number": 1, "text": text}],
        headings=[
            {"level": 1, "text": "Guide"},
            {"level": 2, "text": "Refunds"},
            {"level": 2, "text": "Security"},
        ],
        tables=[],
    )


def test_chunking_preserves_heading_paths_and_token_page_metadata():
    document = _document(
        "# Guide\n\n## Refunds\n\nRefunds are available.\n\n## Security\n\nUse MFA."
    )

    chunks = chunk_document(document, ChunkPolicy(max_tokens=80, overlap_tokens=10))

    assert [chunk.heading_path for chunk in chunks] == [["Guide", "Refunds"], ["Guide", "Security"]]
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.token_count <= ChunkPolicy().max_tokens for chunk in chunks)
    assert all(chunk.acl == frozenset({"support"}) for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == [0, 1]


def test_chunking_applies_deterministic_token_overlap_within_sections():
    document = _document("## Refunds\n\n" + " ".join(f"word{i}" for i in range(12)))

    chunks = chunk_document(document, ChunkPolicy(max_tokens=5, overlap_tokens=2))

    assert [chunk.text.split() for chunk in chunks] == [
        ["word0", "word1", "word2", "word3", "word4"],
        ["word3", "word4", "word5", "word6", "word7"],
        ["word6", "word7", "word8", "word9", "word10"],
        ["word9", "word10", "word11"],
    ]


def test_chunk_ids_are_stable_for_same_document_version():
    document = _document("# Guide\n\n## Refunds\n\nRefunds are available.")

    first = chunk_document(document, ChunkPolicy(max_tokens=80, overlap_tokens=10))
    second = chunk_document(document, ChunkPolicy(max_tokens=80, overlap_tokens=10))

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_chunking_rejects_empty_documents():
    document = _document("   \n\n")

    with pytest.raises(ValueError, match="empty"):
        chunk_document(document, ChunkPolicy())
