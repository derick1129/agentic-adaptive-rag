"""Backend-neutral hybrid retrieval and OpenSearch adapters."""

from adaptive.retrieval.contracts import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    IndexWriter,
    IndexWriteResult,
    Reranker,
    ScoredChunk,
)
from adaptive.retrieval.hybrid import HybridRetrieverService
from adaptive.retrieval.opensearch import OpenSearchIndexWriter

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "IndexWriteResult",
    "IndexWriter",
    "Reranker",
    "ScoredChunk",
    "HybridRetrieverService",
    "OpenSearchIndexWriter",
]
