"""Optional reranking adapters."""

# Provider callables are intentionally generic adapters.
# ruff: noqa: TC001, TC003

from __future__ import annotations

from collections.abc import Awaitable, Callable

from adaptive.interfaces import RetrievalQuery
from adaptive.retrieval.contracts import Reranker, ScoredChunk


class FunctionReranker:
    """Adapts a provider function while keeping the retrieval contract stable."""

    def __init__(
        self,
        scorer: Callable[[str, str], float | Awaitable[float]],
    ) -> None:
        self.scorer = scorer

    async def rerank(
        self, query: RetrievalQuery, candidates: list[ScoredChunk], limit: int
    ) -> list[ScoredChunk]:
        scored = []
        for candidate in candidates:
            score = self.scorer(query.text, candidate.chunk.text)
            if hasattr(score, "__await__"):
                score = await score
            scored.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    score_breakdown={**(candidate.score_breakdown or {}), "reranker": float(score)},
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.chunk.id))[:limit]


__all__ = ["FunctionReranker", "Reranker"]
