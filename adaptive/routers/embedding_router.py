"""Embedding nearest-seed route selection."""

from __future__ import annotations

import inspect
import math
from typing import Any

from adaptive.interfaces import RouteDecision, RouteSource
from adaptive.routers.contracts import QueryContext


async def _maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class EmbeddingRouter:
    """Choose the route represented by the closest labeled embedding seed."""

    def __init__(self, embedder: Any, seeds: dict[str, tuple[Any, Any, list[float]]]) -> None:
        self.embedder = embedder
        self.seeds = seeds

    async def route(
        self, query_context: QueryContext | str, context: Any = None, budget: Any = None
    ) -> RouteDecision:
        if not isinstance(query_context, QueryContext):
            query_context = QueryContext(
                query=query_context, request_context=context, budget=budget
            )
        vectors = await _maybe(self.embedder.embed([query_context.query], input_type="query"))
        query_vector = vectors[0]
        label, (depth, tool, seed) = max(
            self.seeds.items(), key=lambda item: _cosine(query_vector, item[1][2])
        )
        confidence = max(0.0, min(1.0, (_cosine(query_vector, seed) + 1.0) / 2.0))
        return RouteDecision(
            depth=depth,
            tool=tool,
            confidence=confidence,
            source=RouteSource.EMBEDDING,
            rationale=f"nearest route seed: {label}",
        )


EmbeddingSeedRouter = EmbeddingRouter
