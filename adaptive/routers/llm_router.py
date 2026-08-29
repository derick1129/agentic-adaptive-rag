"""Structured-output LLM route selection."""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel, Field

from adaptive.interfaces import RouteDecision, RouteDepth, RouteSource, RouteTool
from adaptive.routers.contracts import QueryContext


class RouteModelOutput(BaseModel):
    depth: RouteDepth
    tool: RouteTool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


async def _maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _context(
    query_context: QueryContext | str, context: Any = None, budget: Any = None
) -> QueryContext:
    if isinstance(query_context, QueryContext):
        return query_context
    return QueryContext(query=query_context, request_context=context, budget=budget)


class LLMRouter:
    """Convert only validated structured model output into a RouteDecision."""

    def __init__(self, generator: Any, *, max_tokens: int = 128) -> None:
        self.generator = generator
        self.max_tokens = max_tokens

    async def route(
        self, query_context: QueryContext | str, context: Any = None, budget: Any = None
    ) -> RouteDecision:
        query_context = _context(query_context, context, budget)
        prompt = (
            "Classify the query into exactly one depth and tool. Return structured output only. "
            f"Query: {query_context.query}"
        )
        try:
            output = await _maybe(
                self.generator.generate_structured(prompt, RouteModelOutput, self.max_tokens)
            )
            if not isinstance(output, RouteModelOutput):
                output = RouteModelOutput.model_validate(output)
            return RouteDecision(source=RouteSource.LLM, **output.model_dump())
        except Exception:
            return RouteDecision(
                depth=RouteDepth.MULTI_HOP,
                tool=RouteTool.VECTOR,
                confidence=0.0,
                source=RouteSource.ESCALATED,
                rationale="invalid or unavailable structured route; escalated safely",
            )


StructuredLLMRouter = LLMRouter
