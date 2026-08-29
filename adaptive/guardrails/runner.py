"""Orchestrates guardrails and exposes trace-ready machine-readable outcomes."""

from __future__ import annotations

from typing import Any, Iterable

from adaptive.guardrails.input import InputRequest, check_input
from adaptive.guardrails.output import check_output
from adaptive.guardrails.retrieved import check_tool_result
from adaptive.interfaces import Answer, GuardrailResult, RequestContext, ToolResult


class GuardrailRunner:
    def __init__(
        self,
        *,
        max_query_chars: int = 8_000,
        max_payload_bytes: int = 64_000,
        groundedness_threshold: float = 0.55,
        citation_coverage_threshold: float = 1.0,
    ) -> None:
        self.max_query_chars = max_query_chars
        self.max_payload_bytes = max_payload_bytes
        self.groundedness_threshold = groundedness_threshold
        self.citation_coverage_threshold = citation_coverage_threshold

    async def check_input(self, request: InputRequest | dict[str, Any]) -> GuardrailResult:
        return check_input(
            request,
            max_query_chars=self.max_query_chars,
            max_payload_bytes=self.max_payload_bytes,
        )

    async def check_tool_result(
        self, result: ToolResult, context: RequestContext
    ) -> GuardrailResult:
        return check_tool_result(result, context)

    async def check_output(
        self,
        answer: Answer,
        evidence: Iterable[Any],
        context: RequestContext,
    ) -> GuardrailResult:
        return check_output(
            answer,
            evidence,
            context,
            groundedness_threshold=self.groundedness_threshold,
            citation_coverage_threshold=self.citation_coverage_threshold,
        )
