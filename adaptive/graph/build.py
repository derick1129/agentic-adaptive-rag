"""Build and execute the cache → route → direct/agent graph."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from adaptive.graph.agent_policy import AgentPolicy
from adaptive.graph.nodes import default_tool, execute_direct, execute_multi_hop
from adaptive.interfaces import (
    Answer,
    Budget,
    CacheEntry,
    CacheLookup,
    CacheStatus,
    RequestContext,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
)
from adaptive.routers.contracts import QueryContext, RoutingPolicy
from adaptive.routers.escalation import apply_escalation


@dataclass
class GraphDeps:
    router: Any
    retriever: Any
    generator: Any
    tools: dict[RouteTool, Any] = field(default_factory=dict)
    cache: Any | None = None
    embedding_provider: Any | None = None
    agent_policy: AgentPolicy = field(default_factory=AgentPolicy)
    evidence_checker: Any | None = None
    synthesizer: Any | None = None
    guardrail_runner: Any | None = None
    cache_ttl_seconds: int = 3600


async def _maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _cache_answer(hit: Any, context: RequestContext) -> Answer:
    route = RouteDecision(
        depth=RouteDepth.SINGLE_HOP,
        tool=RouteTool.VECTOR,
        confidence=1.0,
        source=RouteSource.POLICY,
        rationale="route recovered from cache",
    )
    return Answer(
        text=hit.entry.answer_text,
        citations=list(hit.entry.citations),
        groundedness_status="grounded",
        route_decision=route,
        actual_tools_used=[],
        cache_status=CacheStatus.HIT,
        usage={"cache_similarity": hit.similarity},
        trace_id=context.trace_id,
    )


async def run_query(
    deps: GraphDeps,
    query: str,
    request_context: RequestContext | None = None,
    budget: Budget | None = None,
    *,
    policy: RoutingPolicy | None = None,
    forced_route: RouteTool | str | None = None,
    embedding: list[float] | None = None,
    model_profile: str | None = None,
    response_mode: str | None = None,
) -> Answer:
    request_context = request_context or RequestContext(tenant_id="default", subject_id="anonymous")
    budget = budget or Budget()
    policy = policy or RoutingPolicy()
    if forced_route is not None:
        policy = policy.model_copy(update={"forced_tool": RouteTool(forced_route)})
    model_profile = model_profile or policy.model_profile
    response_mode = response_mode or policy.response_mode

    if embedding is None and deps.embedding_provider is not None:
        embedding = (await _maybe(deps.embedding_provider.embed([query], input_type="query")))[0]
    if deps.cache is not None and embedding is not None:
        lookup = CacheLookup(
            query_text=query,
            query_embedding=embedding,
            context=request_context,
            route_depth=RouteDepth.SINGLE_HOP,
            route_tool=RouteTool.VECTOR,
            model_profile=model_profile,
            response_mode=response_mode,
        )
        hit = await deps.cache.lookup(lookup)
        if hit is not None:
            return _cache_answer(hit, request_context)

    forced = policy.forced_decision()
    if forced is not None:
        depth, tool = forced
        route = RouteDecision(
            depth=depth,
            tool=tool,
            confidence=1.0,
            source=RouteSource.POLICY,
            rationale="server policy forced route",
        )
    else:
        route = await deps.router.route(
            QueryContext(query=query, request_context=request_context, budget=budget)
        )
        route = apply_escalation(route, policy, budget)

    if route.depth is RouteDepth.MULTI_HOP:
        answer = await execute_multi_hop(query, route, request_context, budget, deps)
    else:
        tool_impl = default_tool(route.tool, deps)
        if tool_impl is None:
            return Answer(
                text="I can't use that tool for this request.",
                citations=[],
                groundedness_status="refused",
                refusal_reason="tool_not_available",
                route_decision=route,
                actual_tools_used=[route.tool],
                cache_status=CacheStatus.MISS,
                trace_id=request_context.trace_id,
            )
        direct_evidence = []
        answer = await execute_direct(
            query, tool_impl, route, request_context, budget, evidence=direct_evidence
        )

    if deps.guardrail_runner is not None and answer.groundedness_status != "refused":
        checked = await deps.guardrail_runner.check_output(answer, direct_evidence, request_context)
        if checked.action.value in {"deny", "refuse"}:
            answer = answer.model_copy(
                update={
                    "text": "I don't have enough evidence to answer reliably.",
                    "citations": [],
                    "groundedness_status": "refused",
                    "refusal_reason": checked.details.get("code", "guardrail_denied"),
                }
            )
    if (
        deps.cache is not None
        and embedding is not None
        and answer.groundedness_status in {"grounded", "partial"}
    ):
        await deps.cache.put(
            CacheEntry(
                query_embedding=embedding,
                tenant_id=request_context.tenant_id,
                acl=request_context.acl,
                answer_text=answer.text,
                citations=answer.citations,
                model_profile=model_profile,
                response_mode=response_mode,
                referenced_document_versions={},
                cost_usd=answer.usage.get("cost_usd", 0.0),
                latency_ms=answer.latency_ms,
                expires_at=datetime.now(UTC) + timedelta(seconds=deps.cache_ttl_seconds),
            )
        )
    return answer
