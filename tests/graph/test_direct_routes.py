import pytest

from adaptive.graph.build import GraphDeps, run_query
from adaptive.interfaces import (
    Budget,
    RequestContext,
    RouteDepth,
    RouteDecision,
    RouteSource,
    RouteTool,
    ToolResult,
    ToolStatus,
)
from adaptive.routers.contracts import RoutingPolicy


class NeverRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query):
        self.calls += 1
        raise AssertionError("parametric route must not retrieve")


class ParametricGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt, max_tokens, temperature=0):
        self.calls += 1
        return "4"


class FixedRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls = 0

    async def route(self, query_context):
        self.calls += 1
        return self.decision


class RecordingTool:
    tool_type = RouteTool.VECTOR

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, context):
        self.calls += 1
        return ToolResult(tool=self.tool_type, status=ToolStatus.SUCCESS, text="evidence")


def request_budget() -> Budget:
    return Budget(max_steps=4, max_tokens=1000, max_cost_usd=10)


@pytest.mark.asyncio
async def test_parametric_direct_route_does_not_call_retriever_or_agent():
    retriever = NeverRetriever()
    generator = ParametricGenerator()
    router = FixedRouter(
        RouteDecision(
            depth=RouteDepth.PARAMETRIC,
            tool=RouteTool.PARAMETRIC,
            confidence=1.0,
            source=RouteSource.POLICY,
        )
    )
    result = await run_query(
        GraphDeps(router=router, retriever=retriever, generator=generator),
        "What is two plus two?",
        RequestContext(tenant_id="acme", subject_id="u1"),
        request_budget(),
    )
    assert result.route_decision.depth is RouteDepth.PARAMETRIC
    assert result.actual_tools_used == [RouteTool.PARAMETRIC]
    assert retriever.calls == 0
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_single_hop_executes_exactly_one_selected_tool():
    tool = RecordingTool()
    result = await run_query(
        GraphDeps(
            router=FixedRouter(
                RouteDecision(
                    depth=RouteDepth.SINGLE_HOP,
                    tool=RouteTool.VECTOR,
                    confidence=1.0,
                    source=RouteSource.POLICY,
                )
            ),
            retriever=object(),
            tools={RouteTool.VECTOR: tool},
            generator=ParametricGenerator(),
        ),
        "find it",
        RequestContext(tenant_id="acme", subject_id="u1"),
        request_budget(),
    )
    assert result.actual_tools_used == [RouteTool.VECTOR]
    assert tool.calls == 1


@pytest.mark.asyncio
async def test_cache_hit_returns_before_router():
    router = FixedRouter(
        RouteDecision(
            depth=RouteDepth.PARAMETRIC,
            tool=RouteTool.PARAMETRIC,
            confidence=1.0,
            source=RouteSource.POLICY,
        )
    )
    cache = type(
        "Cache", (), {"lookup": staticmethod(_cache_hit), "put": staticmethod(_cache_put)}
    )()
    result = await run_query(
        GraphDeps(router=router, retriever=object(), generator=ParametricGenerator(), cache=cache),
        "cached",
        RequestContext(tenant_id="acme", subject_id="u1"),
        request_budget(),
        embedding=[1.0, 0.0],
    )
    assert result.cache_status.value == "hit"
    assert result.text == "cached answer"
    assert router.calls == 0


async def _cache_hit(request):
    return type(
        "Hit",
        (),
        {
            "entry": type("Entry", (), {"answer_text": "cached answer", "citations": []})(),
            "similarity": 1.0,
        },
    )()


async def _cache_put(entry):
    raise AssertionError("cache hit must not write")


@pytest.mark.asyncio
async def test_policy_forced_route_bypasses_router():
    generator = ParametricGenerator()
    result = await run_query(
        GraphDeps(router=FixedRouter(_multi_hop()), retriever=NeverRetriever(), generator=generator),
        "forced",
        RequestContext(tenant_id="acme", subject_id="u1"),
        request_budget(),
        policy=RoutingPolicy(forced_tool=RouteTool.PARAMETRIC, forced_depth=RouteDepth.PARAMETRIC),
    )
    assert result.route_decision.tool is RouteTool.PARAMETRIC
    assert generator.calls == 1


def _multi_hop():
    return RouteDecision(
        depth=RouteDepth.MULTI_HOP,
        tool=RouteTool.VECTOR,
        confidence=1.0,
        source=RouteSource.POLICY,
    )
