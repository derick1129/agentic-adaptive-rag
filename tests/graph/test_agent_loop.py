import pytest

from adaptive.graph.agent_subgraph import AgentDeps, run_agent
from adaptive.interfaces import (
    AgentState,
    Budget,
    RequestContext,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
    TerminationReason,
    ToolContext,
    ToolResult,
    ToolStatus,
)


def state(max_steps=6):
    route = RouteDecision(
        depth=RouteDepth.MULTI_HOP,
        tool=RouteTool.VECTOR,
        confidence=0.5,
        source=RouteSource.ESCALATED,
    )
    tc = ToolContext(
        request_context=RequestContext(tenant_id="acme", subject_id="u1"),
        budget=Budget(max_steps=max_steps, max_tokens=1000, max_cost_usd=10),
        query="find the answer",
        route_decision=route,
        trace_id="trace",
    )
    return AgentState(query=tc.query, context=tc, route_decision=route)


class RepeatingGenerator:
    async def generate_structured(self, prompt, schema, max_tokens):
        if schema.__name__ == "EvidenceCheck":
            return schema(sufficient=False, rationale="No evidence")
        return schema(tool=RouteTool.VECTOR, query="same query", done=False)

    async def generate(self, prompt, max_tokens, temperature=0):
        return "insufficient evidence"


class EmptyTool:
    tool_type = RouteTool.VECTOR

    async def run(self, context):
        return ToolResult(tool=self.tool_type, status=ToolStatus.SUCCESS, text="")


@pytest.mark.asyncio
async def test_agent_terminates_when_same_tool_action_repeats():
    result = await run_agent(
        state(),
        AgentDeps(tools={RouteTool.VECTOR: EmptyTool()}, generator=RepeatingGenerator()),
    )
    assert result.termination_reason is TerminationReason.REPEATED_ACTION
    assert result.answer is not None
    assert result.answer.usage["termination_reason"] == "repeated_action"


class SufficientGenerator:
    async def generate_structured(self, prompt, schema, max_tokens):
        if schema.__name__ == "AgentAction":
            return schema(tool=RouteTool.VECTOR, query="answer", done=False)
        return schema(sufficient=True, rationale="The returned observation supports the answer.")

    async def generate(self, prompt, max_tokens, temperature=0):
        return "The answer is supported by the retrieved evidence."


class EvidenceTool:
    tool_type = RouteTool.VECTOR

    async def run(self, context):
        return ToolResult(
            tool=self.tool_type,
            status=ToolStatus.SUCCESS,
            text="Evidence: answer is 42.",
            citations=[{"url": "doc://one"}],
        )


@pytest.mark.asyncio
async def test_agent_synthesizes_only_after_evidence_is_sufficient():
    result = await run_agent(
        state(),
        AgentDeps(tools={RouteTool.VECTOR: EvidenceTool()}, generator=SufficientGenerator()),
    )
    assert result.termination_reason is TerminationReason.ANSWER_COMPLETE
    assert result.answer.groundedness_status == "grounded"
    assert result.answer.citations == [{"url": "doc://one"}]


@pytest.mark.asyncio
async def test_agent_refuses_when_budget_is_exhausted_without_evidence():
    result = await run_agent(
        state(max_steps=1),
        AgentDeps(tools={RouteTool.VECTOR: EmptyTool()}, generator=RepeatingGenerator()),
    )
    assert result.termination_reason in {
        TerminationReason.REPEATED_ACTION,
        TerminationReason.MAX_STEPS,
    }
    assert result.answer.groundedness_status == "refused"
