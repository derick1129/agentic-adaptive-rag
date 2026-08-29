"""Structured policy objects used by the bounded agent loop."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive.interfaces import RouteTool


class AgentAction(BaseModel):
    """The only model output accepted as a next-step action."""

    tool: RouteTool
    query: str = Field(min_length=1, max_length=4000)
    done: bool = False


class EvidenceCheck(BaseModel):
    sufficient: bool
    rationale: str = ""


class AgentPolicy(BaseModel):
    allowed_tools: frozenset[RouteTool] = frozenset(
        {RouteTool.VECTOR, RouteTool.SQL, RouteTool.WEB, RouteTool.PARAMETRIC}
    )
    max_steps: int = Field(default=10, ge=1)
    max_action_retries: int = Field(default=0, ge=0)


def action_key(action: AgentAction) -> tuple[str, str]:
    return action.tool.value, " ".join(action.query.split())
