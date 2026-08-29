"""Server policy route selection."""

from adaptive.interfaces import RouteDecision, RouteSource
from adaptive.routers.contracts import QueryContext, RoutingPolicy


class PolicyRouter:
    def __init__(self, policy: RoutingPolicy) -> None:
        self.policy = policy

    async def route(self, query_context: QueryContext) -> RouteDecision | None:
        forced = self.policy.forced_decision()
        if forced is None:
            return None
        depth, tool = forced
        return RouteDecision(
            depth=depth,
            tool=tool,
            confidence=1.0,
            source=RouteSource.POLICY,
            rationale="server policy forced route",
        )
