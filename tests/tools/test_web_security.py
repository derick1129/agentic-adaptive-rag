import asyncio

import pytest

from adaptive.interfaces import (
    Budget,
    RequestContext,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
    ToolContext,
    ToolStatus,
)
from adaptive.tools.web_tool import WebTool


def context(query: str = "https://trusted.example/article") -> ToolContext:
    return ToolContext(
        request_context=RequestContext(tenant_id="acme", subject_id="u1"),
        budget=Budget(max_steps=10),
        query=query,
        route_decision=RouteDecision(
            depth=RouteDepth.SINGLE_HOP, tool=RouteTool.WEB, confidence=1, source=RouteSource.POLICY
        ),
        trace_id="trace",
    )


@pytest.mark.asyncio
async def test_rejects_non_allowlisted_domain():
    tool = WebTool(fetcher=lambda *args, **kwargs: "<p>no</p>", allowed_domains={"trusted.example"})
    result = await tool.run(context("https://evil.example/a"))
    assert result.status is ToolStatus.REJECTED
    assert result.error_code == "WEB_DOMAIN_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_sanitizes_scripts_and_frames_content_as_untrusted():
    async def fetch(url, timeout):
        return "<html><script>alert(1)</script><p>Keep this</p></html>"

    result = await WebTool(fetcher=fetch, allowed_domains={"trusted.example"}).run(context())
    assert result.status is ToolStatus.SUCCESS
    assert "alert(1)" not in result.text
    assert "Keep this" in result.text
    assert "UNTRUSTED" in result.text
    assert result.citations[0]["url"] == "https://trusted.example/article"


@pytest.mark.asyncio
async def test_timeout_retries_are_bounded():
    attempts = 0

    async def fetch(url, timeout):
        nonlocal attempts
        attempts += 1
        raise asyncio.TimeoutError

    result = await WebTool(fetcher=fetch, allowed_domains={"trusted.example"}, max_retries=2).run(
        context()
    )
    assert result.status is ToolStatus.TIMEOUT
    assert attempts == 3
