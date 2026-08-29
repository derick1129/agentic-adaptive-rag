"""Shared state and dependency injection for Adaptive Agentic RAG V1."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive.interfaces import (
        Budget,
        CacheStore,
        EmbeddingProvider,
        Generator,
        HybridRetriever,
        RequestContext,
        Router,
        Tool,
    )


# Context variables for request-scoped dependencies
request_context_var: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)
budget_var: ContextVar[Budget | None] = ContextVar("budget", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


@dataclass
class AppState:
    """Application-wide shared state (singleton per process)."""

    # Core services (initialized at startup)
    router: Router | None = None
    retriever: HybridRetriever | None = None
    cache: CacheStore | None = None
    embedding_provider: EmbeddingProvider | None = None
    generator: Generator | None = None
    tools: dict[str, Tool] = field(default_factory=dict)

    # Configuration
    settings: Any = None  # Settings instance

    # Runtime state
    is_ready: bool = False
    startup_time: float = field(default_factory=time.time)


# Global app state instance
app_state = AppState()


import time


def get_request_context() -> RequestContext | None:
    """Get the current request context."""
    return request_context_var.get()


def set_request_context(ctx: RequestContext) -> None:
    """Set the current request context."""
    request_context_var.set(ctx)


def get_budget() -> Budget | None:
    """Get the current budget."""
    return budget_var.get()


def set_budget(budget: Budget) -> None:
    """Set the current budget."""
    budget_var.set(budget)


def get_trace_id() -> str | None:
    """Get the current trace ID."""
    return trace_id_var.get()


def set_trace_id(trace_id: str) -> None:
    """Set the current trace ID."""
    trace_id_var.set(trace_id)


def clear_request_state() -> None:
    """Clear all request-scoped context variables."""
    request_context_var.set(None)
    budget_var.set(None)
    trace_id_var.set(None)