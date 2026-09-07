"""Authentication utilities for the API."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, Request, status
from adaptive.interfaces import RequestContext


async def get_request_context(
    x_tenant_id: Optional[str] = Header(None),
    x_subject_id: Optional[str] = Header(None),
    x_acl: Optional[str] = Header(None),
) -> RequestContext:
    """Extract request context from headers."""
    # In a real implementation, this would validate tokens, etc.
    # For now, we use defaults or headers if provided
    
    tenant_id = x_tenant_id or "default"
    subject_id = x_subject_id or "anonymous"
    acl = frozenset(a.strip() for a in (x_acl or "").split(",") if a.strip())
    
    return RequestContext(
        tenant_id=tenant_id,
        subject_id=subject_id,
        acl=acl,
    )


# Optional: API key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if required."""
    # TODO: Implement actual API key validation
    # For now, we allow all requests in development
    pass
