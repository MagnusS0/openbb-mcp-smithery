"""Security context for per-request bearer token forwarding."""

from __future__ import annotations

import contextvars
from typing import Optional

_bearer_token_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "openbb_mcp_bearer_token", default=None
)


def set_bearer_token(token: Optional[str]) -> None:
    """Set bearer token for the current request context."""
    _bearer_token_ctx.set(token)


def get_bearer_token() -> Optional[str]:
    """Get bearer token from the current request context."""
    return _bearer_token_ctx.get()


def clear_bearer_token() -> None:
    """Clear bearer token from the current request context."""
    _bearer_token_ctx.set(None)


