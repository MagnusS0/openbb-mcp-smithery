"""httpx authentication helper that forwards the current request's bearer token."""

from __future__ import annotations

import httpx

from openbb_mcp_server.security.context import get_bearer_token


class ForwardBearerAuth(httpx.Auth):
    """Attach Authorization header from the per-request security context.

    If a token is present, sets `Authorization: Bearer <token>`.
    If not, leaves the request unchanged.
    """

    def auth_flow(self, request: httpx.Request):  # type: ignore[override]
        token = get_bearer_token()
        if token:
            request.headers.setdefault("Authorization", f"Bearer {token}")
        yield request


