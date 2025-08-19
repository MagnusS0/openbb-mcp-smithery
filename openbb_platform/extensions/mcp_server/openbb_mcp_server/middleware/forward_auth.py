"""Middleware that captures Authorization headers for token forwarding."""

from __future__ import annotations

from typing import Optional

from starlette.types import ASGIApp, Receive, Scope, Send

from openbb_mcp_server.security.context import set_bearer_token, clear_bearer_token


def _extract_bearer_from_headers(headers: list[tuple[bytes, bytes]]) -> Optional[str]:
    auth_value: Optional[str] = None
    x_pat: Optional[str] = None
    pat_alt: Optional[str] = None
    for k, v in headers:
        key = k.decode("latin-1").lower()
        val = v.decode("latin-1").strip()
        if key == "authorization":
            auth_value = val
        elif key == "x-openbb-pat":
            x_pat = val
        elif key == "openbb-pat":
            pat_alt = val
    if auth_value and auth_value.lower().startswith("bearer "):
        token = auth_value[7:].strip()
        if token:
            return token
    # Fallbacks mapped to Bearer semantics
    if x_pat:
        return x_pat
    if pat_alt:
        return pat_alt
    return None


class ForwardAuthMiddleware:
    """ASGI middleware to capture bearer token for downstream forwarding."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])  # type: ignore[assignment]
        token = _extract_bearer_from_headers(headers or [])
        set_bearer_token(token)

        async def _send(message):
            if message.get("type") in {"http.response.body", "http.response.start"}:
                # Clear token at the start of response to avoid leaking across requests
                clear_bearer_token()
            return await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            clear_bearer_token()


