"""FastMCP middleware that captures Authorization headers for token forwarding."""

from __future__ import annotations

from typing import Optional

from fastmcp.server.middleware import Middleware, MiddlewareContext

from openbb_mcp_server.security.context import clear_bearer_token, set_bearer_token


def _extract_bearer_from_headers(headers: list[tuple[bytes, bytes]]) -> Optional[str]:
    """Extract bearer token from HTTP headers."""
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


class ForwardAuthMiddleware(Middleware):
    """FastMCP middleware to capture bearer token for downstream forwarding."""

    async def on_message(self, context: MiddlewareContext, call_next):
        """Extract and set bearer token for all MCP messages."""
        token = None

        # Extract token from HTTP headers if available
        request = None

        # Try multiple ways to access the HTTP request
        if (
            hasattr(context, "fastmcp_context")
            and context.fastmcp_context
            and hasattr(context.fastmcp_context, "request")
            and context.fastmcp_context.request
        ):
            request = context.fastmcp_context.request

        if not request:
            try:
                from fastmcp.server.dependencies import get_http_request

                request = get_http_request()
            except Exception:
                pass

        if request and hasattr(request, "headers"):
            headers = []
            for key, value in request.headers.items():
                headers.append((key.encode("latin-1"), value.encode("latin-1")))
            token = _extract_bearer_from_headers(headers)

        # If no token from headers, check runtime session config (for Smithery)
        if not token:
            try:
                from openbb_mcp_server.service.mcp_service import MCPService

                mcp_service = MCPService()
                session_config = mcp_service.runtime_session_config
                token = session_config.get("apiKey")
            except Exception:
                pass

        if token:
            set_bearer_token(token)

        try:
            # Continue with the request
            result = await call_next(context)
            return result
        finally:
            # Clear token after request completes to avoid leaking across requests
            clear_bearer_token()
