"""Smithery session configuration middleware."""

from __future__ import annotations

import base64
import contextlib
import json
from logging import getLogger
from typing import Callable
from urllib.parse import parse_qs, unquote

from fastmcp.server.middleware import Middleware, MiddlewareContext

logger = getLogger(__name__)


class SmitheryConfigMiddleware(Middleware):
    """FastMCP middleware to parse Smithery config from HTTP query params.

    Parameters
    ----------
    config_callback
        A function that will be invoked with a single argument (dict) containing the
        parsed configuration. It will only be called when config is successfully parsed.
    """

    def __init__(self, config_callback: Callable[[dict], None]):
        self.config_callback = config_callback

    async def on_message(self, context: MiddlewareContext, call_next):
        """Parse Smithery config from HTTP query parameters for all MCP messages."""
        request = None

        # Will become deprecated
        if (
            hasattr(context, "fastmcp_context")
            and context.fastmcp_context
            and hasattr(context.fastmcp_context, "get_http_request")
        ):
            with contextlib.suppress(Exception):
                request = context.fastmcp_context.get_http_request()

        if not request:
            try:
                from fastmcp.server.dependencies import get_http_request

                request = get_http_request()
            except Exception:
                pass

        if not request:
            return await call_next(context)

        # Now process the request
        if hasattr(request, "url") and hasattr(request.url, "query"):
            query_string = request.url.query
            if query_string:
                try:
                    parsed = parse_qs(query_string)

                    # Primary: base64-encoded JSON config under `config`
                    if "config" in parsed and parsed["config"]:
                        try:
                            config_b64 = unquote(parsed["config"][0])
                            decoded = base64.b64decode(config_b64)
                            config = json.loads(decoded)
                            if isinstance(config, dict):
                                self.config_callback(config)
                        except Exception:
                            pass

                    # Secondary: smithery.* query parameters
                    smithery_config: dict[str, object] = {}
                    for key, values in parsed.items():
                        if not key.startswith("smithery."):
                            continue
                        value = values[0] if values else None
                        # Build nested structure for smithery.* keys
                        path = key.split(".")[1:]  # drop leading 'smithery'
                        cursor: dict[str, object] = smithery_config
                        for segment in path[:-1]:
                            next_obj = cursor.get(segment)
                            if not isinstance(next_obj, dict):
                                next_obj = {}
                                cursor[segment] = next_obj
                            cursor = next_obj  # type: ignore[assignment]
                        if path:
                            cursor[path[-1]] = value

                    if smithery_config:
                        try:
                            self.config_callback(smithery_config)
                        except Exception:
                            pass

                except Exception:
                    pass

        # Continue with the request
        return await call_next(context)