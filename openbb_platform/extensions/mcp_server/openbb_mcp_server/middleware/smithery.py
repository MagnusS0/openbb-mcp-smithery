"""Smithery session configuration middleware."""

from __future__ import annotations

import base64
import json
from typing import Callable
from urllib.parse import parse_qs, unquote
from logging import getLogger

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
        # Only process HTTP requests that might have query parameters
        if (context.fastmcp_context and
            hasattr(context.fastmcp_context, "get_http_request") and
            context.fastmcp_context.get_http_request):

            request = context.fastmcp_context.get_http_request()
            if hasattr(request, "url") and hasattr(request.url, "query"):
                query_string = request.url.query
                if query_string:
                    try:
                        parsed = parse_qs(query_string)
                        logger.info(f"🔍 Processing MCP request with query: {query_string}")

                        # Primary: base64-encoded JSON config under `config`
                        if "config" in parsed and parsed["config"]:
                            try:
                                config_b64 = unquote(parsed["config"][0])
                                decoded = base64.b64decode(config_b64)
                                config = json.loads(decoded)
                                if isinstance(config, dict):
                                    self.config_callback(config)
                            except Exception:
                                # Silently ignore config parsing errors
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
                                # Silently ignore config callback errors
                                pass

                    except Exception:
                        # Silently ignore query parsing errors
                        pass

        # Continue with the request
        return await call_next(context)
