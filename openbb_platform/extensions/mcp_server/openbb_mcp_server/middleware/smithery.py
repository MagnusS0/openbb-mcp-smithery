"""Smithery session configuration middleware."""

from __future__ import annotations

import base64
import json
from typing import Callable
from urllib.parse import parse_qs, unquote

from starlette.types import ASGIApp, Receive, Scope, Send


class SmitheryConfigMiddleware:
    """ASGI middleware to parse Smithery config from query params.

    Parameters
    ----------
    app
        The downstream ASGI application.
    config_callback
        A function that will be invoked with a single argument (dict) containing the
        parsed configuration. It will only be called when config is successfully parsed.
    """

    def __init__(self, app: ASGIApp, config_callback: Callable[[dict], None]):
        self.app = app
        self.config_callback = config_callback

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        try:
            raw_query = scope.get("query_string", b"")
            query = (
                raw_query.decode()
                if isinstance(raw_query, (bytes, bytearray))
                else str(raw_query or "")
            )
            if not query:
                await self.app(scope, receive, send)
                return

            parsed = parse_qs(query)

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

        finally:
            await self.app(scope, receive, send)
