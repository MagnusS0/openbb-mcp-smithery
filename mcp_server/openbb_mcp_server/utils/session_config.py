"""Per-session configuration handling for Streamable HTTP connections.

Captures query parameters on incoming HTTP requests, normalizes known
configuration keys, and stores them keyed by MCP session ID. Makes the
config available on each request via FastMCP Context state under the
key 'session_config'.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    # FastMCP server middleware API
    from fastmcp.server.middleware import Middleware, MiddlewareContext
except Exception:  # pragma: no cover - fallback if fastmcp API changes
    Middleware = object  # type: ignore[assignment]
    MiddlewareContext = object  # type: ignore[assignment]


def _to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _to_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalize_keys(params: Mapping[str, str]) -> Dict[str, Any]:
    """Map smithery.yaml param names to internal settings keys and coerce types.

    Known keys (from smithery.yaml):
      - openbb_mcp_default_tool_categories -> default_tool_categories (list[str])
      - openbb_mcp_allowed_tool_categories -> allowed_tool_categories (list[str] | None)
      - openbb_mcp_enable_tool_discovery   -> enable_tool_discovery (bool)
      - openbb_mcp_describe_all_responses  -> describe_responses (bool)
      - openbb_mcp_describe_full_response_schema -> describe_responses (bool)

    Also collects any provider keys that start with 'openbb_' so they can be
    used later for credentials mapping if needed.
    """
    out: Dict[str, Any] = {}

    # MCP settings
    if "openbb_mcp_default_tool_categories" in params:
        out["default_tool_categories"] = _to_list(
            params["openbb_mcp_default_tool_categories"]
        )
    if "openbb_mcp_allowed_tool_categories" in params:
        value = params["openbb_mcp_allowed_tool_categories"].strip()
        out["allowed_tool_categories"] = _to_list(value) if value else None
    if "openbb_mcp_enable_tool_discovery" in params:
        out["enable_tool_discovery"] = _to_bool(
            params["openbb_mcp_enable_tool_discovery"]
        )

    # Either of these implies enabling response descriptions
    if "openbb_mcp_describe_all_responses" in params:
        out["describe_responses"] = _to_bool(
            params["openbb_mcp_describe_all_responses"]
        )
    if "openbb_mcp_describe_full_response_schema" in params:
        out["describe_responses"] = _to_bool(
            params["openbb_mcp_describe_full_response_schema"]
        ) or out.get("describe_responses", False)

    # Collect any provider credentials (namespaced under 'providers')
    providers: Dict[str, str] = {}
    for key, value in params.items():
        if key.startswith("openbb_") and not key.startswith("openbb_mcp_"):
            # Map to user_settings.json credential key names by removing the prefix
            # Example: openbb_fmp_api_key -> fmp_api_key
            base_key = key[len("openbb_") :]
            providers[base_key] = value
    if providers:
        out["providers"] = providers

    return out


class SessionConfigStore:
    """In-memory per-session configuration store."""

    def __init__(self) -> None:
        self._by_session: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._by_session.get(session_id)

    def set(self, session_id: str, config: Dict[str, Any]) -> None:
        self._by_session[session_id] = config


class SessionConfigMiddleware(Middleware):  # type: ignore[misc]
    """FastMCP middleware to capture and expose per-session configuration."""

    def __init__(self, store: Optional[SessionConfigStore] = None) -> None:
        super().__init__()
        self.store = store or SessionConfigStore()
        self._settings_lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    def _user_settings_path() -> Path:
        """Locate the user_settings.json path used by OpenBB.

        Prefer OpenBB's configured directory if available, else fallback to
        ~/.openbb_platform/user_settings.json
        """
        try:
            from openbb_core.app.constants import OPENBB_DIRECTORY  # type: ignore

            return Path(OPENBB_DIRECTORY) / "user_settings.json"
        except Exception:
            return Path.home() / ".openbb_platform" / "user_settings.json"

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def on_call_tool(self, context: MiddlewareContext, call_next):  # type: ignore[override]
        # Access FastMCP Context
        fastmcp_ctx = context.fastmcp_context

        # Best-effort: get session id and HTTP request (if using HTTP transport)
        session_id: Optional[str] = None
        try:
            session_id = fastmcp_ctx.session_id()
        except Exception:
            session_id = None

        try:
            request = fastmcp_ctx.get_http_request()
        except Exception:
            request = None

        # Derive/refresh config from query parameters if available
        config: Dict[str, Any] = {}
        if request is not None:
            params = {k: v for k, v in request.query_params.items()}
            normalized = _normalize_keys(params)
            if normalized:
                config.update(normalized)

        # If we could not get from request but have a session id, attempt to load cached
        if not config and session_id:
            cached = self.store.get(session_id)
            if cached:
                config.update(cached)

        # Persist and expose on context state
        if session_id and config:
            self.store.set(session_id, config)

        if config:
            fastmcp_ctx.set_state("session_config", config)

        # If provider credentials are present, temporarily write them into
        # user_settings.json for the duration of this tool call.
        provider_creds: Dict[str, Any] = config.get("providers", {}) if config else {}
        if not provider_creds:
            return await call_next()

        settings_path = self._user_settings_path()
        async with self._settings_lock:
            original = self._read_json(settings_path)
            merged = dict(original) if isinstance(original, dict) else {}
            creds = dict(merged.get("credentials", {}))
            creds.update(provider_creds)
            merged["credentials"] = creds
            try:
                self._write_json(settings_path, merged)
                return await call_next()
            finally:
                # Restore original settings after the tool call
                self._write_json(settings_path, original if isinstance(original, dict) else {})


__all__ = [
    "SessionConfigStore",
    "SessionConfigMiddleware",
]


