"""Per-session configuration handling for Streamable HTTP connections.

Captures query parameters on incoming HTTP requests, normalizes known
configuration keys, and stores them keyed by MCP session ID. Makes the
config available on each request via FastMCP Context state under the
key 'session_config'.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    # FastMCP server middleware API
    from fastmcp.server.middleware import Middleware, MiddlewareContext
except Exception:  # pragma: no cover - fallback if fastmcp API changes
    Middleware = object  # type: ignore[assignment]
    MiddlewareContext = object  # type: ignore[assignment]


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _to_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _decode_base64_json(value: str) -> dict:
    """Best-effort decode of a base64/base64url JSON string.

    Accepts values with missing padding and URL-safe characters.
    Returns an empty dict on any failure.
    """
    if not value:
        return {}
    # Normalize padding
    padded = value + "=" * ((4 - len(value) % 4) % 4)
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            raw = decoder(padded)
            return json.loads(raw.decode("utf-8"))
        except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return {}


def _coerce_params_types(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce incoming JSON values into the shapes expected by normalizer.

    - Booleans remain booleans
    - Lists of strings remain lists
    - Other values are cast to str
    """
    coerced: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            coerced[key] = value
        elif isinstance(value, list):
            coerced[key] = [str(v) for v in value]
        else:
            coerced[key] = str(value)
    return coerced


def _normalize_keys(params: Mapping[str, Any]) -> Dict[str, Any]:
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
        raw = params["openbb_mcp_allowed_tool_categories"]
        value = str(raw).strip()
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
            providers[base_key] = str(value)
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
            # Start with direct query params
            params: Dict[str, Any] = {k: v for k, v in request.query_params.items()}
            # Smithery compatibility: allow generic api_key + profile
            if "api_key" in params and "openbb_fmp_api_key" not in params:
                params["openbb_fmp_api_key"] = params["api_key"]
            if "profile" in params:
                # Surface profile for observability/troubleshooting
                params["openbb_mcp_profile"] = params["profile"]
            # Support Smithery-style base64 JSON config blob
            if "config" in params and isinstance(params["config"], str):
                decoded = _decode_base64_json(params["config"]) or {}
                params.update(_coerce_params_types(decoded))
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

    async def on_get_tools(self, context: MiddlewareContext, call_next):  # type: ignore[override]
        """Filter tools per session based on query params and cached config.

        This ensures each session can see only the categories allowed or selected
        as defaults, similar to Smithery's stateful server behavior.
        """
        fastmcp_ctx = context.fastmcp_context

        session_id: Optional[str] = None
        try:
            session_id = fastmcp_ctx.session_id()
        except Exception:
            session_id = None

        request = None
        try:
            request = fastmcp_ctx.get_http_request()
        except Exception:
            request = None

        config: Dict[str, Any] = {}
        if request is not None:
            params: Dict[str, Any] = {k: v for k, v in request.query_params.items()}
            if "api_key" in params and "openbb_fmp_api_key" not in params:
                params["openbb_fmp_api_key"] = params["api_key"]
            if "profile" in params:
                params["openbb_mcp_profile"] = params["profile"]
            if "config" in params and isinstance(params["config"], str):
                decoded = _decode_base64_json(params["config"]) or {}
                params.update(_coerce_params_types(decoded))
            normalized = _normalize_keys(params)
            if normalized:
                config.update(normalized)

        if not config and session_id:
            cached = self.store.get(session_id)
            if cached:
                config.update(cached)

        if session_id and config:
            self.store.set(session_id, config)

        if config:
            fastmcp_ctx.set_state("session_config", config)

        # Get full toolset and then filter for this session
        tools = await call_next()
        try:
            default_cats = set(config.get("default_tool_categories") or [])
            allowed_cats = config.get("allowed_tool_categories")
            allowed_cats_set = set(allowed_cats) if allowed_cats else None

            def tool_in_categories(tool) -> bool:
                tags = set(getattr(tool, "tags", set()) or [])
                # Enforce allowed set if provided
                if allowed_cats_set is not None and not (tags & allowed_cats_set):
                    return False
                # Default set is treated as the initial exposure list
                if default_cats and "all" not in default_cats and not (tags & default_cats):
                    return False
                return True

            if isinstance(tools, dict):
                return {name: tool for name, tool in tools.items() if tool_in_categories(tool)}
        except Exception:
            # Best effort: never fail tool listing due to filtering
            return tools

        return tools


__all__ = [
    "SessionConfigStore",
    "SessionConfigMiddleware",
]


