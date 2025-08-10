"""Session configuration schema and parsing utilities."""

import json
from base64 import b64decode
from typing import Any, Dict, Mapping, MutableMapping

from pydantic import BaseModel, Field, ValidationError, field_validator


class ProviderCredentials(BaseModel):
    """Container for provider credentials."""

    model_config = {
        "extra": "allow",
    }


class SessionConfig(BaseModel):
    """Session-scoped configuration for a stateful MCP session."""

    default_tool_categories: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Default active tool categories on session start.",
    )
    allowed_tool_categories: list[str] | None = Field(
        default=None,
        description="If set, restrict available tool categories to this list.",
    )

    enable_tool_discovery: bool = Field(
        default=False,
        description="Enable tool discovery; disable for multi-client deployments.",
    )
    describe_responses: bool = Field(
        default=False,
        description="Include response types in tool descriptions.",
    )

    providers: ProviderCredentials | Mapping[str, Any] | None = Field(
        default=None,
        description="Per-session provider credentials mapping by API Key Name.",
    )

    @field_validator(
        "default_tool_categories", "allowed_tool_categories", mode="before"
    )
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v


def _set_deep(mapping: MutableMapping[str, Any], path: list[str], value: Any) -> None:
    """Set a value into a nested mapping using a list path."""

    current: MutableMapping[str, Any] = mapping
    for key in path[:-1]:
        nxt = current.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            current[key] = nxt
        current = nxt
    current[path[-1]] = value


def parse_and_validate_config(query_params: Mapping[str, Any]) -> SessionConfig:
    """Parse base64 `config` and dotted params into a SessionConfig."""

    payload: Dict[str, Any] = {}
    cfg_param = query_params.get("config")
    if isinstance(cfg_param, str) and cfg_param:
        try:
            decoded = b64decode(cfg_param)
            loaded = json.loads(decoded.decode("utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid base64 config parameter") from exc

    for raw_key, raw_value in query_params.items():
        if raw_key in {"config", "api_key", "profile"}:
            continue
        if not isinstance(raw_key, str):
            continue
        key_path = [p for p in raw_key.split(".") if p]
        if not key_path:
            continue

        value: Any = raw_value
        if isinstance(raw_value, str):
            try:
                value = json.loads(raw_value)
            except Exception:  # noqa: BLE001
                value = raw_value
        _set_deep(payload, key_path, value)

    try:
        return SessionConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
