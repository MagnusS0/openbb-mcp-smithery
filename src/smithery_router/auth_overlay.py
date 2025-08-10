"""Per-session OpenBB environment helpers."""

from typing import Any, Dict, Mapping


def _is_safe_env_value(raw: str) -> bool:
    """Basic validation for provider env values.

    - Cap length to avoid abuse
    - Disallow control characters (except newlines)
    """
    if len(raw) > 4096:
        return False
    for ch in raw:
        code = ord(ch)
        if code == 10:  # allow newline
            continue
        if code < 32 or code == 127:
            return False
    return True


def build_env_from_providers(providers: Mapping[str, Any] | None) -> Dict[str, str]:
    """Map provider keys to environment variables."""

    env: Dict[str, str] = {}
    if not providers:
        return env
    for key, value in providers.items():
        if not isinstance(value, str) or not value:
            continue
        if not _is_safe_env_value(value):
            continue
        env[key.upper()] = value
    return env


__all__ = ["build_env_from_providers"]
