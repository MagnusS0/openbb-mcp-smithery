"""Per-session OpenBB environment helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


def build_env_from_providers(providers: Mapping[str, Any] | None) -> Dict[str, str]:
    """Map provider keys to environment variables."""

    env: Dict[str, str] = {}
    if not providers:
        return env
    for key, value in providers.items():
        if not isinstance(value, str) or not value:
            continue
        env[key.upper()] = value
    return env


def make_session_env(
    session_dir: Path,
    base_env: Mapping[str, str] | None,
    provider_env: Mapping[str, str],
) -> Dict[str, str]:
    """Construct environment variables for a per-session worker."""

    session_dir.mkdir(parents=True, exist_ok=True)
    env = dict(base_env or os.environ)
    env["OPENBB_DIRECTORY"] = str(session_dir)
    for k, v in provider_env.items():
        env.setdefault(k, v)
    return env


def write_user_settings_overlay(
    session_dir: Path,
    base_settings: Mapping[str, Any] | None,
    session_overrides: Mapping[str, Any] | None,
) -> Path:
    """Write a user_settings.json overlay inside the session directory."""

    settings: Dict[str, Any] = {}
    if base_settings:
        settings.update(json.loads(json.dumps(base_settings)))
    providers = settings.setdefault("providers", {})
    session_providers = (session_overrides or {}).get("providers") or {}
    if isinstance(session_providers, dict):
        for provider, cfg in session_providers.items():
            if isinstance(cfg, dict):
                existing = providers.setdefault(provider, {})
                existing.update(cfg)
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "user_settings.json"
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return path
