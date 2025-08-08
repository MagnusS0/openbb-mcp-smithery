#!/usr/bin/env bash
set -euo pipefail

# Ensure OpenBB user settings directory exists
OPENBB_DIR="${HOME}/.openbb_platform"
SETTINGS_FILE="${OPENBB_DIR}/user_settings.json"
MCP_SETTINGS_FILE="${OPENBB_DIR}/mcp_settings.json"

mkdir -p "${OPENBB_DIR}"

# Generate/merge user_settings.json with credentials from env
python - <<'PY'
import json, os, pathlib, sys

openbb_dir = pathlib.Path(os.environ.get('HOME', str(pathlib.Path.home()))) / '.openbb_platform'
settings_file = openbb_dir / 'user_settings.json'
settings = {}
if settings_file.exists():
    try:
        settings = json.loads(settings_file.read_text())
    except Exception:
        settings = {}

credentials = dict(settings.get('credentials', {}))

skip_suffixes = {
    'mcp_name',
    'mcp_description',
    'mcp_default_tool_categories',
    'mcp_allowed_tool_categories',
    'mcp_enable_tool_discovery',
    'mcp_describe_all_responses',
    'mcp_describe_full_response_schema',
}

for k, v in os.environ.items():
    if not k.startswith('OPENBB_'):
        continue
    suffix = k[len('OPENBB_'):].lower()
    if suffix in skip_suffixes:
        continue
    # Map common alias typos
    if suffix == 'tingo_token':
        suffix = 'tiingo_token'
    credentials[suffix] = v

# Optional: merge from OPENBB_CREDENTIALS_JSON
override_json = os.environ.get('OPENBB_CREDENTIALS_JSON')
if override_json:
    try:
        override = json.loads(override_json)
        override_creds = override.get('credentials', override)
        if isinstance(override_creds, dict):
            credentials.update(override_creds)
    except Exception:
        pass

settings['credentials'] = credentials
settings_file.write_text(json.dumps(settings, indent=2))
PY

# Initialize default MCP settings if absent
if [[ ! -f "${MCP_SETTINGS_FILE}" ]]; then
  cat > "${MCP_SETTINGS_FILE}" <<'JSON'
{
  "name": "OpenBB MCP",
  "description": "All OpenBB REST endpoints exposed as MCP tools.",
  "default_tool_categories": ["all"],
  "allowed_tool_categories": null,
  "enable_tool_discovery": true,
  "describe_responses": false
}
JSON
fi

# Apply server config env overrides to upstream if set
ARGS=()
if [[ -n "${OPENBB_MCP_DEFAULT_TOOL_CATEGORIES:-}" ]]; then
  ARGS+=("--default-categories" "${OPENBB_MCP_DEFAULT_TOOL_CATEGORIES}")
fi
if [[ -n "${OPENBB_MCP_ALLOWED_TOOL_CATEGORIES:-}" ]]; then
  ARGS+=("--allowed-categories" "${OPENBB_MCP_ALLOWED_TOOL_CATEGORIES}")
fi
if [[ -n "${OPENBB_MCP_ENABLE_TOOL_DISCOVERY:-}" ]]; then
  if [[ "${OPENBB_MCP_ENABLE_TOOL_DISCOVERY}" == "false" || "${OPENBB_MCP_ENABLE_TOOL_DISCOVERY}" == "0" ]]; then
    ARGS+=("--no-tool-discovery")
  fi
fi

# Start OpenBB MCP server over Streamable HTTP bound to PORT
if command -v openbb-mcp >/dev/null 2>&1; then
  exec openbb-mcp --transport streamable-http --host 0.0.0.0 --port "${PORT}" "${ARGS[@]}"
fi

# Fallback: run via Python module if available
if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('openbb_mcp_server') else 1)
PY
then
  exec python -m openbb_mcp_server.main --transport streamable-http --host 0.0.0.0 --port "${PORT}" "${ARGS[@]}"
fi

echo "openbb-mcp not found and openbb_mcp_server module unavailable. Ensure 'openbb-mcp-server' is installed in the image." >&2
exit 1


