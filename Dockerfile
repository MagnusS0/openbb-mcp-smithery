# syntax=docker/dockerfile:1.7-labs

# Use Python 3.12 slim for smaller image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Poetry to not create virtual environments (use system Python)
ENV POETRY_VENV_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy the entire source code
COPY . /app

# Install Poetry and then use dev_install script for proper dependency management
RUN python -m pip install poetry \
    && cd openbb_platform \
    && python smithery_install.py

# Expose HTTP port used by Smithery
EXPOSE 8080

# Runtime environment for multi-session auth
# - Enforce API auth and use PAT extension
# - Disable tool discovery for multi-session deployments
ENV OPENBB_API_AUTH=true \
    OPENBB_API_AUTH_EXTENSION=auth_pat \
    OPENBB_MCP_FORWARD_BEARER_ENABLED=true \
    OPENBB_MCP_SMITHERY_ENABLED=true \
    OPENBB_MCP_SMITHERY_ALLOW_ORIGINS=* \
    OPENBB_MCP_ENABLE_TOOL_DISCOVERY=false \
    OPENBB_DEV_MODE=false

# Entrypoint: run MCP in HTTP mode on port 8080 with discovery disabled by default
# Users provide PAT via Authorization header; Smithery can inject via config param
CMD ["openbb-mcp", "--host", "0.0.0.0", "--port", "8080", "--no-tool-discovery"]


