# syntax=docker/dockerfile:1.7-labs

# --- Builder: create a virtual environment with project + deps using uv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install transitive dependencies (without the project) in a separate layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the full project
ADD . /app

# Sync the project into the environment (editable by default)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# --- Final runtime image (no uv in base image) ---
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER=1 \
    PORT=8080

# Minimal runtime system dependencies for networking and OpenBB
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      libxml2 \
      libxslt1.1 \
      libffi8 \
      libssl3 \
      libstdc++6 \
      && rm -rf /var/lib/apt/lists/*

# Provide uv/uvx binaries for runtime (router uses uvx to launch child server)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user and group
RUN useradd -ms /bin/bash app

# Copy the application (including the synced .venv)
COPY --from=builder --chown=app:app /app /app

USER app
WORKDIR /app

# Use the project's virtual environment by default
ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/home/app

# Create working dirs used by the router
RUN mkdir -p /home/app/.openbb_platform /tmp/openbb_mcp_sessions

# Optionally pre-warm uvx cache to speed up first request
RUN uvx --from openbb-mcp-server --with openbb openbb-mcp --help >/dev/null 2>&1 || true

EXPOSE ${PORT}

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

# Ensure the package is importable if editable install is used
ENV PYTHONPATH=/app/src

# Entrypoint
ENTRYPOINT ["python", "-m", "smithery_router.router"]
