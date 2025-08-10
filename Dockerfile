# syntax=docker/dockerfile:1.7-labs
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER=1 \
    PORT=8080

# System deps
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ca-certificates \
      git \
      libxml2 \
      libxslt1.1 \
      libffi8 \
      libssl3 \
      libstdc++6 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal Smithery router source
COPY src /app/src

# Install runtime deps: fastapi, uvicorn, httpx, pydantic
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache fastapi uvicorn httpx pydantic uvloop httptools

# Remove build-only dependencies to slim the image
RUN apt-get purge -y --auto-remove build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create runtime user
RUN useradd -ms /bin/bash appuser
USER appuser

ENV HOME=/home/appuser

# Create working dirs
RUN mkdir -p /home/appuser/.openbb_platform /tmp/openbb_mcp_sessions

# Pre-warm uvx cache
RUN uvx --from openbb-mcp-server --with openbb openbb-mcp --help >/dev/null 2>&1 || true

EXPOSE ${PORT}

# Run the Smithery router
ENV PYTHONPATH=/app/src

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

# Entrypoint: respect PORT env inside the app
ENTRYPOINT ["python", "-m", "smithery_router.router"]
