# syntax=docker/dockerfile:1.7-labs
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
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

# Install OpenBB with all extensions
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache openbb-mcp-server openbb

# Remove build-only dependencies to slim the image
RUN apt-get purge -y --auto-remove build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create runtime user
RUN useradd -ms /bin/bash appuser
USER appuser

ENV HOME=/home/appuser

# Create OpenBB config dir
RUN mkdir -p /home/appuser/.openbb_platform

# Copy entrypoint
COPY --chown=appuser:appuser docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE ${PORT}

ENTRYPOINT ["/app/docker-entrypoint.sh"]


