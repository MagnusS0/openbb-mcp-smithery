"""Smithery-compatible HTTP router that spawns per-session workers."""

import asyncio
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Mapping, Optional

import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .auth_overlay import _is_safe_env_value, build_env_from_providers
from .config_schema import parse_and_validate_config
from .session import LRUSessionStore, SessionEntry

# Configuration constants
DEFAULT_HOST = os.environ.get("OPENBB_ROUTER_CHILD_HOST", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("PORT", "8080"))
SESSION_TTL_SECONDS = int(os.environ.get("OPENBB_ROUTER_SESSION_TTL", "1800"))
MAX_SESSIONS = int(os.environ.get("OPENBB_ROUTER_MAX_SESSIONS", "100"))
FORWARD_TIMEOUT_SECONDS = float(os.environ.get("OPENBB_ROUTER_FORWARD_TIMEOUT", "300"))
CHILD_READY_TIMEOUT_SECONDS = float(
    os.environ.get("OPENBB_ROUTER_CHILD_READY_TIMEOUT", "15")
)
CHILD_LOG_MODE = os.environ.get("OPENBB_ROUTER_CHILD_LOG_MODE", "inherit")

# CORS Configuration
EXPOSE_CORS = os.environ.get("OPENBB_ROUTER_ENABLE_CORS", "1") not in {
    "0",
    "false",
    "False",
}
CORS_ORIGINS = os.environ.get("OPENBB_ROUTER_CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = (
    os.environ.get("OPENBB_ROUTER_CORS_CREDENTIALS", "false").lower() == "true"
)

# Circuit breaker constants
MAX_CHILD_FAILURES = int(os.environ.get("OPENBB_ROUTER_MAX_CHILD_FAILURES", "5"))
CHILD_FAILURE_WINDOW_SECONDS = int(
    os.environ.get("OPENBB_ROUTER_CHILD_FAILURE_WINDOW", "300")
)
MAX_REDIRECT_ATTEMPTS = 5
HTTP_CLIENT_POOL_SIZE = int(os.environ.get("OPENBB_ROUTER_HTTP_POOL_SIZE", "100"))
HTTP_CLIENT_POOL_CONNECTIONS = int(
    os.environ.get("OPENBB_ROUTER_HTTP_POOL_CONNECTIONS", "20")
)


def _find_free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def _write_env_file(session_dir: Path, env_map: Dict[str, str]) -> Path:
    """Write a .env file with the provided environment variables."""
    session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    env_file = session_dir / ".env"

    def _quote_env_value(raw: str) -> str:
        # escape newlines first, then escape backslashes and quotes
        v = raw.replace("\n", "\\n")
        v = v.replace("\\", "\\\\").replace('"', '\\"')
        needs_quotes = v.strip() != v or any(
            c in v for c in (" ", "\t", "#", "=", '"', "'")
        )
        return f'"{v}"' if needs_quotes else v

    lines = []
    for key, value in env_map.items():
        safe_value = _quote_env_value(str(value))
        lines.append(f"{key}={safe_value}")

    async with aiofiles.open(env_file, "w", encoding="utf-8") as f:
        await f.write("\n".join(lines) + "\n")
    env_file.chmod(0o600)
    return env_file


def _openbb_env_from_query(query_params: Mapping[str, Any]) -> Dict[str, str]:
    """Pass through ALL_CAPS keys from query params into env, safely.

    Smithery config should provide keys that match OpenBB environment variables
    exactly (e.g., FMP_API_KEY, TIINGO_TOKEN, OPENBB_MCP_ENABLE_TOOL_DISCOVERY).
    """
    env: Dict[str, str] = {}
    denylist: set[str] = {"OPENBB_DIRECTORY", "PATH", "PYTHONPATH"}
    for raw_key, raw_value in query_params.items():
        if not isinstance(raw_key, str):
            continue
        if not isinstance(raw_value, (str, int, float, bool)):
            continue
        # Only accept ALL_CAPS style keys to reduce ambiguity
        if not raw_key.isupper() or any(
            c for c in raw_key if not (c.isalnum() or c == "_")
        ):
            continue
        if raw_key in denylist:
            continue
        value_str = str(raw_value)
        if not _is_safe_env_value(value_str):
            continue
        env[raw_key] = value_str
    return env


async def _prepare_child_environment(
    session_id: str, cfg: Dict[str, Any], query_params: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Create a session directory and an env file for the child process."""
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else None
    provider_env = build_env_from_providers(providers)

    session_dir = Path(tempfile.gettempdir()) / "openbb_mcp_sessions" / session_id

    env_map: Dict[str, str] = {
        "OPENBB_DIRECTORY": str(session_dir),
    }

    if isinstance(cfg.get("default_tool_categories"), list):
        env_map["OPENBB_MCP_DEFAULT_TOOL_CATEGORIES"] = ",".join(
            cfg["default_tool_categories"]
        )
    if (
        isinstance(cfg.get("allowed_tool_categories"), list)
        and cfg["allowed_tool_categories"]
    ):
        env_map["OPENBB_MCP_ALLOWED_TOOL_CATEGORIES"] = ",".join(
            cfg["allowed_tool_categories"]
        )
    if "enable_tool_discovery" in cfg:
        env_map["OPENBB_MCP_ENABLE_TOOL_DISCOVERY"] = str(
            bool(cfg["enable_tool_discovery"])
        )
    if "describe_responses" in cfg:
        env_map["OPENBB_MCP_DESCRIBE_RESPONSES"] = str(bool(cfg["describe_responses"]))

    env_map.update(provider_env)
    env_map.update(_openbb_env_from_query(query_params))

    env_file = await _write_env_file(session_dir, env_map)
    return env_file, session_dir


@dataclass
class ChildProcess:
    pid: int
    port: int
    base_url: str
    proc: subprocess.Popen
    session_dir: Path


@dataclass
class CircuitBreakerState:
    """Track circuit breaker state for child process failures."""

    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False

    def record_failure(self) -> None:
        """Record a failure and check if circuit should open."""
        current_time = time.time()

        if current_time - self.last_failure_time > CHILD_FAILURE_WINDOW_SECONDS:
            self.failure_count = 0

        self.failure_count += 1
        self.last_failure_time = current_time

        if self.failure_count >= MAX_CHILD_FAILURES:
            self.is_open = True

    def can_attempt(self) -> bool:
        """Check if we can attempt to spawn a child process."""
        if not self.is_open:
            return True

        current_time = time.time()
        if current_time - self.last_failure_time > CHILD_FAILURE_WINDOW_SECONDS:
            self.is_open = False
            self.failure_count = 0
            return True

        return False


class ProcessRegistry:
    """Track mapping from mcp-session-id to child process."""

    def __init__(self) -> None:
        self._by_session = LRUSessionStore(
            max_size=MAX_SESSIONS,
            ttl_seconds=SESSION_TTL_SECONDS,
            on_evict=self._on_evict,
        )
        self._keys: set[str] = set()
        self._circuit_breaker = CircuitBreakerState()

    async def _on_evict(self, entry: SessionEntry) -> None:
        child: ChildProcess = entry.value
        # Keep tracking set in sync with LRU store evictions
        self._keys.discard(entry.key)
        await self.cleanup_child_process(child)

    async def cleanup_session_dir(self, session_dir: Path) -> None:
        """Asynchronously clean up session directory."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: shutil.rmtree(session_dir, ignore_errors=True),
            )
        except Exception:
            pass

    async def cleanup_child_process(self, child: ChildProcess) -> None:
        """Safely cleanup a child process and its session directory."""
        try:
            if child.proc.poll() is None:
                child.proc.terminate()
                try:
                    child.proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    child.proc.kill()
        finally:
            await self.cleanup_session_dir(child.session_dir)

    def get(self, session_id: str) -> Optional[ChildProcess]:
        return self._by_session.get(session_id)

    def set(self, session_id: str, child: ChildProcess) -> None:
        self._by_session.set(session_id, child)
        self._keys.add(session_id)

    def delete(self, session_id: str) -> bool:
        deleted = self._by_session.delete(session_id)
        if deleted:
            self._keys.discard(session_id)
        return deleted

    def shutdown(self) -> None:
        for sid in list(self._keys):
            self.delete(sid)

    def can_spawn_child(self) -> bool:
        """Check if circuit breaker allows spawning new child processes."""
        return self._circuit_breaker.can_attempt()

    def record_child_failure(self) -> None:
        """Record a child process failure for circuit breaker."""
        self._circuit_breaker.record_failure()

    def stats(self) -> Dict[str, Any]:
        stats = self._by_session.stats()
        stats["trackedSessionKeys"] = len(self._keys)
        stats["circuitBreakerOpen"] = self._circuit_breaker.is_open
        stats["failureCount"] = self._circuit_breaker.failure_count
        return stats

    def sweep(self) -> None:
        """Run TTL/size sweep on the underlying store."""
        self._by_session.sweep()


registry = ProcessRegistry()

LOGGER = logging.getLogger("openbb_mcp_router")


def _create_http_client() -> httpx.AsyncClient:
    """Create HTTP client with connection pooling and proper limits."""
    limits = httpx.Limits(
        max_keepalive_connections=HTTP_CLIENT_POOL_CONNECTIONS,
        max_connections=HTTP_CLIENT_POOL_SIZE,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(FORWARD_TIMEOUT_SECONDS)

    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        follow_redirects=False,
        http2=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = LOGGER
    Path(tempfile.gettempdir(), "openbb_mcp_sessions").mkdir(
        parents=True, exist_ok=True
    )

    # Initialize and attach the HTTP client to app state for dependency-style access
    app.state.http_client = _create_http_client()

    # Periodic session sweeper to enforce TTL for idle sessions
    stop_event = asyncio.Event()
    sweep_interval = int(os.environ.get("OPENBB_ROUTER_SWEEP_INTERVAL", "60"))

    async def _sweeper() -> None:
        while not stop_event.is_set():
            try:
                registry.sweep()
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sweep_interval)
            except asyncio.TimeoutError:
                # loop and sweep again
                pass

    sweeper_task = asyncio.create_task(_sweeper())

    logger.info("router.start host=%s port=%s", DEFAULT_HOST, ROUTER_PORT)

    yield

    registry.shutdown()
    stop_event.set()
    try:
        await asyncio.wait_for(sweeper_task, timeout=2.0)
    except Exception:
        pass

    client: Optional[httpx.AsyncClient] = getattr(app.state, "http_client", None)
    if client:
        await client.aclose()
        app.state.http_client = None

    logger.info("router.stop")


def _build_app() -> FastAPI:
    app = FastAPI(title="OpenBB MCP Smithery Router", lifespan=lifespan)

    if EXPOSE_CORS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=CORS_ALLOW_CREDENTIALS,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )

    logger = LOGGER
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    @app.get("/healthz")
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok", status_code=200)

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse({"status": "ok", **registry.stats()})

    return app


def _filter_forward_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    hop_by_hop = {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "accept-encoding",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


def _filter_response_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    return _filter_forward_headers(headers)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
)
async def _await_child_ready(
    port: int, proc: subprocess.Popen, deadline_seconds: float
) -> None:
    url = f"http://{DEFAULT_HOST}:{port}/mcp/"
    start = time.time()

    timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while time.time() - start < deadline_seconds:
            if proc.poll() is not None:
                # Child exited before becoming ready
                raise RuntimeError("Child process exited before readiness")
            try:
                resp = await client.get(url)
                # We expect a 406 Not Acceptable meaning the child is up
                if resp.status_code == 406:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                await asyncio.sleep(0.2)
            except Exception as e:
                LOGGER.warning(
                    "Unexpected error during child readiness check: %s", str(e)
                )
                await asyncio.sleep(0.2)
    # Timed out
    raise TimeoutError("Child not ready before deadline")


def _spawn_child(env_file: Path, port: int) -> subprocess.Popen:
    mcp_server_spec = os.environ.get("OPENBB_MCP_SERVER_SPEC", "openbb-mcp-server")
    args = [
        "uvx",
        "--isolated",
        "--env-file",
        str(env_file),
        "--from",
        mcp_server_spec,
        "--with",
        "openbb",
        "openbb-mcp",
        "--transport",
        "streamable-http",
        "--host",
        DEFAULT_HOST,
        "--port",
        str(port),
    ]

    stdout_opt: Optional[int] | None
    stderr_opt: Optional[int] | None
    if CHILD_LOG_MODE == "discard":
        stdout_opt = subprocess.DEVNULL
        stderr_opt = subprocess.DEVNULL
    else:
        stdout_opt = None
        stderr_opt = None
    env = os.environ.copy()

    preexec_fn = None
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            PR_SET_PDEATHSIG = 1

            def _preexec() -> None:
                try:
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    pass

            preexec_fn = _preexec
        except Exception:  # noqa: BLE001
            preexec_fn = None
    proc = subprocess.Popen(
        args,
        stdout=stdout_opt,
        stderr=stderr_opt,
        env=env,
        text=False,
        close_fds=True,
        preexec_fn=preexec_fn,
    )
    return proc


async def _forward_request(child: ChildProcess, req: Request) -> Response:
    """Forward request to child process with connection pooling and retry logic."""
    client: Optional[httpx.AsyncClient] = getattr(req.app.state, "http_client", None)
    if client is None:
        raise RuntimeError("HTTP client not initialized")

    url = f"{child.base_url}/mcp/"
    headers = _filter_forward_headers(req.headers)
    body = await req.body()

    current_url = url
    current_method = "POST"
    current_content = body
    current_params = req.query_params
    resp: httpx.Response

    for _ in range(MAX_REDIRECT_ATTEMPTS + 1):
        try:
            resp = await client.request(
                current_method,
                current_url,
                headers=headers,
                params=current_params,
                content=current_content,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            LOGGER.error("Request to child failed: %s", str(e))
            raise HTTPException(status_code=502, detail="Child process unavailable")
        except Exception as e:
            LOGGER.error("Unexpected error in forward request: %s", str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location") or resp.headers.get("location")
            if not location:
                break
            try:
                new_url = str(httpx.URL(location))
            except httpx.InvalidURL:
                break

            if resp.status_code == 303:
                current_method = "GET"
                current_content = None
            current_url = new_url
            current_params = None
            continue
        break

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_filter_response_headers(resp.headers),
    )


async def _forward_get_sse(child: ChildProcess, req: Request) -> StreamingResponse:
    """Forward SSE request to child process using a dedicated streaming client."""

    url = f"{child.base_url}/mcp/"
    headers = _filter_forward_headers(req.headers)

    stream_client = httpx.AsyncClient(
        timeout=None,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=1),
    )

    try:
        stream_ctx = stream_client.stream(
            "GET", url, headers=headers, params=req.query_params, follow_redirects=True
        )
        resp = await stream_ctx.__aenter__()
    except Exception as e:
        await stream_client.aclose()
        LOGGER.error("Failed to start SSE stream: %s", str(e))
        raise HTTPException(status_code=502, detail="Child process unavailable")

    async def _close_resp_and_client() -> None:
        try:
            try:
                # Close the response stream properly
                await stream_ctx.__aexit__(None, None, None)
            except Exception:
                # Fallback to closing the response directly if needed
                try:
                    await resp.aclose()
                except Exception:
                    pass
        finally:
            try:
                await stream_client.aclose()
            except Exception:
                pass

    def _schedule_close() -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop available to schedule
            return
        try:
            loop.create_task(_close_resp_and_client())
        except Exception:
            pass

    async def stream_body() -> AsyncIterator[bytes]:
        yield b": start\n\n"
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        except Exception as e:
            LOGGER.error("Error during SSE streaming: %s", str(e))
            yield f'data: {{"error": "Stream interrupted: {str(e)}"}}\n\n'.encode()

    background = BackgroundTask(_schedule_close)
    return StreamingResponse(
        stream_body(),
        status_code=resp.status_code,
        headers=_filter_response_headers(resp.headers),
        background=background,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


async def _forward_delete(child: ChildProcess, req: Request) -> Response:
    """Forward DELETE request using global HTTP client."""
    client: Optional[httpx.AsyncClient] = getattr(req.app.state, "http_client", None)
    if client is None:
        raise RuntimeError("HTTP client not initialized")

    url = f"{child.base_url}/mcp/"
    headers = _filter_forward_headers(req.headers)

    try:
        resp = await client.delete(url, headers=headers, params=req.query_params)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_filter_response_headers(resp.headers),
        )
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        LOGGER.error("DELETE request to child failed: %s", str(e))
        raise HTTPException(status_code=502, detail="Child process unavailable")
    except Exception as e:
        LOGGER.error("Unexpected error in DELETE request: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


app = _build_app()


async def _handle_existing_session(session_id: str, req: Request) -> Response:
    """Handle request for existing session."""
    child = registry.get(session_id)
    if not child:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    LOGGER.info("forward.post session_id=%s", session_id)
    return await _forward_request(child, req)


async def _spawn_new_session(cfg: Dict[str, Any], req: Request) -> ChildProcess:
    """Spawn a new child process session."""
    # Check circuit breaker
    if not registry.can_spawn_child():
        LOGGER.warning("Circuit breaker open - too many recent child failures")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable - too many child process failures",
        )

    session_id = uuid.uuid4().hex
    port = _find_free_port()

    proc = None
    session_dir = None
    child = None
    try:
        env_file, session_dir = await _prepare_child_environment(
            session_id, cfg, req.query_params
        )
        proc = _spawn_child(env_file, port)
        child = ChildProcess(
            pid=proc.pid,
            port=port,
            base_url=f"http://{DEFAULT_HOST}:{port}",
            proc=proc,
            session_dir=session_dir,
        )

        try:
            await _await_child_ready(port, proc, CHILD_READY_TIMEOUT_SECONDS)
        except TimeoutError:
            raise HTTPException(
                status_code=502, detail="Child process failed to become ready"
            )
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        if proc.poll() is not None and proc.returncode not in (0, None):
            registry.record_child_failure()
            raise HTTPException(status_code=502, detail="Child process failed to start")

        return child

    except Exception:
        registry.record_child_failure()
        try:
            if child is not None:
                await registry.cleanup_child_process(child)
            elif proc is not None and session_dir is not None:
                # Fallback for when child object wasn't created yet
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                await registry.cleanup_session_dir(session_dir)
        except Exception:
            pass
        raise


async def _finalize_session(
    child: ChildProcess, resp: Response, req: Request
) -> Response:
    """Finalize session registration or cleanup."""
    child_session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get(
        "mcp-session-id"
    )
    if child_session_id:
        registry.set(child_session_id, child)
        LOGGER.info(
            "child.register session_id=%s port=%s pid=%s",
            child_session_id,
            child.port,
            child.proc.pid,
        )
        return resp

    LOGGER.warning("child.no_session_id port=%s pid=%s", child.port, child.proc.pid)
    await registry.cleanup_child_process(child)

    return resp


@app.post("/mcp")
async def mcp_post(req: Request) -> Response:
    """Handle MCP POST requests - forward to existing session or create new one."""
    existing_session_id = req.headers.get("mcp-session-id")

    if existing_session_id:
        return await _handle_existing_session(existing_session_id, req)

    # Parse configuration for new session
    try:
        cfg = parse_and_validate_config(dict(req.query_params)).model_dump(
            exclude_none=True
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Spawn new session
    child = await _spawn_new_session(cfg, req)

    try:
        resp = await _forward_request(child, req)
        return await _finalize_session(child, resp, req)
    except Exception:
        # Emergency cleanup
        try:
            await registry.cleanup_child_process(child)
        except Exception:
            pass
        raise


@app.get("/mcp")
async def mcp_get(req: Request) -> StreamingResponse:
    session_id = req.headers.get("mcp-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing mcp-session-id header")
    child = registry.get(session_id)
    if not child:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    LOGGER.info("forward.sse session_id=%s", session_id)
    return await _forward_get_sse(child, req)


@app.delete("/mcp")
async def mcp_delete(req: Request) -> Response:
    session_id = req.headers.get("mcp-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing mcp-session-id header")
    child = registry.get(session_id)
    if not child:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    try:
        resp = await _forward_delete(child, req)
    finally:
        registry.delete(session_id)
        LOGGER.info("child.delete session_id=%s", session_id)
    return resp


def main() -> None:
    import uvicorn

    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    kwargs: Dict[str, Any] = {}
    use_uvloop = os.environ.get("USE_UVLOOP", "").lower() in {"1", "true", "yes"}
    use_httptools = os.environ.get("USE_HTTPTOOLS", "").lower() in {"1", "true", "yes"}
    if use_uvloop:
        try:
            import uvloop  # noqa: F401

            kwargs["loop"] = "uvloop"
        except Exception:
            pass
    if use_httptools:
        try:
            import httptools  # noqa: F401

            kwargs["http"] = "httptools"
        except Exception:
            pass
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ROUTER_PORT,
        workers=workers,
        **kwargs,
    )


if __name__ == "__main__":
    main()
