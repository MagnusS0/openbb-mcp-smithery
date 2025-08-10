"""Smithery-compatible HTTP router that spawns per-session workers."""

from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Mapping, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from .auth_overlay import build_env_from_providers
from .config_schema import parse_and_validate_config, session_config_json_schema
from .session import LRUSessionStore, SessionEntry

DEFAULT_HOST = os.environ.get("OPENBB_ROUTER_CHILD_HOST", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("PORT", "8080"))
SESSION_TTL_SECONDS = int(os.environ.get("OPENBB_ROUTER_SESSION_TTL", "1800"))
MAX_SESSIONS = int(os.environ.get("OPENBB_ROUTER_MAX_SESSIONS", "100"))
FORWARD_TIMEOUT_SECONDS = float(os.environ.get("OPENBB_ROUTER_FORWARD_TIMEOUT", "300"))
CHILD_READY_TIMEOUT_SECONDS = float(
    os.environ.get("OPENBB_ROUTER_CHILD_READY_TIMEOUT", "15")
)
EXPOSE_CORS = os.environ.get("OPENBB_ROUTER_ENABLE_CORS", "1") not in {
    "0",
    "false",
    "False",
}
CHILD_LOG_MODE = os.environ.get("OPENBB_ROUTER_CHILD_LOG_MODE", "inherit")


def _find_free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _write_env_file(session_dir: Path, env_map: Dict[str, str]) -> Path:
    """Write a .env file with the provided environment variables."""
    session_dir.mkdir(parents=True, exist_ok=True)
    env_file = session_dir / ".env"
    lines = []
    for key, value in env_map.items():
        safe_value = str(value).replace("\n", "\\n")
        lines.append(f"{key}={safe_value}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file


def _openbb_env_from_query(query_params: Mapping[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw_key, raw_value in query_params.items():
        if not isinstance(raw_key, str):
            continue
        if not isinstance(raw_value, (str, int, float, bool)):
            continue
        value_str = str(raw_value)
        if raw_key.startswith("openbb_mcp_"):
            suffix = raw_key[len("openbb_mcp_") :]
            env[f"OPENBB_MCP_{suffix.upper()}"] = value_str
        elif raw_key.startswith("openbb_"):
            suffix = raw_key[len("openbb_") :]
            env[suffix.upper()] = value_str
    return env


def _prepare_child_environment(
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
        )  # type: ignore[arg-type]
    if (
        isinstance(cfg.get("allowed_tool_categories"), list)
        and cfg["allowed_tool_categories"]
    ):
        env_map["OPENBB_MCP_ALLOWED_TOOL_CATEGORIES"] = ",".join(
            cfg["allowed_tool_categories"]
        )  # type: ignore[arg-type]
    if "enable_tool_discovery" in cfg:
        env_map["OPENBB_MCP_ENABLE_TOOL_DISCOVERY"] = str(
            bool(cfg["enable_tool_discovery"])
        )
    if "describe_responses" in cfg:
        env_map["OPENBB_MCP_DESCRIBE_RESPONSES"] = str(bool(cfg["describe_responses"]))

    env_map.update(provider_env)
    env_map.update(_openbb_env_from_query(query_params))

    env_file = _write_env_file(session_dir, env_map)
    return env_file, session_dir


@dataclass
class ChildProcess:
    pid: int
    port: int
    base_url: str
    proc: subprocess.Popen
    session_dir: Path


class ProcessRegistry:
    """Track mapping from mcp-session-id to child process."""

    def __init__(self) -> None:
        self._by_session = LRUSessionStore(
            max_size=MAX_SESSIONS,
            ttl_seconds=SESSION_TTL_SECONDS,
            on_evict=self._on_evict,
        )
        self._keys: set[str] = set()

    def _on_evict(self, entry: SessionEntry) -> None:
        child: ChildProcess = entry.value
        try:
            if child.proc.poll() is None:
                child.proc.terminate()
                try:
                    child.proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    child.proc.kill()
        finally:
            try:
                shutil.rmtree(child.session_dir, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass

    def get(self, session_id: str) -> Optional[ChildProcess]:
        return self._by_session.get(session_id)  # type: ignore[return-value]

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

    def stats(self) -> Dict[str, Any]:
        stats = self._by_session.stats()
        stats["trackedSessionKeys"] = len(self._keys)
        return stats


registry = ProcessRegistry()


LOGGER = logging.getLogger("openbb_mcp_router")


def _build_app() -> FastAPI:
    app = FastAPI(title="OpenBB MCP Smithery Router")

    if EXPOSE_CORS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
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

    @app.get("/.well-known/mcp-config")
    async def well_known_mcp_config() -> JSONResponse:
        schema = session_config_json_schema()
        return JSONResponse(
            content=schema, media_type="application/schema+json; charset=utf-8"
        )

    @app.get("/healthz")
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok", status_code=200)

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse({"status": "ok", **registry.stats()})

    @app.on_event("startup")
    async def on_startup() -> None:
        Path(tempfile.gettempdir(), "openbb_mcp_sessions").mkdir(
            parents=True, exist_ok=True
        )
        logger.info("router.start host=%s port=%s", DEFAULT_HOST, ROUTER_PORT)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        registry.shutdown()
        logger.info("router.stop")

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


async def _await_child_ready(
    port: int, proc: subprocess.Popen, deadline_seconds: float
) -> None:
    url = f"http://{DEFAULT_HOST}:{port}/.well-known/mcp-config"
    start = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(1.0, read=1.0)) as client:
        while time.time() - start < deadline_seconds:
            if proc.poll() is not None:
                await asyncio.sleep(0.1)
                break
            try:
                resp = await client.get(url)
                if resp.status_code in (200, 404):
                    return
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.2)


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
    else:  # inherit
        stdout_opt = None
        stderr_opt = None
    env = os.environ.copy()

    preexec_fn = None
    if sys.platform.startswith("linux"):
        try:
            import ctypes  # type: ignore

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
        preexec_fn=preexec_fn,  # type: ignore[arg-type]
    )
    return proc


async def _forward_request(child: ChildProcess, req: Request) -> Response:
    url = f"{child.base_url}/mcp/"
    headers = _filter_forward_headers(req.headers)
    body = await req.body()
    timeout = httpx.Timeout(FORWARD_TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        max_redirects = 5
        current_url = url
        current_method = "POST"
        current_content = body
        current_params = req.query_params
        resp: httpx.Response
        for _ in range(max_redirects + 1):
            resp = await client.request(
                current_method,
                current_url,
                headers=headers,
                params=current_params,
                content=current_content,
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location") or resp.headers.get("location")
                if not location:
                    break
                try:
                    new_url = str(httpx.URL(location))
                except Exception:  # noqa: BLE001
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
    url = f"{child.base_url}/mcp/"
    headers = _filter_forward_headers(req.headers)

    client = httpx.AsyncClient(timeout=None, follow_redirects=True)
    resp = await client.stream(
        "GET", url, headers=headers, params=req.query_params, follow_redirects=True
    )

    async def _close_resp_and_client() -> None:
        try:
            await resp.aclose()
        finally:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass

    def _schedule_close() -> None:
        try:
            asyncio.get_event_loop().create_task(_close_resp_and_client())
        except Exception:  # noqa: BLE001
            pass

    async def stream_body() -> AsyncIterator[bytes]:
        yield b": start\n\n"
        async for chunk in resp.aiter_raw():
            yield chunk

    background = BackgroundTask(_schedule_close)
    return StreamingResponse(
        stream_body(),
        status_code=resp.status_code,
        headers=_filter_response_headers(resp.headers),
        background=background,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


async def _forward_delete(child: ChildProcess, req: Request) -> Response:
    url = f"{child.base_url}/mcp/"
    headers = _filter_forward_headers(req.headers)
    timeout = httpx.Timeout(FORWARD_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.delete(url, headers=headers, params=req.query_params)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_filter_response_headers(resp.headers),
        )


app = _build_app()


@app.post("/mcp")
async def mcp_post(req: Request) -> Response:
    existing_session_id = req.headers.get("mcp-session-id")
    if existing_session_id:
        child = registry.get(existing_session_id)
        if not child:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        LOGGER.info("forward.post session_id=%s", existing_session_id)
        return await _forward_request(child, req)

    cfg = parse_and_validate_config(dict(req.query_params)).model_dump(
        exclude_none=True
    )
    session_id = uuid.uuid4().hex
    port = _find_free_port()
    env_file, session_dir = _prepare_child_environment(
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

    await _await_child_ready(port, proc, CHILD_READY_TIMEOUT_SECONDS)
    if proc.poll() is not None and proc.returncode not in (0, None):  # type: ignore[attr-defined]
        # Child exited before ready; surface a clear error
        raise HTTPException(status_code=502, detail="Child process failed to start")

    try:
        resp = await _forward_request(child, req)
        child_session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get(
            "mcp-session-id"
        )
        if child_session_id:
            registry.set(child_session_id, child)
            LOGGER.info(
                "child.register session_id=%s port=%s pid=%s",
                child_session_id,
                port,
                proc.pid,
            )
            return resp

        LOGGER.warning("child.no_session_id port=%s pid=%s", port, proc.pid)
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    proc.kill()
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)

        return resp
    except Exception:
        try:
            if proc.poll() is None:
                proc.kill()
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ROUTER_PORT,
        workers=workers,
        loop="uvloop",
        http="httptools",
    )


if __name__ == "__main__":
    main()
