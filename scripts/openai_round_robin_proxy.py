#!/usr/bin/env python3
"""Round-robin OpenAI-compatible proxy for local vLLM/SGLang replicas.

The DeepSWE task containers can reach the host through 172.17.0.1.nip.io.
This proxy lets the containers use one OpenAI-compatible base URL while
spreading requests across per-GPU servers.
"""

from __future__ import annotations

import asyncio
import os
from itertools import count

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


BACKENDS = [
    url.rstrip("/")
    for url in os.environ.get(
        "OPENAI_BACKENDS",
        "http://127.0.0.1:8101,"
        "http://127.0.0.1:8102,"
        "http://127.0.0.1:8103,"
        "http://127.0.0.1:8104,"
        "http://127.0.0.1:8105,"
        "http://127.0.0.1:8106,"
        "http://127.0.0.1:8107",
    ).split(",")
    if url.strip()
]

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

app = FastAPI()
client = httpx.AsyncClient(timeout=None)
counter = count()


def _next_backend() -> str:
    if not BACKENDS:
        raise RuntimeError("OPENAI_BACKENDS is empty")
    return BACKENDS[next(counter) % len(BACKENDS)]


def _target_url(backend: str, request: Request) -> str:
    suffix = request.url.path
    if request.url.query:
        suffix += f"?{request.url.query}"
    return f"{backend}{suffix}"


def _headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


@app.get("/health")
async def health() -> JSONResponse:
    async def check(backend: str) -> dict[str, object]:
        try:
            response = await client.get(f"{backend}/health", timeout=5)
            return {"backend": backend, "ok": response.status_code == 200}
        except Exception as exc:
            return {"backend": backend, "ok": False, "error": str(exc)}

    results = await asyncio.gather(*(check(backend) for backend in BACKENDS))
    return JSONResponse(
        {
            "ok": any(result["ok"] for result in results),
            "backends": results,
        },
        status_code=200 if any(result["ok"] for result in results) else 503,
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request) -> Response:
    body = await request.body()
    last_error: Exception | None = None

    for _ in range(len(BACKENDS)):
        backend = _next_backend()
        try:
            upstream = await client.send(
                client.build_request(
                    request.method,
                    _target_url(backend, request),
                    content=body,
                    headers=_headers(request.headers),
                ),
                stream=True,
            )
            break
        except Exception as exc:
            last_error = exc
    else:
        return JSONResponse(
            {"error": f"no backend available: {last_error}"},
            status_code=503,
        )

    async def stream_body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream.status_code,
        headers=_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
