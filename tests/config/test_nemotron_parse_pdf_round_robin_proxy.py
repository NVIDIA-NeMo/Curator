# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify proxy routing, payload preservation and deadlines over local HTTP."""

import asyncio
import gzip
import importlib
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_proxy_does_not_replay_failed_post(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("round_robin_proxy")
    calls = []

    async def first(request: web.Request) -> web.Response:
        calls.append(await request.read())
        return web.Response(status=503, text="busy")

    async def second(_request: web.Request) -> web.Response:
        calls.append(b"second")
        return web.Response(text="ok")

    app1, app2 = web.Application(), web.Application()
    app1.router.add_post("/v1/chat/completions", first)
    app2.router.add_post("/v1/chat/completions", second)
    async with TestServer(app1) as server1, TestServer(app2) as server2:
        app = module.create_app([str(server1.make_url("")), str(server2.make_url(""))])
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", data=b"page")
            assert response.status == 503
            assert calls == [b"page"]
            response = await client.post("/v1/chat/completions", data=b"next")
            assert response.status == 200
            assert calls == [b"page", b"second"]


@pytest.mark.anyio
async def test_proxy_preserves_encoded_response(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("round_robin_proxy")

    async def encoded(_request: web.Request) -> web.Response:
        return web.Response(body=gzip.compress(b"original response"), headers={"Content-Encoding": "gzip"})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", encoded)
    async with (
        TestServer(upstream) as server,
        TestClient(TestServer(module.create_app([str(server.make_url(""))]))) as client,
    ):
        response = await client.get("/v1/models")
        assert await response.read() == b"original response"


@pytest.mark.anyio
async def test_proxy_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("round_robin_proxy")

    async def slow(_request: web.Request) -> web.Response:
        await asyncio.sleep(0.1)
        return web.Response(text="late")

    upstream = web.Application()
    upstream.router.add_get("/health", slow)
    async with TestServer(upstream) as server:
        app = module.create_app([str(server.make_url(""))], request_timeout_s=0.01)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/health")
            assert response.status == 504


@pytest.mark.anyio
async def test_proxy_does_not_fail_over_after_connection_loss(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("round_robin_proxy")
    calls = []

    async def disconnect(request: web.Request) -> web.Response:
        calls.append("first")
        request.transport.close()
        return web.Response()

    async def other(_request: web.Request) -> web.Response:
        calls.append("second")
        return web.Response(text="must not replay")

    first, second = web.Application(), web.Application()
    first.router.add_post("/v1/chat/completions", disconnect)
    second.router.add_post("/v1/chat/completions", other)
    async with TestServer(first) as server1, TestServer(second) as server2:
        app = module.create_app([str(server1.make_url("")), str(server2.make_url(""))])
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", data=b"one model request")
            assert response.status == 502
            assert calls == ["first"]
