# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nemo_curator.core.serve.dynamo import admission_proxy as admission


@pytest.mark.parametrize("value", ["NaN", "Inf", "-1", "bad"])
def test_invalid_metrics_fail_closed(value: str) -> None:
    with pytest.raises(ValueError, match=r"finite|convert"):
        admission._prometheus_metric_total(f'vllm:num_requests_waiting{{model="m"}} {value}')
    with pytest.raises(ValueError, match="not found"):
        admission._prometheus_metric_total("unrelated_metric 0")


def test_metric_series_are_summed() -> None:
    assert (
        admission._prometheus_metric_total(
            '# HELP ignored\nvllm:num_requests_waiting{rank="0"} 3\nvllm:num_requests_waiting{rank="1"} 4'
        )
        == 7
    )


def test_metric_labels_can_contain_spaces_escaped_quotes_and_braces() -> None:
    payload = (
        'vllm:num_requests_waiting{model="a model",instance="0"} 3 123456\n'
        r'vllm:num_requests_waiting{model="a \"quoted\" } model"} 4'
        '\nvllm:num_requests_waiting_by_reason{reason="capacity"} 99\n'
        'vllm:num_requests_waiting{model="example"} 5 # {trace_id="a b"} 6 123456\n'
    )
    assert admission._prometheus_metric_total(payload) == 12


def test_production_server_cancels_disconnected_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        upstream="http://upstream",
        metrics_url=["http://worker/metrics"],
        max_waiting_requests=10,
        max_concurrent_requests=1,
    )
    run_app = Mock()
    monkeypatch.setattr(admission, "_parser", lambda: Mock(parse_args=lambda: args))
    monkeypatch.setattr(admission.web, "run_app", run_app)
    admission.main()
    assert run_app.call_args.kwargs["handler_cancellation"] is True


def test_aimd_reduces_once_per_cascade_recovers_and_releases_cancelled_waiter() -> None:
    async def run() -> None:
        state = admission.AIMDState(maximum=8, success_window=2, rampup_seconds=0)
        await state.acquire()
        await state.rate_limited(release_permit=True)
        await state.rate_limited(release_permit=False)
        assert state.in_flight == 0
        assert state.current_limit == 6
        assert state.multiplicative_decreases_total == 1
        state.blocked_until = 0
        for _ in range(2):
            await state.acquire()
            await state.release_success()
        assert state.current_limit == 7
        await state.rate_limited(release_permit=False)
        assert state.current_limit == 5
        state.blocked_until = 0
        state.current_limit = 1
        await state.acquire()
        waiter = asyncio.create_task(state.acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert state.in_flight == 1
        await state.release_failure()
        await asyncio.wait_for(state.acquire(), timeout=1)
        await state.release_success()
        assert state.in_flight == 0

    asyncio.run(run())


def test_proxy_metrics_recovery_uneven_queues_retries_and_shutdown(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admission, "_POLL_INTERVAL_SECONDS", 0.01)

    async def run() -> None:  # noqa: PLR0915
        queues = [0, 0]
        missing = [True]
        upstream_status = [200]
        forwarded = []

        async def metric(request: web.Request) -> web.Response:
            index = int(request.match_info["index"])
            if index == 1 and missing[0]:
                return web.Response(text="unrelated_metric 0")
            return web.Response(text=f"vllm:num_requests_waiting {queues[index]}")

        async def inference(request: web.Request) -> web.Response:
            forwarded.append(await request.json())
            return web.json_response({"result": "ok"}, status=upstream_status[0], headers={"Retry-After": "7"})

        upstream = web.Application()
        upstream.router.add_get("/metrics/{index}", metric)
        upstream.router.add_post("/v1/chat/completions", inference)
        async with TestServer(upstream) as server:
            proxy = admission.AdmissionProxy(
                argparse.Namespace(
                    upstream=str(server.make_url("/")),
                    metrics_url=[str(server.make_url(f"/metrics/{i}")) for i in range(2)],
                    max_waiting_requests=10,
                    max_concurrent_requests=8,
                )
            )
            proxy.aimd = admission.AIMDState(maximum=8, rampup_seconds=0)
            app = web.Application()
            app.router.add_route("*", "/{path_info:.*}", proxy.handle)
            app.on_startup.append(proxy.start)
            app.on_cleanup.append(proxy.close)

            async def wait_snapshot(expected: float | None) -> None:
                async with asyncio.timeout(2):
                    while proxy.sampler.snapshot()[0] != expected:  # noqa: ASYNC110
                        await asyncio.sleep(0.01)

            async with TestClient(TestServer(app)) as client:
                response = await client.post("/v1/chat/completions", json={"model": "m"})
                assert response.status == 503
                assert response.headers["Retry-After"] == "2"
                assert forwarded == []
                missing[0] = False
                await wait_snapshot(0)
                response = await client.post("/v1/chat/completions", json={"model": "m"})
                assert response.status == 200
                assert forwarded == [{"model": "m"}]
                queues[0] = 11
                await wait_snapshot(11)
                response = await client.post("/v1/chat/completions", json={"model": "m"})
                assert response.status == 429
                assert len(forwarded) == 1
                assert proxy.aimd.current_limit == 6
                queues[0] = 0
                await wait_snapshot(0)
                proxy.aimd.blocked_until = 0
                upstream_status[0] = 429
                response = await client.post("/v1/chat/completions", json={"model": "m"})
                assert response.status == 429
                assert response.headers["Retry-After"] == "7"
                assert proxy.upstream_429_total == 1
                assert proxy.aimd.in_flight == 0
                missing[0] = True
                await wait_snapshot(None)
                response = await client.get("/ready")
                assert response.status == 503
                response = await client.get("/metrics")
                assert response.status == 200
                assert "curator_admission_queue_metric_fresh 0" in await response.text()
            assert proxy.sampler._task.done()
            assert proxy.session.closed

    asyncio.run(run())


def test_stale_metrics_and_request_cancellation_release_capacity() -> None:
    async def run() -> None:
        proxy = admission.AdmissionProxy(
            argparse.Namespace(
                upstream="http://upstream",
                metrics_url=["http://worker/metrics"],
                max_waiting_requests=10,
                max_concurrent_requests=1,
            )
        )
        proxy.aimd = admission.AIMDState(maximum=1, rampup_seconds=0)
        proxy.sampler.value = 0
        proxy.sampler.sampled_at = time.monotonic() - admission._STALE_AFTER_SECONDS - 1
        request = Mock(path="/v1/chat/completions", headers={}, rel_url="/v1/chat/completions")
        assert (await proxy.handle(request)).status == 503
        proxy.sampler.sampled_at = time.monotonic()
        entered = asyncio.Event()

        @contextlib.asynccontextmanager
        async def pending_request(*_args: object, **_kwargs: object) -> AsyncIterator[None]:
            entered.set()
            await asyncio.Event().wait()
            yield

        proxy.session = Mock(request=pending_request)
        task = asyncio.create_task(proxy.handle(request))
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert proxy.aimd.in_flight == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proxy.aimd.in_flight == 0
        await asyncio.wait_for(proxy.aimd.acquire(), timeout=1)
        await proxy.aimd.release_failure()

    asyncio.run(run())


def test_disconnected_http_client_releases_forwarding_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "_POLL_INTERVAL_SECONDS", 0.01)

    async def run() -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def metric(_request: web.Request) -> web.Response:
            return web.Response(text="vllm:num_requests_waiting 0")

        async def inference(request: web.Request) -> web.Response:
            await request.read()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        upstream = web.Application()
        upstream.router.add_get("/metrics", metric)
        upstream.router.add_post("/v1/chat/completions", inference)
        async with TestServer(upstream) as server:
            proxy = admission.AdmissionProxy(
                argparse.Namespace(
                    upstream=str(server.make_url("/")),
                    metrics_url=[str(server.make_url("/metrics"))],
                    max_waiting_requests=10,
                    max_concurrent_requests=1,
                )
            )
            app = web.Application()
            app.router.add_route("*", "/{path_info:.*}", proxy.handle)
            app.on_startup.append(proxy.start)
            app.on_cleanup.append(proxy.close)
            async with TestServer(app) as gateway:
                async with asyncio.timeout(2):
                    while not proxy.sampler.snapshot()[1]:  # noqa: ASYNC110
                        await asyncio.sleep(0.01)
                _, writer = await asyncio.open_connection(gateway.host, gateway.port)
                try:
                    writer.write(
                        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
                    )
                    await writer.drain()
                    await asyncio.wait_for(entered.wait(), timeout=2)
                    assert proxy.aimd.in_flight == 1
                finally:
                    writer.close()
                    await writer.wait_closed()
                await asyncio.wait_for(cancelled.wait(), timeout=2)
                async with asyncio.timeout(2):
                    while proxy.aimd.in_flight != 0:  # noqa: ASYNC110
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(proxy.aimd.acquire(), timeout=1)
                await proxy.aimd.release_failure()
            assert proxy.session.closed

    asyncio.run(run())


def test_upstream_timeout_preserves_error_and_releases_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        proxy = admission.AdmissionProxy(
            argparse.Namespace(
                upstream="http://upstream",
                metrics_url=["http://worker/metrics"],
                max_waiting_requests=10,
                max_concurrent_requests=1,
            )
        )
        proxy.sampler.value = 0
        proxy.sampler.sampled_at = time.monotonic()
        error = TimeoutError("upstream timed out")
        monkeypatch.setattr(proxy, "_forward", AsyncMock(side_effect=error))
        with pytest.raises(TimeoutError) as raised:
            await proxy.handle(Mock(path="/v1/chat/completions"))
        assert raised.value is error
        assert proxy.aimd.in_flight == 0
        await asyncio.wait_for(proxy.aimd.acquire(), timeout=1)
        await proxy.aimd.release_failure()

    asyncio.run(run())
