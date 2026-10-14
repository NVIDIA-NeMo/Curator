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

import asyncio
import threading

import pytest

from nemo_curator.models.client.async_runner import PersistentEventLoop


def test_repeated_calls_share_loop_and_shutdown_joins_thread() -> None:
    runner = PersistentEventLoop("test")
    runner.start()
    thread = runner._thread

    async def loop_identity() -> tuple[asyncio.AbstractEventLoop, int]:
        return asyncio.get_running_loop(), threading.get_ident()

    try:
        first = runner.run(loop_identity(), timeout=1)
        runner.start()
        assert runner.run(loop_identity(), timeout=1) == first
        assert first[1] != threading.get_ident()
    finally:
        runner.close(timeout=1)
    assert not thread.is_alive()
    runner.close(timeout=1)


def test_deadline_cancels_coroutine_and_runner_can_be_closed() -> None:
    cancelled = threading.Event()

    async def wait() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runner = PersistentEventLoop("test")
    runner.start()
    try:
        with pytest.raises(TimeoutError, match="batch exceeded"):
            runner.run(wait(), timeout=0.05)
        assert cancelled.wait(timeout=1)
    finally:
        runner.close(timeout=1)


def test_request_timeout_is_not_replaced_by_batch_timeout() -> None:
    async def fail() -> None:
        message = "request failed"
        raise TimeoutError(message)

    runner = PersistentEventLoop("test")
    runner.start()
    try:
        with pytest.raises(TimeoutError, match="request failed"):
            runner.run(fail(), timeout=1)
    finally:
        runner.close(timeout=1)


def test_closed_runner_rejects_and_closes_unscheduled_coroutine() -> None:
    runner = PersistentEventLoop("test")
    coroutine = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="not running"):
        runner.run(coroutine, timeout=1)
    assert coroutine.cr_frame is None


def test_shutdown_cancels_outstanding_work() -> None:
    cancelled = threading.Event()

    async def pending() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def spawn() -> asyncio.Task:
        task = asyncio.create_task(pending())
        await asyncio.sleep(0)
        return task

    runner = PersistentEventLoop("test")
    runner.start()
    task = runner.run(spawn(), timeout=1)
    runner.close(timeout=1)
    assert cancelled.is_set()
    assert task.cancelled()
