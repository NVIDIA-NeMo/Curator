# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""A single, long-lived asyncio event loop on a dedicated thread.

Synchronous callers (e.g. a Ray Data ``process_batch``) need to drive an async
client repeatedly. The naive ``asyncio.run(coro)`` per call creates and then
*closes* a fresh event loop every time. A client object (and its connection
pool / ``anyio`` synchronization primitives) created on the first loop is then
reused from later loops whose predecessor is already closed — which can park on
a future bound to the dead loop and **hang forever**, with no timeout firing
because the timeout machinery itself lives on the dead loop.

``PersistentEventLoop`` keeps ONE loop running on a daemon thread for the
caller's lifetime, so the client binds to a single stable loop. ``run()``
submits a coroutine to that loop and blocks (with a wall-clock timeout) for the
result, turning any genuine hang into a loud ``TimeoutError`` instead of a
silent freeze.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

_T = TypeVar("_T")


class PersistentEventLoop:
    """Run one asyncio event loop on a background daemon thread.

    Usage::

        runner = PersistentEventLoop(name="MyStage")
        runner.start()
        result = runner.run(client.do_something(), timeout=60)
        runner.close()
    """

    def __init__(self, name: str = "PersistentEventLoop") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Create the loop and start it running on a daemon thread."""
        if self._thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"{self._name}-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None  # noqa: S101
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, _T], timeout: float | None = None) -> _T:
        """Submit *coro* to the loop and block until it finishes or *timeout*.

        Raises ``TimeoutError`` if the coroutine does not complete within
        *timeout* seconds (the coroutine is cancelled on the loop first), and
        re-raises any exception the coroutine itself raised.
        """
        if self._loop is None or self._thread is None:
            msg = "PersistentEventLoop.run() called before start()"
            raise RuntimeError(msg)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            msg = f"{self._name}: coroutine did not complete within {timeout}s"
            raise TimeoutError(msg) from exc

    def close(self) -> None:
        """Stop the loop and join its thread. Safe to call more than once."""
        if self._loop is not None and self._thread is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()
        self._loop = None
        self._thread = None
