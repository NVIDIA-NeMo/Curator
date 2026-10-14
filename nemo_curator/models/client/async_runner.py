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

"""Drive a worker's async client on one event loop across synchronous batches."""

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class PersistentEventLoop:
    """Keep connection pools on their owning loop, with bounded waits and shutdown."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=f"{self._name}-loop", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        loop = self._loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def run(self, coroutine: Coroutine[Any, Any, _T], *, timeout: float) -> _T:
        if self._loop is None or self._thread is None or not self._thread.is_alive():
            coroutine.close()
            msg = f"{self._name}: event loop is not running"
            raise RuntimeError(msg)
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # Preserve a request's own TimeoutError when it completed before the deadline.
            if future.done():
                return future.result()
            future.cancel()
            msg = f"{self._name}: batch exceeded its {timeout}s deadline"
            raise TimeoutError(msg) from None

    @staticmethod
    async def _cancel_pending() -> None:
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def close(self, *, timeout: float = 30.0) -> None:
        if self._thread is None:
            return
        loop, thread = self._loop, self._thread
        try:
            self.run(self._cancel_pending(), timeout=timeout)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=timeout)
            if thread.is_alive():
                msg = f"{self._name}: event loop did not stop within {timeout}s"
                raise TimeoutError(msg)
            self._loop = None
            self._thread = None
