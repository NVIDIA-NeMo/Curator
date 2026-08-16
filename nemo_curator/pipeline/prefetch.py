# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral, bounded one-item lookahead prefetching."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

_WorkT = TypeVar("_WorkT")
_ValueT = TypeVar("_ValueT")


class BoundedOneAheadPrefetchIterator(Generic[_WorkT, _ValueT]):
    """Load work in order while overlapping at most one next item.

    ``max_inflight_bytes`` bounds the estimated bytes retained by the current
    value and its one prefetched successor. An individual item larger than the
    bound is still loaded synchronously so callers can handle or reject it.
    The helper does not know about Ray, payload refs, audio, or model adapters.
    """

    def __init__(
        self,
        work: Iterable[_WorkT],
        *,
        loader: Callable[[_WorkT], _ValueT],
        size_bytes: Callable[[_WorkT], int],
        max_inflight_bytes: int,
        thread_name_prefix: str = "curator-prefetch",
    ) -> None:
        if int(max_inflight_bytes) <= 0:
            msg = "max_inflight_bytes must be positive"
            raise ValueError(msg)
        self._work = work
        self._loader = loader
        self._size_bytes = size_bytes
        self._max_inflight_bytes = int(max_inflight_bytes)
        self._thread_name_prefix = thread_name_prefix

    def __iter__(self) -> Iterator[tuple[_WorkT, _ValueT]]:
        work_iter = iter(self._work)
        try:
            current_work = next(work_iter)
        except StopIteration:
            return

        current_value = self._loader(current_work)
        current_size = self._checked_size(current_work)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._thread_name_prefix)
        pending: Future[_ValueT] | None = None
        try:
            while True:
                try:
                    next_work = next(work_iter)
                except StopIteration:
                    yield current_work, current_value
                    return

                next_size = self._checked_size(next_work)
                if current_size + next_size <= self._max_inflight_bytes:
                    pending = executor.submit(self._loader, next_work)

                yield current_work, current_value

                if pending is None:
                    next_value = self._loader(next_work)
                else:
                    next_value = pending.result()
                    pending = None
                current_work = next_work
                current_value = next_value
                current_size = next_size
        finally:
            if pending is not None:
                pending.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _checked_size(self, work: _WorkT) -> int:
        size = int(self._size_bytes(work))
        if size < 0:
            msg = "size_bytes must return a non-negative value"
            raise ValueError(msg)
        return size
