# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from threading import Event

import pytest

from nemo_curator.pipeline.prefetch import BoundedOneAheadPrefetchIterator


def test_one_ahead_prefetch_overlaps_next_load_and_preserves_order() -> None:
    second_started = Event()
    release_second = Event()

    def load(value: int) -> str:
        if value == 2:
            second_started.set()
            assert release_second.wait(timeout=2.0)
        return f"loaded-{value}"

    iterator = iter(
        BoundedOneAheadPrefetchIterator(
            [1, 2],
            loader=load,
            size_bytes=lambda _value: 4,
            max_inflight_bytes=8,
        )
    )

    assert next(iterator) == (1, "loaded-1")
    assert second_started.wait(timeout=2.0)
    release_second.set()
    assert next(iterator) == (2, "loaded-2")
    with pytest.raises(StopIteration):
        next(iterator)


def test_one_ahead_prefetch_respects_combined_byte_bound() -> None:
    loaded: list[int] = []

    def load(value: int) -> int:
        loaded.append(value)
        return value

    iterator = iter(
        BoundedOneAheadPrefetchIterator(
            [1, 2],
            loader=load,
            size_bytes=lambda _value: 6,
            max_inflight_bytes=10,
        )
    )

    assert next(iterator) == (1, 1)
    assert loaded == [1]
    assert next(iterator) == (2, 2)
    assert loaded == [1, 2]


def test_one_ahead_prefetch_propagates_loader_errors() -> None:
    def load(value: int) -> int:
        if value == 2:
            msg = "cannot load second item"
            raise RuntimeError(msg)
        return value

    iterator = iter(
        BoundedOneAheadPrefetchIterator(
            [1, 2],
            loader=load,
            size_bytes=lambda _value: 1,
            max_inflight_bytes=2,
        )
    )

    assert next(iterator) == (1, 1)
    with pytest.raises(RuntimeError, match="cannot load second item"):
        next(iterator)
