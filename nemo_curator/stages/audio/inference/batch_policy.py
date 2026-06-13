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

"""Cost-aware batch policy for GPU inference stages.

Hydra-instantiable policy for heterogeneous batches. Backend adapters use
``bucketize`` to split mixed task batches before calling a stage's
``process_batch``; direct stage calls can use the same helper as a safety
helper before model dispatch. Cost is supplied via ``cost_fn``; bucket edges
and the per-batch budget are in the same units (audio seconds for ASR, the
default consumer).

When ``enabled`` is true, supporting executors form bucket-aware batches before
workers receive them. Stages with fan-out work units should centralize work-unit
creation through executor hooks, while the shared scheduler owns bucket queues
and dispatch batch formation. ``prebatching_window_size`` optionally bounds the
executor candidate window; when unset, executors derive the current default from
the policy's item caps. ``flush_interval_ms`` is used by
``BucketQueueScheduler`` for persistent queues; finite ``bucketize`` plans drain
all queued items at the end of the candidate window.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ReadyBatch:
    """A scheduler-emitted batch that is ready to dispatch to a worker."""

    indices: list[int]
    items: list[object]
    total_cost: float
    bucket_index: int
    flush_reason: str


@dataclass(frozen=True)
class _QueuedItem:
    index: int
    item: object
    cost: float
    enqueued_ms: float


class BucketQueueScheduler:
    """Persistent per-bucket queue scheduler for cost-aware GPU dispatch.

    The scheduler accepts already-costed work units one at a time. Each unit
    enters exactly one bucket queue. A queue flushes when adding work would
    overflow its item/cost budget, when it reaches a budget exactly, when its
    timer expires, or when the caller drains it. This is the shared primitive
    used by finite ``BatchPolicy.bucketize_with_costs`` plans and by executor
    paths that can hold a persistent scheduling window.
    """

    def __init__(self, policy: BatchPolicy, *, enable_timer: bool = True) -> None:
        self.policy = policy
        self._enable_timer = enable_timer
        self._queues: list[deque[_QueuedItem]] = [deque() for _ in range(policy.num_buckets)]
        self._costs: list[float] = [0.0 for _ in range(policy.num_buckets)]
        self._first_enqueued_ms: list[float | None] = [None for _ in range(policy.num_buckets)]

    def enqueue(self, index: int, item: object, cost: float, now_ms: float | None = None) -> list[ReadyBatch]:
        """Queue one item and return any batches that became ready."""
        now = self._now_ms(now_ms) if self._enable_timer else self._static_ms(now_ms)
        ready = self.flush_due(now) if self._enable_timer else []

        bucket_index = self.policy.bucket_for(float(cost))
        if self._would_overflow(bucket_index, float(cost)):
            flushed = self._flush_bucket(bucket_index, "capacity")
            if flushed is not None:
                ready.append(flushed)

        queue = self._queues[bucket_index]
        if not queue:
            self._first_enqueued_ms[bucket_index] = now
        queue.append(_QueuedItem(index=index, item=item, cost=float(cost), enqueued_ms=now))
        self._costs[bucket_index] += float(cost)

        ready_reason = self._ready_reason(bucket_index)
        if ready_reason is not None:
            flushed = self._flush_bucket(bucket_index, ready_reason)
            if flushed is not None:
                ready.append(flushed)
        return ready

    def flush_due(self, now_ms: float | None = None) -> list[ReadyBatch]:
        """Flush queues whose first item has exceeded ``flush_interval_ms``."""
        if not self._enable_timer:
            return []
        interval_ms = float(getattr(self.policy, "flush_interval_ms", 0) or 0)
        if interval_ms <= 0:
            return []

        now = self._now_ms(now_ms)
        ready: list[ReadyBatch] = []
        for bucket_index, first_ms in enumerate(self._first_enqueued_ms):
            if first_ms is not None and now - first_ms >= interval_ms:
                flushed = self._flush_bucket(bucket_index, "timer")
                if flushed is not None:
                    ready.append(flushed)
        return ready

    def flush_all(self, reason: str = "drain") -> list[ReadyBatch]:
        """Drain all non-empty bucket queues."""
        ready: list[ReadyBatch] = []
        for bucket_index in range(self.policy.num_buckets):
            flushed = self._flush_bucket(bucket_index, reason)
            if flushed is not None:
                ready.append(flushed)
        return ready

    def _would_overflow(self, bucket_index: int, cost: float) -> bool:
        queue = self._queues[bucket_index]
        if not queue:
            return False
        item_cap = int(self.policy.max_items_per_batch_by_bucket[bucket_index])
        if len(queue) >= item_cap:
            return True
        cost_cap = self.policy.max_audio_sec_per_batch
        return cost_cap is not None and self._costs[bucket_index] + cost > float(cost_cap)

    def _ready_reason(self, bucket_index: int) -> str | None:
        queue = self._queues[bucket_index]
        item_cap = int(self.policy.max_items_per_batch_by_bucket[bucket_index])
        if len(queue) >= item_cap:
            return "item_cap"
        cost_cap = self.policy.max_audio_sec_per_batch
        if cost_cap is not None and self._costs[bucket_index] >= float(cost_cap):
            return "cost_cap"
        return None

    def _flush_bucket(self, bucket_index: int, reason: str) -> ReadyBatch | None:
        queue = self._queues[bucket_index]
        if not queue:
            return None
        queued = list(queue)
        queue.clear()
        total_cost = self._costs[bucket_index]
        self._costs[bucket_index] = 0.0
        self._first_enqueued_ms[bucket_index] = None
        return ReadyBatch(
            indices=[queued_item.index for queued_item in queued],
            items=[queued_item.item for queued_item in queued],
            total_cost=total_cost,
            bucket_index=bucket_index,
            flush_reason=reason,
        )

    @staticmethod
    def _now_ms(now_ms: float | None) -> float:
        if now_ms is not None:
            return float(now_ms)
        return time.monotonic() * 1000.0

    @staticmethod
    def _static_ms(now_ms: float | None) -> float:
        if now_ms is not None:
            return float(now_ms)
        return 0.0


@dataclass
class BatchPolicy:
    """Cost-bucketed batching policy.

    Defaults match the Qwen-Omni tutorial layout (``buckets_sec=[0, 600, 1200,
    2400]`` when ``ideal_inference_segment_s=2400``).

    Args:
        enabled: When ``False``, the policy is carried in config but backend
            adapters and ``run_bucketed`` dispatch one normal batch, matching
            ``policy=None``.
        strategy: Only ``"duration_bucketed"`` is implemented; other values are
            reserved for future use.
        buckets_sec: Strictly-increasing left edges starting at ``0`` (cost
            units). Bucket ``i`` covers ``[buckets_sec[i], buckets_sec[i+1])``;
            the last covers ``[buckets_sec[-1], +inf)``.
        max_items_per_batch_by_bucket: Per-bucket item cap; length must equal
            ``len(buckets_sec)``.
        max_audio_sec_per_batch: Optional per-sub-batch total-cost cap (``None``
            = only item caps apply).
        prebatching_window_size: Optional executor candidate-window size. ``None``
            preserves the derived default ``sum(max_items_per_batch_by_bucket)``.
        flush_interval_ms: Cross-call queue flush timer (ms). Persistent
            schedulers use it directly; finite ``bucketize`` calls drain at the
            end of the supplied item window.
    """

    enabled: bool = True
    strategy: str = "duration_bucketed"
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 600.0, 1200.0, 2400.0])
    max_items_per_batch_by_bucket: list[int] = field(default_factory=lambda: [32, 16, 8, 4])
    max_audio_sec_per_batch: float | None = 2400.0
    prebatching_window_size: int | None = None
    flush_interval_ms: int = 250

    def __post_init__(self) -> None:
        self._validate_flags()
        if not self.enabled:
            return
        self._validate_strategy()
        self._validate_bucket_edges()
        self._validate_batch_caps()
        self._validate_prebatching_window()

    def _validate_flags(self) -> None:
        if not isinstance(self.enabled, bool):
            msg = f"BatchPolicy: enabled must be a bool, got {type(self.enabled).__name__}"
            raise TypeError(msg)
        if self.prebatching_window_size is not None and (
            isinstance(self.prebatching_window_size, bool) or not isinstance(self.prebatching_window_size, int)
        ):
            msg = (
                f"BatchPolicy: prebatching_window_size must be an int or None, "
                f"got {type(self.prebatching_window_size).__name__}"
            )
            raise TypeError(msg)
        if isinstance(self.flush_interval_ms, bool) or not isinstance(self.flush_interval_ms, int):
            msg = (
                f"BatchPolicy: flush_interval_ms must be an int, "
                f"got {type(self.flush_interval_ms).__name__}"
            )
            raise TypeError(msg)

    def _validate_strategy(self) -> None:
        if self.strategy != "duration_bucketed":
            msg = (
                f"BatchPolicy: strategy={self.strategy!r} not yet implemented; "
                "only 'duration_bucketed' is supported."
            )
            raise ValueError(msg)

    def _validate_bucket_edges(self) -> None:
        if not self.buckets_sec:
            msg = "BatchPolicy: buckets_sec must contain at least one edge"
            raise ValueError(msg)
        for edge in self.buckets_sec:
            if isinstance(edge, bool) or not isinstance(edge, Real):
                msg = f"BatchPolicy: every buckets_sec entry must be numeric, got {type(edge).__name__}"
                raise TypeError(msg)
        if self.buckets_sec[0] != 0.0:
            msg = f"BatchPolicy: buckets_sec must start at 0.0, got {self.buckets_sec[0]}"
            raise ValueError(msg)
        for i in range(len(self.buckets_sec) - 1):
            if self.buckets_sec[i + 1] <= self.buckets_sec[i]:
                msg = (
                    f"BatchPolicy: buckets_sec must be strictly increasing; "
                    f"got {self.buckets_sec[i]} -> {self.buckets_sec[i + 1]}"
                )
                raise ValueError(msg)

    def _validate_batch_caps(self) -> None:
        if len(self.max_items_per_batch_by_bucket) != len(self.buckets_sec):
            msg = (
                f"BatchPolicy: max_items_per_batch_by_bucket has "
                f"{len(self.max_items_per_batch_by_bucket)} entries but buckets_sec has "
                f"{len(self.buckets_sec)}; lengths must match"
            )
            raise ValueError(msg)
        for cap in self.max_items_per_batch_by_bucket:
            if isinstance(cap, bool) or not isinstance(cap, int):
                msg = (
                    f"BatchPolicy: every max_items_per_batch_by_bucket entry must be an int, "
                    f"got {type(cap).__name__}"
                )
                raise TypeError(msg)
            if cap <= 0:
                msg = f"BatchPolicy: every max_items_per_batch_by_bucket entry must be > 0, got {cap}"
                raise ValueError(msg)
        if self.max_audio_sec_per_batch is not None:
            if isinstance(self.max_audio_sec_per_batch, bool) or not isinstance(self.max_audio_sec_per_batch, Real):
                msg = (
                    f"BatchPolicy: max_audio_sec_per_batch must be numeric or None, "
                    f"got {type(self.max_audio_sec_per_batch).__name__}"
                )
                raise TypeError(msg)
            if self.max_audio_sec_per_batch <= 0:
                msg = f"BatchPolicy: max_audio_sec_per_batch must be > 0 (or None), got {self.max_audio_sec_per_batch}"
                raise ValueError(msg)

    def _validate_prebatching_window(self) -> None:
        if self.prebatching_window_size is not None and self.prebatching_window_size <= 0:
            msg = f"BatchPolicy: prebatching_window_size must be > 0 (or None), got {self.prebatching_window_size}"
            raise ValueError(msg)
        if self.flush_interval_ms < 0:
            msg = f"BatchPolicy: flush_interval_ms must be >= 0, got {self.flush_interval_ms}"
            raise ValueError(msg)

    @property
    def num_buckets(self) -> int:
        return len(self.buckets_sec)

    def bucket_for(self, cost: float) -> int:
        """Return the bucket index for an item with the given cost.

        Left-edge semantics: cost 600 with ``[0, 600, 1200, 2400]`` lands in
        bucket 1 (``[600, 1200)``). Items at/above the top edge clamp into the
        last bucket (the pre-slicer should prevent this, but the clamp keeps the
        helper robust).
        """
        for i in range(self.num_buckets - 1, -1, -1):
            if cost >= self.buckets_sec[i]:
                return i
        return 0

    def bucketize(
        self,
        items: list[Any],
        cost_fn: Callable[[Any], float],
    ) -> list[tuple[list[int], list[Any]]]:
        """Re-partition ``items`` into bucket-respecting sub-batches.

        Args:
            items: Flat list of tasks or model-input items to partition.
            cost_fn: Returns the per-item cost (audio seconds by default).

        Returns:
            ``(orig_indices, sub_items)`` tuples whose indices union to
            ``range(len(items))``. The finite planner dispatches heavier
            sub-batches first to reduce multi-worker tail time; results should
            always be realigned by the caller.

        Per-sub-batch invariants:
            * all items share one bucket;
            * size <= ``max_items_per_batch_by_bucket[bucket]``;
            * total cost <= ``max_audio_sec_per_batch`` if set, except a single
              over-cost item is its own sub-batch so it always fires.
        """
        return [
            (orig_indices, sub_items)
            for orig_indices, sub_items, _total_cost in self.bucketize_with_costs(items, cost_fn)
        ]

    def bucketize_with_costs(
        self,
        items: list[Any],
        cost_fn: Callable[[Any], float],
    ) -> list[tuple[list[int], list[Any], float]]:
        """Re-partition ``items`` and return each sub-batch's total cost.

        This is the planning form of :meth:`bucketize`: it computes
        ``cost_fn(item)`` once per item, then carries the accumulated sub-batch
        cost forward so callers can sort or account without re-inspecting
        expensive payloads.
        """
        if not items:
            return []
        if not self.enabled:
            return [(list(range(len(items))), list(items), 0.0)]

        scheduler = BucketQueueScheduler(self, enable_timer=False)
        ready_batches: list[ReadyBatch] = []
        for i, it in enumerate(items):
            ready_batches.extend(scheduler.enqueue(i, it, float(cost_fn(it))))
        ready_batches.extend(scheduler.flush_all())

        return [
            (batch.indices, batch.items, batch.total_cost)
            for batch in sorted(ready_batches, key=lambda batch: batch.total_cost, reverse=True)
        ]


def run_bucketed(
    items: list[Any],
    run_fn: Callable[[list[Any]], list[Any]],
    *,
    cost_fn: Callable[[Any], float],
    policy: BatchPolicy | None = None,
) -> list[Any]:
    """Dispatch ``run_fn`` over cost-bucketed sub-batches, preserving order.

    The importable direct-call helper for GPU inference stages, so stages
    don't re-implement the bucketize -> dispatch -> reassemble loop.
    ``policy=None`` / ``policy.enabled=False`` (or empty ``items``) runs a
    single ``run_fn`` call; otherwise each sub-batch is dispatched and results
    are realigned to ``items`` order so callers never see the internal bucket
    ordering. Scheduler-backed backend execution can pre-bucket stage-specific
    work units before calling the stage.

    Args:
        items: Flat list of per-item payloads the stage assembled this call.
        run_fn: Runs one sub-batch, returning one result per item (1:1, in order).
        cost_fn: Returns the per-item cost (audio seconds by default).
        policy: Optional bucketing policy; ``None`` or disabled runs a single
            batch.

    Returns:
        Results aligned 1:1 with ``items``.

    Raises:
        RuntimeError: If ``run_fn`` returns a count that mismatches its sub-batch.
    """
    if not items:
        return []

    if policy is not None and policy.enabled:
        sub_batches = policy.bucketize(items, cost_fn=cost_fn)
    else:
        sub_batches = [(list(range(len(items))), list(items))]

    results: list[Any] = [None] * len(items)
    for sub_indices, sub_items in sub_batches:
        if not sub_items:
            continue
        sub_results = run_fn(sub_items)
        if len(sub_results) != len(sub_items):
            msg = f"run_fn returned {len(sub_results)} results for {len(sub_items)} items (must match 1:1)"
            raise RuntimeError(msg)
        for i, r in zip(sub_indices, sub_results, strict=True):
            results[i] = r
    return results
