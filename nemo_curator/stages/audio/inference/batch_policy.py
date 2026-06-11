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

Hydra-instantiable policy for heterogeneous batches. ``bucketize`` re-partitions
items within a single ``process_batch`` so one model call does not mix expensive
and cheap items. Cost is supplied via ``cost_fn``; bucket edges and the per-batch
budget are in the same units (audio seconds for ASR, the default consumer).

``flush_interval_ms`` (cross-call flush timers) is a scheduler-level feature,
stored for forward compatibility but unused by ``bucketize``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BatchPolicy:
    """Cost-bucketed batching policy.

    Defaults match the Qwen-Omni tutorial layout (``buckets_sec=[0, 600, 1200,
    2400]`` when ``ideal_inference_segment_s=2400``).

    Args:
        strategy: Only ``"duration_bucketed"`` is implemented; other values are
            reserved for future use.
        buckets_sec: Strictly-increasing left edges starting at ``0`` (cost
            units). Bucket ``i`` covers ``[buckets_sec[i], buckets_sec[i+1])``;
            the last covers ``[buckets_sec[-1], +inf)``.
        max_items_per_batch_by_bucket: Per-bucket item cap; length must equal
            ``len(buckets_sec)``.
        max_audio_sec_per_batch: Optional per-sub-batch total-cost cap (``None``
            = only item caps apply).
        flush_interval_ms: Cross-call flush timer (ms); recorded for forward
            compat, not consumed by ``bucketize``.
    """

    strategy: str = "duration_bucketed"
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 600.0, 1200.0, 2400.0])
    max_items_per_batch_by_bucket: list[int] = field(default_factory=lambda: [32, 16, 8, 4])
    max_audio_sec_per_batch: float | None = 2400.0
    flush_interval_ms: int = 250

    def __post_init__(self) -> None:
        if self.strategy != "duration_bucketed":
            msg = (
                f"BatchPolicy: strategy={self.strategy!r} not yet implemented; "
                "only 'duration_bucketed' is supported."
            )
            raise ValueError(msg)
        if not self.buckets_sec:
            msg = "BatchPolicy: buckets_sec must contain at least one edge"
            raise ValueError(msg)
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
        if len(self.max_items_per_batch_by_bucket) != len(self.buckets_sec):
            msg = (
                f"BatchPolicy: max_items_per_batch_by_bucket has "
                f"{len(self.max_items_per_batch_by_bucket)} entries but buckets_sec has "
                f"{len(self.buckets_sec)}; lengths must match"
            )
            raise ValueError(msg)
        for cap in self.max_items_per_batch_by_bucket:
            if cap <= 0:
                msg = f"BatchPolicy: every max_items_per_batch_by_bucket entry must be > 0, got {cap}"
                raise ValueError(msg)
        if self.max_audio_sec_per_batch is not None and self.max_audio_sec_per_batch <= 0:
            msg = f"BatchPolicy: max_audio_sec_per_batch must be > 0 (or None), got {self.max_audio_sec_per_batch}"
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
            items: Flat list of items the stage assembled this call.
            cost_fn: Returns the per-item cost (audio seconds by default).

        Returns:
            ``(orig_indices, sub_items)`` tuples whose indices union to
            ``range(len(items))``, ordered smallest-cost bucket first.

        Per-sub-batch invariants:
            * all items share one bucket;
            * size <= ``max_items_per_batch_by_bucket[bucket]``;
            * total cost <= ``max_audio_sec_per_batch`` if set, except a single
              over-cost item is its own sub-batch so it always fires.
        """
        if not items:
            return []

        per_bucket: dict[int, list[tuple[int, Any, float]]] = defaultdict(list)
        for i, it in enumerate(items):
            c = float(cost_fn(it))
            per_bucket[self.bucket_for(c)].append((i, it, c))

        sub_batches: list[tuple[list[int], list[Any]]] = []
        for b_idx in sorted(per_bucket.keys()):
            cap_items = self.max_items_per_batch_by_bucket[b_idx]
            cap_cost = self.max_audio_sec_per_batch
            cur_idx: list[int] = []
            cur_items: list[Any] = []
            cur_cost = 0.0
            for orig_i, it, c in per_bucket[b_idx]:
                would_overflow_items = len(cur_items) >= cap_items
                would_overflow_cost = cap_cost is not None and cur_cost + c > cap_cost
                # ``cur_items`` guard: a single over-cost item still fires as its
                # own sub-batch.
                if cur_items and (would_overflow_items or would_overflow_cost):
                    sub_batches.append((cur_idx, cur_items))
                    cur_idx, cur_items, cur_cost = [], [], 0.0
                cur_idx.append(orig_i)
                cur_items.append(it)
                cur_cost += c
            if cur_items:
                sub_batches.append((cur_idx, cur_items))

        return sub_batches


def run_bucketed(
    items: list[Any],
    run_fn: Callable[[list[Any]], list[Any]],
    *,
    cost_fn: Callable[[Any], float],
    policy: BatchPolicy | None = None,
) -> list[Any]:
    """Dispatch ``run_fn`` over cost-bucketed sub-batches, preserving order.

    The single importable bucketing entry point for GPU inference stages, so
    stages don't re-implement the bucketize -> dispatch -> reassemble loop.
    ``policy=None`` (or empty ``items``) runs a single ``run_fn`` call; otherwise
    each sub-batch is dispatched and results are realigned to ``items`` order so
    callers never see the internal bucket ordering.

    Args:
        items: Flat list of per-item payloads the stage assembled this call.
        run_fn: Runs one sub-batch, returning one result per item (1:1, in order).
        cost_fn: Returns the per-item cost (audio seconds by default).
        policy: Optional bucketing policy; ``None`` runs a single batch.

    Returns:
        Results aligned 1:1 with ``items``.

    Raises:
        RuntimeError: If ``run_fn`` returns a count that mismatches its sub-batch.
    """
    if not items:
        return []

    if policy is not None:
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
