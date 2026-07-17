# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finite, cost-aware policy used by metadata-only global planners."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


@dataclass
class BatchPolicy:
    """Pack model-input items by duration without inspecting their payloads."""

    enabled: bool = True
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 30.0, 120.0, 600.0])
    max_items_per_batch_by_bucket: list[int] = field(default_factory=lambda: [32, 16, 8, 1])
    max_audio_sec_per_batch: float | None = 600.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            msg = "BatchPolicy.enabled must be a bool"
            raise TypeError(msg)
        if not self.buckets_sec or self.buckets_sec[0] != 0:
            msg = "BatchPolicy.buckets_sec must start at 0"
            raise ValueError(msg)
        if any(isinstance(edge, bool) or not isinstance(edge, Real) for edge in self.buckets_sec):
            msg = "BatchPolicy.buckets_sec entries must be numeric"
            raise TypeError(msg)
        if any(right <= left for left, right in zip(self.buckets_sec, self.buckets_sec[1:], strict=False)):
            msg = "BatchPolicy.buckets_sec must be strictly increasing"
            raise ValueError(msg)
        if len(self.max_items_per_batch_by_bucket) != len(self.buckets_sec):
            msg = "BatchPolicy item caps must match the number of buckets"
            raise ValueError(msg)
        if any(
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0 for cap in self.max_items_per_batch_by_bucket
        ):
            msg = "BatchPolicy item caps must be positive integers"
            raise ValueError(msg)
        if self.max_audio_sec_per_batch is not None and (
            isinstance(self.max_audio_sec_per_batch, bool)
            or not isinstance(self.max_audio_sec_per_batch, Real)
            or self.max_audio_sec_per_batch <= 0
        ):
            msg = "BatchPolicy.max_audio_sec_per_batch must be positive or None"
            raise ValueError(msg)

    @property
    def num_buckets(self) -> int:
        return len(self.buckets_sec)

    def bucket_for(self, cost: float) -> int:
        for index in range(self.num_buckets - 1, -1, -1):
            if cost >= self.buckets_sec[index]:
                return index
        return 0

    def dispatch_signature(self, *, cost_unit: str = "seconds") -> str:
        if not self.enabled:
            msg = "An atomic dispatch signature requires an enabled BatchPolicy"
            raise ValueError(msg)
        return json.dumps(
            {
                "buckets": [float(value) for value in self.buckets_sec],
                "cost_unit": cost_unit,
                "item_caps": self.max_items_per_batch_by_bucket,
                "total_cost_cap": self.max_audio_sec_per_batch,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def bucketize_with_costs(
        self,
        items: list[Any],
        costs: list[float],
    ) -> list[tuple[int, list[Any], list[float]]]:
        """Return globally packed batches, ordered by decreasing total cost."""
        if len(items) != len(costs):
            msg = f"Received {len(items)} items but {len(costs)} costs"
            raise ValueError(msg)
        if not items:
            return []
        if not self.enabled:
            return [(0, list(items), [float(cost) for cost in costs])]

        queues: list[list[tuple[Any, float]]] = [[] for _ in range(self.num_buckets)]
        for item, raw_cost in zip(items, costs, strict=True):
            cost = float(raw_cost)
            if cost < 0:
                msg = f"BatchPolicy costs must be non-negative, got {cost}"
                raise ValueError(msg)
            queues[self.bucket_for(cost)].append((item, cost))

        batches: list[tuple[int, list[Any], list[float]]] = []
        for bucket_index, queue in enumerate(queues):
            cap = self.max_items_per_batch_by_bucket[bucket_index]
            current_items: list[Any] = []
            current_costs: list[float] = []
            for item, cost in queue:
                over_items = len(current_items) >= cap
                over_cost = (
                    bool(current_items)
                    and self.max_audio_sec_per_batch is not None
                    and sum(current_costs) + cost > self.max_audio_sec_per_batch
                )
                if over_items or over_cost:
                    batches.append((bucket_index, current_items, current_costs))
                    current_items, current_costs = [], []
                current_items.append(item)
                current_costs.append(cost)
            if current_items:
                batches.append((bucket_index, current_items, current_costs))
        return sorted(batches, key=lambda batch: sum(batch[2]), reverse=True)
