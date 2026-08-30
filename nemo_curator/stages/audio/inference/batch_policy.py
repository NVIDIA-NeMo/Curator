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

"""Finite duration bucketing for audio inference stages.

BatchPolicy only re-partitions the model-input items supplied by one
ASRStage.process_batch call. Every item is assigned from its audio_seconds
value, split by the configured item and duration caps, and returned with its
original index so the stage can restore parent-row order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


@dataclass
class BatchPolicy:
    """Duration-bucketed adapter-call policy.

    Args:
        buckets_sec: Strictly increasing left edges beginning at zero. Bucket
            i covers [buckets_sec[i], buckets_sec[i + 1]); the last bucket has
            no upper bound.
        max_items_per_batch_by_bucket: Per-bucket item caps. Its length must
            equal the number of bucket edges.
        max_audio_sec_per_batch: Optional total audio-duration cap for each
            adapter call. A single item over this cap is still emitted alone.
    """

    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 600.0, 1200.0, 2400.0])
    max_items_per_batch_by_bucket: list[int] = field(default_factory=lambda: [32, 16, 8, 4])
    max_audio_sec_per_batch: float | None = 2400.0

    def __post_init__(self) -> None:
        self._validate_bucket_edges()
        self._validate_batch_caps()

    @property
    def num_buckets(self) -> int:
        return len(self.buckets_sec)

    def bucket_for(self, audio_seconds: float) -> int:
        """Return the left-edge bucket for one audio duration."""
        for index in range(self.num_buckets - 1, -1, -1):
            if audio_seconds >= self.buckets_sec[index]:
                return index
        return 0

    def bucketize(
        self,
        items: list[dict[str, Any]],
    ) -> list[tuple[list[int], list[dict[str, Any]]]]:
        """Plan duration-coherent adapter calls for one finite item list.

        Completed calls are ordered by total audio duration, heaviest first.
        Items within each call retain their relative input order. Returned
        indices let the caller restore the full original order.
        """
        if not items:
            return []

        by_bucket: list[list[tuple[int, dict[str, Any], float]]] = [[] for _ in range(self.num_buckets)]
        for index, item in enumerate(items):
            audio_seconds = self._item_audio_seconds(item)
            by_bucket[self.bucket_for(audio_seconds)].append((index, item, audio_seconds))

        planned: list[tuple[float, list[int], list[dict[str, Any]]]] = []
        for bucket_index, bucket_items in enumerate(by_bucket):
            item_cap = self.max_items_per_batch_by_bucket[bucket_index]
            current_indices: list[int] = []
            current_items: list[dict[str, Any]] = []
            current_audio_seconds = 0.0

            def emit_current() -> None:
                nonlocal current_indices, current_items, current_audio_seconds
                if current_items:
                    planned.append((current_audio_seconds, current_indices, current_items))
                    current_indices = []
                    current_items = []
                    current_audio_seconds = 0.0

            for index, item, audio_seconds in bucket_items:
                duration_overflow = (
                    self.max_audio_sec_per_batch is not None
                    and bool(current_items)
                    and current_audio_seconds + audio_seconds > self.max_audio_sec_per_batch
                )
                if len(current_items) >= item_cap or duration_overflow:
                    emit_current()

                current_indices.append(index)
                current_items.append(item)
                current_audio_seconds += audio_seconds
                duration_cap_reached = (
                    self.max_audio_sec_per_batch is not None and current_audio_seconds >= self.max_audio_sec_per_batch
                )
                if len(current_items) >= item_cap or duration_cap_reached:
                    emit_current()

            emit_current()

        planned.sort(key=lambda batch: batch[0], reverse=True)
        return [(indices, batch_items) for _audio_seconds, indices, batch_items in planned]

    def _validate_bucket_edges(self) -> None:
        if not self.buckets_sec:
            msg = "BatchPolicy: buckets_sec must contain at least one edge"
            raise ValueError(msg)
        for edge in self.buckets_sec:
            if isinstance(edge, bool) or not isinstance(edge, Real):
                msg = f"BatchPolicy: every buckets_sec entry must be numeric, got {type(edge).__name__}"
                raise TypeError(msg)
            if not math.isfinite(float(edge)) or edge < 0:
                msg = f"BatchPolicy: every buckets_sec entry must be finite and non-negative, got {edge}"
                raise ValueError(msg)
        if self.buckets_sec[0] != 0.0:
            msg = f"BatchPolicy: buckets_sec must start at 0.0, got {self.buckets_sec[0]}"
            raise ValueError(msg)
        for left, right in zip(self.buckets_sec, self.buckets_sec[1:], strict=False):
            if right <= left:
                msg = f"BatchPolicy: buckets_sec must be strictly increasing; got {left} -> {right}"
                raise ValueError(msg)

    def _validate_batch_caps(self) -> None:
        if len(self.max_items_per_batch_by_bucket) != self.num_buckets:
            msg = (
                "BatchPolicy: max_items_per_batch_by_bucket has "
                f"{len(self.max_items_per_batch_by_bucket)} entries but buckets_sec has "
                f"{self.num_buckets}; lengths must match"
            )
            raise ValueError(msg)
        for cap in self.max_items_per_batch_by_bucket:
            if isinstance(cap, bool) or not isinstance(cap, int):
                msg = (
                    f"BatchPolicy: every max_items_per_batch_by_bucket entry must be an int, got {type(cap).__name__}"
                )
                raise TypeError(msg)
            if cap <= 0:
                msg = f"BatchPolicy: every max_items_per_batch_by_bucket entry must be > 0, got {cap}"
                raise ValueError(msg)
        if self.max_audio_sec_per_batch is None:
            return
        if isinstance(self.max_audio_sec_per_batch, bool) or not isinstance(self.max_audio_sec_per_batch, Real):
            msg = (
                "BatchPolicy: max_audio_sec_per_batch must be numeric or None, "
                f"got {type(self.max_audio_sec_per_batch).__name__}"
            )
            raise TypeError(msg)
        if not math.isfinite(float(self.max_audio_sec_per_batch)) or self.max_audio_sec_per_batch <= 0:
            msg = (
                "BatchPolicy: max_audio_sec_per_batch must be finite and > 0 "
                f"(or None), got {self.max_audio_sec_per_batch}"
            )
            raise ValueError(msg)

    @staticmethod
    def _item_audio_seconds(item: dict[str, Any]) -> float:
        value = item.get("audio_seconds")
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = f"BatchPolicy: every item must provide numeric audio_seconds, got {type(value).__name__}"
            raise TypeError(msg)
        audio_seconds = float(value)
        if not math.isfinite(audio_seconds) or audio_seconds < 0:
            msg = f"BatchPolicy: item audio_seconds must be finite and non-negative, got {value}"
            raise ValueError(msg)
        return audio_seconds
