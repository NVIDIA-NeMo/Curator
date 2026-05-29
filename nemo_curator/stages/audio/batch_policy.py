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

"""Duration-aware batch policy for SDP-V2 audio stages (design doc §0.3).

The policy is a *typed Hydra-instantiable knob bag* the audio stages can
honour when they're given a heterogeneous batch of items (variable audio
duration per item).

The full §0.3 model is "each bucket has its own cross-process_batch queue
with a flush timer", which is a *scheduler-level* feature that requires
Curator-framework support. The stage-side helper below implements the
weaker but immediately-useful within-call invariant the doc's worked
example (§0.3, lines 353-354) describes:

* every item is already ≤ ``ideal_inference_segment_s`` because the stage
  pre-sliced any over-long clip first; and
* whatever multi-task batch Curator hands the stage gets re-partitioned
  into bucket-respecting sub-batches so a single 40-minute sub-chunk
  never ends up in the same vLLM call as a 5-second sub-chunk.

``flush_interval_ms`` is recorded on the dataclass for forward-compat
(framework-level scheduler integration is a follow-up PR) but is NOT
consumed by ``bucketize`` - this helper has no notion of cross-call
time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BatchPolicy:
    """Duration-aware bucketed batching policy (SDP-V2 §0.3).

    The defaults mirror the §6 (Qwen-Omni) values from the design doc, so
    a YAML that just declares ``_target_:
    nemo_curator.stages.audio.batch_policy.BatchPolicy`` with no overrides
    is the same as declaring the full Qwen-Omni policy block.

    Args:
        strategy: Bucketing mode. Currently only ``"duration_bucketed"``
            is implemented (per-item audio-second bucket lookup).
            ``"segment_bucketed"`` and ``"token_bucketed"`` are reserved
            future strategies (see doc lines 360-372).
        buckets_sec: Left-edges of each duration bucket, in audio
            seconds, matching the doc-literal layout
            ``[0, ideal/4, ideal/2, ideal]``. Must be strictly
            increasing and start at ``0``. Bucket ``i`` covers
            ``[buckets_sec[i], buckets_sec[i+1])`` and the last bucket
            covers ``[buckets_sec[-1], +inf)``.
        max_items_per_batch_by_bucket: Per-bucket cap on items per
            sub-batch. Length must equal ``len(buckets_sec)``.
        max_audio_sec_per_batch: Cross-bucket cap on total audio seconds
            per sub-batch (the global "audio-second budget" from §0.3,
            line 349). When set to ``None``, only the per-bucket item
            caps apply.
        flush_interval_ms: Cross-process_batch flush timer (ms). Recorded
            for forward-compat; the stage-side helper does not consume
            this value because it has no notion of cross-call time.
    """

    strategy: str = "duration_bucketed"
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 600.0, 1200.0, 2400.0])
    max_items_per_batch_by_bucket: list[int] = field(default_factory=lambda: [32, 16, 8, 4])
    max_audio_sec_per_batch: float | None = 480.0
    flush_interval_ms: int = 250

    def __post_init__(self) -> None:
        if self.strategy != "duration_bucketed":
            msg = (
                f"BatchPolicy: strategy={self.strategy!r} not yet implemented; "
                "only 'duration_bucketed' is supported (see SDP-V2 design doc §0.3)."
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

    def bucket_for(self, duration_s: float) -> int:
        """Return the bucket index for an item with the given duration.

        Left-edge semantics: a 600 s clip with
        ``buckets_sec=[0, 600, 1200, 2400]`` lands in bucket ``1``
        (covering ``[600, 1200)``). Items at or above the top edge
        clamp into the last bucket - the stage's pre-slicer should have
        already ensured no such item reaches here in well-configured
        deployments, but the last-bucket clamp keeps the helper robust.
        """
        for i in range(self.num_buckets - 1, -1, -1):
            if duration_s >= self.buckets_sec[i]:
                return i
        return 0

    def bucketize(
        self,
        items: list[Any],
        duration_fn: Callable[[Any], float],
    ) -> list[tuple[list[int], list[Any]]]:
        """Re-partition ``items`` into bucket-respecting sub-batches.

        Args:
            items: A flat list of items the stage assembled this call.
            duration_fn: Returns the audio duration (seconds) of one item.

        Returns:
            A list of ``(orig_indices, sub_items)`` tuples; the union of
            the ``orig_indices`` is exactly ``range(len(items))`` and the
            tuples are ordered with all sub-batches of the smallest
            bucket first (the smallest bucket fires fastest, matching
            the doc's expectation that fast sub-chunks don't wait on
            longer ones).

        Per-sub-batch invariants:
            * all items in a sub-batch belong to the same bucket;
            * sub-batch size ≤ ``max_items_per_batch_by_bucket[bucket]``;
            * sub-batch total audio seconds ≤ ``max_audio_sec_per_batch``
              (if set) - or one item, whichever is bigger (a single
              over-long item is its own sub-batch so it always fires).
        """
        if not items:
            return []

        per_bucket: dict[int, list[tuple[int, Any, float]]] = defaultdict(list)
        for i, it in enumerate(items):
            d = float(duration_fn(it))
            per_bucket[self.bucket_for(d)].append((i, it, d))

        sub_batches: list[tuple[list[int], list[Any]]] = []
        for b_idx in sorted(per_bucket.keys()):
            cap_items = self.max_items_per_batch_by_bucket[b_idx]
            cap_audio = self.max_audio_sec_per_batch
            cur_idx: list[int] = []
            cur_items: list[Any] = []
            cur_audio = 0.0
            for orig_i, it, d in per_bucket[b_idx]:
                would_overflow_items = len(cur_items) >= cap_items
                would_overflow_audio = (
                    cap_audio is not None and cur_audio + d > cap_audio and cur_items
                )
                if cur_items and (would_overflow_items or would_overflow_audio):
                    sub_batches.append((cur_idx, cur_items))
                    cur_idx, cur_items, cur_audio = [], [], 0.0
                cur_idx.append(orig_i)
                cur_items.append(it)
                cur_audio += d
            if cur_items:
                sub_batches.append((cur_idx, cur_items))

        return sub_batches
