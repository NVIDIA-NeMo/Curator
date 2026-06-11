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

"""Tests for the generic ``BucketedInferenceStage`` base via a minimal non-audio subclass."""

from __future__ import annotations

import numpy as np

from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.audio.inference.bucketed_stage import BucketedInferenceStage
from nemo_curator.tasks import AudioTask


class _SumBucketStage(BucketedInferenceStage):
    """Minimal stage: fan ``vals`` into items, x10 each, then sum results back per parent task."""

    name = "test_sum_bucket"

    def process(self, task: AudioTask) -> AudioTask:
        raise NotImplementedError

    def build_items(self, tasks: list[AudioTask]) -> tuple[list[float], list[int]]:
        items: list[float] = []
        parent_of: list[int] = []
        for i, t in enumerate(tasks):
            for v in t.data["vals"]:
                items.append(v)
                parent_of.append(i)
        return items, parent_of

    def item_cost(self, item: float) -> float:
        return float(item)

    def run_inference(self, items: list[float]) -> list[float]:
        self.calls.append(list(items))
        return [v * 10 for v in items]

    def assemble(
        self,
        tasks: list[AudioTask],
        items: list[float],
        parent_of: list[int],
        results: list[float],
    ) -> list[AudioTask]:
        sums = [0.0 for _ in tasks]
        for r, p in zip(results, parent_of, strict=True):
            sums[p] += r
        for t, s in zip(tasks, sums, strict=True):
            t.data["out"] = s
        return tasks


def test_bucketed_inference_stage_fans_out_buckets_and_reassembles() -> None:
    """The base drives build_items -> bucketed dispatch -> assemble, one output per input task."""
    stage = _SumBucketStage()
    stage.calls = []
    stage.batch_policy = BatchPolicy(
        buckets_sec=[0, 30],
        max_items_per_batch_by_bucket=[8, 8],
        max_audio_sec_per_batch=None,
    )
    t0 = AudioTask(data={"vals": [5.0, 100.0]})  # short + long -> two buckets
    t1 = AudioTask(data={"vals": [10.0]})  # short

    out = stage.process_batch([t0, t1])

    assert out == [t0, t1]
    assert t0.data["out"] == (5.0 + 100.0) * 10
    assert t1.data["out"] == 10.0 * 10
    assert len(stage.calls) == 2  # one dispatch per occupied bucket


def test_bucketed_inference_stage_empty_batch_short_circuits() -> None:
    stage = _SumBucketStage()
    stage.calls = []
    assert stage.process_batch([]) == []
    assert stage.calls == []


def test_bucketed_inference_stage_accepts_numpy_task_batch() -> None:
    """Ray Data passes ``map_batches`` columns as ndarrays — not Python lists."""
    stage = _SumBucketStage()
    stage.calls = []
    t0 = AudioTask(data={"vals": [2.0]})
    batch = np.array([t0], dtype=object)
    out = stage.process_batch(batch)
    assert out == [t0]
    assert t0.data["out"] == 20.0
    assert len(stage.calls) == 1
