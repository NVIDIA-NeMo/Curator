# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import hashlib
from dataclasses import dataclass

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task


@dataclass
class SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


@dataclass
class Repeat(ProcessingStage[SimpleTask, SimpleTask]):
    """
    Dummy stage that returns `times` new instances of the incoming task.
    """

    times: int = 3
    name: str = "repeat"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: SimpleTask) -> list[SimpleTask]:
        # Important: construct fresh Task objects so each gets a fresh _uuid
        return [
            SimpleTask(
                task_id=f"{task.task_id}_{i}",
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=task._metadata.copy(),
                _stage_perf=task._stage_perf.copy(),
            )
            for i in range(self.times)
        ]


def _sample_task() -> SimpleTask:
    return SimpleTask(task_id="t0", dataset_name="test", data=[1, 2, 3])


def test_fanout_tasks_have_unique_uuid():
    task = _sample_task()
    stage = Repeat(times=3)
    output = stage.process(task)

    assert len(output) == 3
    uuids = [t._uuid for t in output]
    assert len(set(uuids)) == 3, f"Expected unique _uuid per task, got {uuids}"


def _sha256_32(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def test_lineage_path_and_udid_format():
    # Empty parent → just the child index
    task = SimpleTask(task_id="root", dataset_name="t", data=[])
    task._set_lineage([], 4)
    assert task._lineage_path == "4"
    assert task._udid == _sha256_32("4")

    # Single non-empty parent
    child = SimpleTask(task_id="c", dataset_name="t", data=[])
    child._set_lineage(["3"], 0)
    assert child._lineage_path == "3_0"
    assert child._udid == _sha256_32("3_0")

    # Multi-parent join
    grandchild = SimpleTask(task_id="g", dataset_name="t", data=[])
    grandchild._set_lineage(["3_0", "4_1"], 2)
    assert grandchild._lineage_path == "3_0_4_1_2"
    assert grandchild._udid == _sha256_32("3_0_4_1_2")


def test_fanout_udid_from_empty_root():
    # Driving through the adapter triggers the default process_batch which
    # calls assign_child_lineage. Parent _lineage_path is "" (no lineage
    # assigned yet), so children get indices as their root paths.
    task = _sample_task()
    output = BaseStageAdapter(Repeat(times=3)).process_batch([task])

    assert [t._lineage_path for t in output] == ["0", "1", "2"]
    assert [t._udid for t in output] == [_sha256_32("0"), _sha256_32("1"), _sha256_32("2")]
    # Original _uuid stays random and unique per task.
    assert len({t._uuid for t in output}) == 3


def test_udid_deterministic_across_runs():
    # Same pipeline run twice over the same input must yield byte-identical
    # _udid / _lineage_path sequences. (`_uuid` will differ because it's a
    # fresh uuid4 each run; that's expected and not what _udid is for.)
    def run_once() -> tuple[list[str], list[str]]:
        task = _sample_task()
        after_first = BaseStageAdapter(Repeat(times=2)).process_batch([task])
        after_second = BaseStageAdapter(Repeat(times=3)).process_batch(after_first)
        return (
            [t._lineage_path for t in after_second],
            [t._udid for t in after_second],
        )

    paths_a, udids_a = run_once()
    paths_b, udids_b = run_once()
    assert paths_a == paths_b
    assert udids_a == udids_b
    # Sanity: lineage paths follow the documented "{parent_idx}_{child_idx}" shape.
    assert paths_a == [f"{i}_{j}" for i in range(2) for j in range(3)]
