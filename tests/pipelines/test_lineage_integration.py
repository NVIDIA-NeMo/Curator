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
"""End-to-end lineage-checkpoint tests.

Drive a pipeline through the default :meth:`ProcessingStage.process_batch`
(which calls :func:`assign_child_lineage` + :func:`record_lineage` separately)
while a real :class:`LineageWriterActor` is registered, and verify the
resulting on-disk DAG matches the topology. Without an actor, recording is a
true no-op.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import ray

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage, assign_child_lineage
from nemo_curator.tasks import Task
from nemo_curator.utils.lineage_store import (
    LINEAGE_ACTOR_NAME,
    LineageWriterActor,
    _path_to_udid,
    record_lineage,
)


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


@dataclass
class _FanOut(ProcessingStage[_SimpleTask, _SimpleTask]):
    times: int = 3
    name: str = "fanout"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> list[_SimpleTask]:
        return [
            _SimpleTask(task_id=f"{task.task_id}_{i}", dataset_name=task.dataset_name, data=task.data)
            for i in range(self.times)
        ]


@dataclass
class _Passthrough(ProcessingStage[_SimpleTask, _SimpleTask]):
    name: str = "passthrough"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        return _SimpleTask(task_id=f"{task.task_id}_pt", dataset_name=task.dataset_name, data=task.data)


@dataclass
class _FanIn(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Override ``process_batch`` to combine the whole batch into one output.
    Demonstrates the multi-parent path of the lineage contract — separate
    :func:`assign_child_lineage` and :func:`record_lineage` calls."""

    name: str = "fanin"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        _ = task
        msg = "FanIn only supports batched execution"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[_SimpleTask]) -> list[_SimpleTask]:
        combined: list[int] = []
        for t in tasks:
            combined.extend(t.data)
        merged = _SimpleTask(task_id="merged", dataset_name=tasks[0].dataset_name, data=combined)
        children = assign_child_lineage([t._lineage_path for t in tasks], merged)
        record_lineage([t._udid for t in tasks], [c._udid for c in children])
        return children


def _drive(pipeline: Pipeline, initial_tasks: list[Task]) -> list[Task]:
    current = initial_tasks
    for stage in pipeline.stages:
        current = BaseStageAdapter(stage).process_batch(current)
    return current


def _kill_actor_if_present() -> None:
    with contextlib.suppress(ValueError):
        handle = ray.get_actor(LINEAGE_ACTOR_NAME)
        ray.kill(handle)


@pytest.fixture
def actor(tmp_path: Path, shared_ray_client: None) -> tuple[object, Path]:  # noqa: ARG001
    """Spawn a real :class:`LineageWriterActor` so ``record_lineage`` has somewhere
    to write."""
    _kill_actor_if_present()
    path = tmp_path / "lineage.mdb"
    handle = LineageWriterActor.options(
        name=LINEAGE_ACTOR_NAME,
        get_if_exists=True,
    ).remote(path=str(path))
    try:
        yield handle, path
    finally:
        with contextlib.suppress(Exception):
            ray.get(handle.close.remote())
        ray.kill(handle)


def test_fanout_passthrough_fanin_records_full_dag(actor: tuple[object, Path]) -> None:
    """Drive a 4-stage pipeline and verify the on-disk DAG matches the topology.

        Input ─▶ FanOut(3) ─▶ Passthrough ─▶ FanIn ─▶ Passthrough ─▶ Output
    """
    actor_handle, _ = actor
    pipeline = Pipeline(
        name="fanout_fanin",
        stages=[_FanOut(times=3), _Passthrough(name="pt1"), _FanIn(), _Passthrough(name="pt2")],
    )
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])
    final = _drive(pipeline, [root])

    records = dict(ray.get(actor_handle.iter_records.remote()))

    # 3 FanOut outputs, 3 Passthrough-1 outputs, 1 FanIn output, 1 Passthrough-2 output = 8 records.
    assert len(records) == 8

    fanout_paths = ["0", "1", "2"]
    pt1_paths = ["0_0", "1_0", "2_0"]
    fanin_path = "0_0_1_0_2_0_0"
    pt2_path = "0_0_1_0_2_0_0_0"

    fanout_udids = {_path_to_udid(p) for p in fanout_paths}
    pt1_udids = {_path_to_udid(p) for p in pt1_paths}
    fanin_udid = _path_to_udid(fanin_path)
    pt2_udid = _path_to_udid(pt2_path)

    # FanOut roots: source (no parents, have children at the next stage).
    for u in fanout_udids:
        rec = records[u]
        assert rec.parents == []
        assert len(rec.children) == 1
        assert rec.task_type == "source"

    # PT1: each has 1 parent (a FanOut output) and 1 child (the FanIn).
    for u in pt1_udids:
        rec = records[u]
        assert len(rec.parents) == 1
        assert rec.parents[0] in fanout_udids
        assert rec.children == [fanin_udid]
        assert rec.task_type == "middle"

    # FanIn: 3 parents (the PT1 tasks), 1 child (the PT2 output).
    fanin_rec = records[fanin_udid]
    assert set(fanin_rec.parents) == pt1_udids
    assert fanin_rec.children == [pt2_udid]
    assert fanin_rec.task_type == "middle"

    # PT2: 1 parent (the FanIn), 0 children → leaf.
    pt2_rec = records[pt2_udid]
    assert pt2_rec.parents == [fanin_udid]
    assert pt2_rec.children == []
    assert pt2_rec.task_type == "leaf"

    # The final returned task should match the leaf in the store.
    assert len(final) == 1
    assert final[0]._udid == pt2_udid


def test_actor_exposes_transitive_traversal(actor: tuple[object, Path]) -> None:
    """The actor surfaces ``get_all_parents`` / ``get_all_children`` for transitive
    DAG inspection. Drives the same fanout/passthrough/fanin/passthrough pipeline
    as :func:`test_fanout_passthrough_fanin_records_full_dag` and walks both
    directions from the leaf and from one source."""
    actor_handle, _ = actor
    pipeline = Pipeline(
        name="fanout_fanin_traverse",
        stages=[_FanOut(times=3), _Passthrough(name="pt1"), _FanIn(), _Passthrough(name="pt2")],
    )
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])
    _drive(pipeline, [root])

    fanout_udids = {_path_to_udid(p) for p in ["0", "1", "2"]}
    pt1_udids = {_path_to_udid(p) for p in ["0_0", "1_0", "2_0"]}
    fanin_udid = _path_to_udid("0_0_1_0_2_0_0")
    pt2_udid = _path_to_udid("0_0_1_0_2_0_0_0")

    # From the final leaf, every upstream node should be reachable.
    ancestors = ray.get(actor_handle.get_all_parents.remote(pt2_udid))
    assert set(ancestors.keys()) == fanout_udids | pt1_udids | {fanin_udid}

    # From one fanout root, descendants are its own pt1 + the shared fanin + pt2.
    one_fanout = next(iter(fanout_udids))
    descendants = ray.get(actor_handle.get_all_children.remote(one_fanout))
    descendant_pt1 = pt1_udids & set(descendants.keys())
    assert len(descendant_pt1) == 1
    assert set(descendants.keys()) == descendant_pt1 | {fanin_udid, pt2_udid}


def test_no_lineage_recording_when_actor_absent(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """Driving a pipeline with no LineageWriterActor registered must not create any LMDB file."""
    _kill_actor_if_present()

    pipeline = Pipeline(name="no_lineage", stages=[_FanOut(times=2), _Passthrough(name="pt")])
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])
    out = _drive(pipeline, [root])
    assert len(out) == 2
    # No files created anywhere by the lineage subsystem.
    assert list(tmp_path.iterdir()) == []
