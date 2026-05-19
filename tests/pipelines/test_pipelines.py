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
from unittest.mock import Mock, patch

import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage, assign_child_lineage, assign_root_lineage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task


def test_pipeline_uses_xenna_executor_by_default():
    mock_xenna_instance = Mock()

    with patch("nemo_curator.backends.xenna.XennaExecutor") as mock_xenna_class:
        mock_xenna_class.return_value = mock_xenna_instance

        pipeline = Pipeline(name="test")
        pipeline.add_stage(Mock(spec=ProcessingStage))

        pipeline.run()

        mock_xenna_class.assert_called_once_with()
        mock_xenna_instance.execute.assert_called_once()


def test_logs_info_when_ray_serve_active_with_gpu_stages_non_xenna() -> None:
    """Non-Xenna executors log an info message when Serve is active with GPU stages."""
    gpu_stage = Mock(spec=ProcessingStage)
    gpu_stage.name = "EmbeddingStage"
    gpu_stage.resources = Resources(gpus=1.0)

    with patch("nemo_curator.core.serve.is_inference_server_active", return_value=True):
        mock_executor = Mock()
        pipeline = Pipeline(name="test", stages=[gpu_stage])

        with patch("nemo_curator.pipeline.pipeline.logger") as mock_logger:
            pipeline.run(executor=mock_executor)

            mock_logger.info.assert_called()
            info_msgs = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Ray Serve is active" in msg for msg in info_msgs)
            assert any("EmbeddingStage" in msg for msg in info_msgs)


def test_raises_when_ray_serve_active_with_xenna_and_gpu_stages() -> None:
    """XennaExecutor raises RuntimeError when Serve is active with GPU stages."""
    from nemo_curator.backends.xenna import XennaExecutor

    gpu_stage = Mock(spec=ProcessingStage)
    gpu_stage.name = "EmbeddingStage"
    gpu_stage.resources = Resources(gpus=1.0)

    with patch("nemo_curator.core.serve.is_inference_server_active", return_value=True):
        mock_executor = Mock(spec=XennaExecutor)
        pipeline = Pipeline(name="test", stages=[gpu_stage])

        with pytest.raises(RuntimeError, match="Cannot run XennaExecutor"):
            pipeline.run(executor=mock_executor)


# ---------------------------------------------------------------------------
# Deterministic _udid / _lineage_path end-to-end
# ---------------------------------------------------------------------------


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


@dataclass
class _Repeat(ProcessingStage[_SimpleTask, _SimpleTask]):
    times: int = 3
    name: str = "repeat"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> list[_SimpleTask]:
        return [
            _SimpleTask(
                task_id=f"{task.task_id}_{i}",
                dataset_name=task.dataset_name,
                data=task.data,
            )
            for i in range(self.times)
        ]


def _drive(pipeline: Pipeline, initial_tasks: list[Task]) -> list[Task]:
    """Walk a built pipeline by hand, threading tasks through BaseStageAdapter
    for each stage. This is what every real executor does internally; using it
    here lets us exercise the determinism contract without needing Ray."""
    assign_root_lineage(initial_tasks)
    current = initial_tasks
    for stage in pipeline.stages:
        current = BaseStageAdapter(stage).process_batch(current)
    return current


def test_pipeline_udid_deterministic_across_runs():
    def run_once() -> tuple[list[str], list[str]]:
        pipeline = Pipeline(name="det", stages=[_Repeat(times=2), _Repeat(times=3)])
        pipeline.build()
        root = _SimpleTask(task_id="r", dataset_name="d", data=[1, 2])
        out = _drive(pipeline, [root])
        return [t._lineage_path for t in out], [t._udid for t in out]

    paths_a, udids_a = run_once()
    paths_b, udids_b = run_once()
    assert paths_a == paths_b
    assert udids_a == udids_b
    # Root index "0" is prepended by `assign_root_lineage`; subsequent fan-outs
    # extend the path one segment at a time per the documented
    # "{root_idx}_{child_idx}_{grandchild_idx}" shape.
    assert paths_a == [f"0_{i}_{j}" for i in range(2) for j in range(3)]
    assert udids_a == [hashlib.sha256(p.encode()).hexdigest()[:32] for p in paths_a]


# ---------------------------------------------------------------------------
# Fan-out / passthrough / fan-in topology with explicit expected _udid values
# ---------------------------------------------------------------------------


@dataclass
class _Passthrough(ProcessingStage[_SimpleTask, _SimpleTask]):
    name: str = "passthrough"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        return _SimpleTask(
            task_id=f"{task.task_id}_pt",
            dataset_name=task.dataset_name,
            data=task.data,
        )


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
            _SimpleTask(
                task_id=f"{task.task_id}_{i}",
                dataset_name=task.dataset_name,
                data=task.data,
            )
            for i in range(self.times)
        ]


@dataclass
class _FanIn(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Overrides `process_batch` to combine the whole batch into a single
    output. Demonstrates the multi-parent path of `assign_child_lineage`."""

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
        merged = _SimpleTask(
            task_id="merged",
            dataset_name=tasks[0].dataset_name,
            data=combined,
        )
        return assign_child_lineage([t._lineage_path for t in tasks], merged)


def test_pipeline_udid_fanout_passthrough_fanin_passthrough():
    """End-to-end: a 4-stage pipeline that exercises 1:N, 1:1, N:1, 1:1 and
    verifies the exact `_lineage_path` / `_udid` values at every step.

    Pipeline topology:

        Input ─▶ FanOut(3) ─▶ Passthrough ─▶ FanIn ─▶ Passthrough ─▶ Output

    Starting from one root task assigned ``_lineage_path = "0"`` by
    ``assign_root_lineage``, the framework should produce the following paths:

        After FanOut:      ["0_0", "0_1", "0_2"]
        After Passthrough: ["0_0_0", "0_1_0", "0_2_0"]
        After FanIn:       ["0_0_0_0_1_0_0_2_0_0"]   (all 3 parents + idx 0)
        After Passthrough: ["0_0_0_0_1_0_0_2_0_0_0"]
    """
    pipeline = Pipeline(
        name="fanout_fanin",
        stages=[
            _FanOut(times=3),
            _Passthrough(name="pt1"),
            _FanIn(),
            _Passthrough(name="pt2"),
        ],
    )
    pipeline.build()

    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])
    assign_root_lineage([root])

    # Drive stage-by-stage so we can inspect each intermediate set of tasks.
    after_fanout = BaseStageAdapter(pipeline.stages[0]).process_batch([root])
    after_passthrough_1 = BaseStageAdapter(pipeline.stages[1]).process_batch(after_fanout)
    after_fanin = BaseStageAdapter(pipeline.stages[2]).process_batch(after_passthrough_1)
    after_passthrough_2 = BaseStageAdapter(pipeline.stages[3]).process_batch(after_fanin)

    # Expected paths at every level.
    assert [t._lineage_path for t in after_fanout] == ["0_0", "0_1", "0_2"]
    assert [t._lineage_path for t in after_passthrough_1] == ["0_0_0", "0_1_0", "0_2_0"]
    assert [t._lineage_path for t in after_fanin] == ["0_0_0_0_1_0_0_2_0_0"]
    assert [t._lineage_path for t in after_passthrough_2] == ["0_0_0_0_1_0_0_2_0_0_0"]

    # Expected _udid values are exactly sha256(lineage_path)[:32].
    def udid(path: str) -> str:
        return hashlib.sha256(path.encode()).hexdigest()[:32]

    assert [t._udid for t in after_fanout] == [udid("0_0"), udid("0_1"), udid("0_2")]
    assert [t._udid for t in after_passthrough_1] == [udid("0_0_0"), udid("0_1_0"), udid("0_2_0")]
    assert [t._udid for t in after_fanin] == [udid("0_0_0_0_1_0_0_2_0_0")]
    assert [t._udid for t in after_passthrough_2] == [udid("0_0_0_0_1_0_0_2_0_0_0")]

    # Uniqueness: every task emitted anywhere in the pipeline has a distinct
    # _udid (and a distinct _lineage_path).
    all_tasks = [*after_fanout, *after_passthrough_1, *after_fanin, *after_passthrough_2]
    all_udids = [t._udid for t in all_tasks]
    all_paths = [t._lineage_path for t in all_tasks]
    assert len(set(all_udids)) == len(all_udids)
    assert len(set(all_paths)) == len(all_paths)

    # Determinism: running the same pipeline shape over the same input again
    # yields byte-identical _udid and _lineage_path everywhere.
    pipeline2 = Pipeline(
        name="fanout_fanin",
        stages=[
            _FanOut(times=3),
            _Passthrough(name="pt1"),
            _FanIn(),
            _Passthrough(name="pt2"),
        ],
    )
    pipeline2.build()
    second_run = _drive(pipeline2, [_SimpleTask(task_id="r", dataset_name="d", data=[1])])
    assert [t._lineage_path for t in second_run] == [t._lineage_path for t in after_passthrough_2]
    assert [t._udid for t in second_run] == [t._udid for t in after_passthrough_2]


# ---------------------------------------------------------------------------
# In-place stages (process() returns the same task) preserve lineage
# ---------------------------------------------------------------------------


@dataclass
class _InPlace(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Mutates the input task and returns the same instance — the pattern used
    by ImageEmbeddingStage and ~28 other stages across audio/image/video."""

    name: str = "inplace"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        task.data = [*(task.data or []), 0]
        return task


def test_inplace_stage_preserves_lineage():
    pipeline = Pipeline(
        name="inplace",
        stages=[_Repeat(times=2), _InPlace(name="ip1"), _InPlace(name="ip2")],
    )
    pipeline.build()

    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])
    assign_root_lineage([root])
    after_fanout = BaseStageAdapter(pipeline.stages[0]).process_batch([root])
    after_ip1 = BaseStageAdapter(pipeline.stages[1]).process_batch(after_fanout)
    after_ip2 = BaseStageAdapter(pipeline.stages[2]).process_batch(after_ip1)

    # Fan-out gave the children paths "0_0" and "0_1". The two in-place stages
    # must NOT extend the lineage path — same instances come back unchanged.
    assert [t._lineage_path for t in after_fanout] == ["0_0", "0_1"]
    assert [t._lineage_path for t in after_ip1] == ["0_0", "0_1"]
    assert [t._lineage_path for t in after_ip2] == ["0_0", "0_1"]

    def udid(path: str) -> str:
        return hashlib.sha256(path.encode()).hexdigest()[:32]

    expected_udids = [udid("0_0"), udid("0_1")]
    assert [t._udid for t in after_fanout] == expected_udids
    assert [t._udid for t in after_ip1] == expected_udids
    assert [t._udid for t in after_ip2] == expected_udids

    # Identity check: the in-place stages return the same task instances.
    assert all(a is b for a, b in zip(after_fanout, after_ip1, strict=True))
    assert all(a is b for a, b in zip(after_ip1, after_ip2, strict=True))


# ---------------------------------------------------------------------------
# Multiple root tasks must produce distinct _udid through a 1:1 first stage
# ---------------------------------------------------------------------------


def test_pipeline_udid_no_collision_across_multiple_roots():
    """Multiple root tasks through a 1:1 first stage must produce distinct _udid.

    Without ``assign_root_lineage`` every root carries ``_lineage_path = ""``;
    the empty-string filter in ``_set_lineage`` then collapses all first-stage
    children onto the same path ("0"), so their ``_udid`` collides.
    """
    pipeline = Pipeline(name="multi_root", stages=[_Passthrough(name="pt")])
    pipeline.build()

    roots = [
        _SimpleTask(task_id="r0", dataset_name="d", data=[1]),
        _SimpleTask(task_id="r1", dataset_name="d", data=[2]),
        _SimpleTask(task_id="r2", dataset_name="d", data=[3]),
    ]
    out = _drive(pipeline, roots)

    paths = [t._lineage_path for t in out]
    udids = [t._udid for t in out]
    assert paths == ["0_0", "1_0", "2_0"]
    assert len(set(udids)) == len(udids)
