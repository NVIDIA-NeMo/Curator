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

"""Integration tests for pipeline resumability with the ray_data backend.

These tests spin up a real (local) Ray cluster and verify that:
  - Completed source partitions are skipped on resume.
  - The checkpoint recorder writes completion entries for all leaf tasks.
  - A pipeline without a source stage raises ValueError when checkpoint_path is set.
  - Fan-out pipelines resume correctly.
"""

from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Minimal test stages
# ---------------------------------------------------------------------------


class AppendColumnStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Adds a constant column — lets us detect which stages ran."""

    resources = Resources(cpus=0.5)

    def __init__(self, column: str, value: str):
        self.name = f"append_{column}"
        self._column = column
        self._value = value

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas().copy()
        df[self._column] = self._value
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=dict(task._metadata),
            _stage_perf=task._stage_perf,
        )


class SourceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Minimal stage that marks itself as a source and sets resumability_key."""

    name = "source_stage"
    resources = Resources(cpus=0.5)

    def __init__(self, partition_suffix: str = ""):
        self._suffix = partition_suffix

    def is_source_stage(self) -> bool:
        return True

    def process(self, task: DocumentBatch) -> DocumentBatch:
        key = f"source_partition_{task.task_id}{self._suffix}"
        task._metadata["resumability_key"] = key
        return task


class FanOutStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Splits each task into N tasks."""

    resources = Resources(cpus=0.5)

    def __init__(self, factor: int = 3):
        self.name = f"fanout_{factor}"
        self._factor = factor

    def process(self, task: DocumentBatch) -> list[DocumentBatch]:
        df = task.to_pandas()
        return [
            DocumentBatch(
                task_id=f"{task.task_id}_fan{i}",
                dataset_name=task.dataset_name,
                data=df.copy(),
                _metadata=dict(task._metadata),
            )
            for i in range(self._factor)
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tasks(n: int = 3) -> list[DocumentBatch]:
    return [
        DocumentBatch(
            task_id=f"t{i}",
            dataset_name="test",
            data=pd.DataFrame({"text": [f"hello {i}"]}),
        )
        for i in range(n)
    ]


def _build_pipeline(stages: list[ProcessingStage]) -> Pipeline:
    p = Pipeline(name="test_resumability")
    for s in stages:
        p.add_stage(s)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_source_stage_raises(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """Pipeline.run() with checkpoint_path but no is_source_stage() must raise."""
    pipeline = _build_pipeline([AppendColumnStage("col_a", "A")])
    executor = RayDataExecutor()
    with pytest.raises(ValueError, match="no source stage found"):
        pipeline.run(executor, initial_tasks=_make_tasks(2), checkpoint_path=str(tmp_path))


def test_completed_partitions_skipped_on_resume(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """Source partitions that were fully processed should be skipped on the second run."""
    checkpoint_dir = str(tmp_path / "ckpt")
    stages = [SourceStage(), AppendColumnStage("col_a", "A")]
    initial = _make_tasks(4)

    # First run: all partitions processed.
    pipeline = _build_pipeline(stages)
    executor = RayDataExecutor()
    output1 = pipeline.run(executor, initial_tasks=initial, checkpoint_path=checkpoint_dir)
    assert len(output1) == 4

    # Second run: all partitions already checkpointed — output should still be 4 tasks
    # but no stage should re-process them (they're filtered by _CheckpointFilterStage).
    pipeline2 = _build_pipeline([SourceStage(), AppendColumnStage("col_a", "A")])
    executor2 = RayDataExecutor()
    output2 = pipeline2.run(executor2, initial_tasks=initial, checkpoint_path=checkpoint_dir)
    # Resumed run returns 0 tasks (all filtered) since the pipeline has no output for skipped tasks.
    assert len(output2) == 0


def test_checkpoint_written_for_each_source_partition(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """After a complete run, each source partition should have a completion entry."""
    checkpoint_dir = str(tmp_path / "ckpt2")
    n_tasks = 3
    initial = _make_tasks(n_tasks)

    pipeline = _build_pipeline([SourceStage(), AppendColumnStage("col_a", "A")])
    executor = RayDataExecutor()
    pipeline.run(executor, initial_tasks=initial, checkpoint_path=checkpoint_dir)

    mgr = CheckpointManager(checkpoint_dir)
    for task in initial:
        key = f"source_partition_{task.task_id}"
        assert mgr.is_task_completed(key), f"Partition {key!r} not marked complete"


def test_fanout_pipeline_completes_all_partitions(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """Fan-out: one source task → 3 leaf tasks.  All must complete for partition to be done."""
    checkpoint_dir = str(tmp_path / "ckpt_fan")
    factor = 3
    initial = _make_tasks(2)  # 2 source tasks → 6 leaf tasks total

    pipeline = _build_pipeline([SourceStage(), FanOutStage(factor), AppendColumnStage("col_a", "A")])
    executor = RayDataExecutor()
    output = pipeline.run(executor, initial_tasks=initial, checkpoint_path=checkpoint_dir)
    assert len(output) == 2 * factor

    mgr = CheckpointManager(checkpoint_dir)
    for task in initial:
        key = f"source_partition_{task.task_id}"
        assert mgr.is_task_completed(key), f"Fan-out partition {key!r} not complete"


def test_fanout_resume_skips_completed_partitions(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """After a full fan-out run, a resumed run should skip all partitions."""
    checkpoint_dir = str(tmp_path / "ckpt_fan_resume")
    factor = 2
    initial = _make_tasks(3)

    pipeline = _build_pipeline([SourceStage(), FanOutStage(factor), AppendColumnStage("col_a", "A")])
    executor = RayDataExecutor()
    out1 = pipeline.run(executor, initial_tasks=initial, checkpoint_path=checkpoint_dir)
    assert len(out1) == 3 * factor

    # Resume: should produce 0 tasks (all partitions already done)
    pipeline2 = _build_pipeline([SourceStage(), FanOutStage(factor), AppendColumnStage("col_a", "A")])
    executor2 = RayDataExecutor()
    out2 = pipeline2.run(executor2, initial_tasks=initial, checkpoint_path=checkpoint_dir)
    assert len(out2) == 0


class _ActorPassThroughStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Pass-through with an overridden ``setup`` so ``is_actor_stage()`` is True.

    Ray-Data task-style stages skip ``BaseStageAdapter.setup``, which leaves
    ``_checkpoint_actor`` unset and disables ``_drop_completed_inputs`` on the
    last user stage. Forcing actor-style execution wires the actor up so the
    leaf-skip path actually runs in this test.
    """

    name = "actor_passthrough"
    resources = Resources(cpus=0.5)

    def setup(self, _worker_metadata: object | None = None) -> None:
        return

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return task


def test_resume_skips_completed_leaves_after_reset(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """A partition with already-completed leaves must not re-run them on resume.

    Pre-populates the checkpoint DB with state mimicking an interrupted run: each
    partition has expected=factor (as if fan-out fired), one of the fan-out leaves
    is already marked complete, and the partition is NOT finalized.  On resume,
    ``reset_partition`` must preserve those leaf entries; the next run's fan-out
    will re-register expected, and ``_drop_completed_inputs`` at the last user
    stage will drop the pre-completed leaves.
    """
    checkpoint_dir = str(tmp_path / "ckpt")
    factor = 3
    initial = _make_tasks(2)

    # Leaf-task_key shape produced by the propagation chain
    # source -> _CheckpointFilterStage -> FanOutStage:
    # each 1->1 stage appends "::0", and fan-out appends "::i".  The pre-completed
    # leaf corresponds to fan-out index 0 ("_fan0").
    def _preexisting_leaf_key(source_key: str) -> str:
        return f"{source_key}::0::0::0"

    mgr = CheckpointManager(checkpoint_dir)
    try:
        for task in initial:
            source_key = f"source_partition_{task.task_id}"
            mgr.init_partition(source_key)
            mgr.add_expected(source_key, factor - 1)  # expected = factor
            mgr.mark_completed(_preexisting_leaf_key(source_key), source_key)
            assert not mgr.is_task_completed(source_key), (
                "pre-populated state must look unfinalized so the filter stage routes "
                "this partition through reset_partition on resume"
            )
    finally:
        mgr.close()

    # The last user stage must be actor-style so its adapter goes through
    # BaseStageAdapter.setup and wires up ``_checkpoint_actor``; otherwise
    # ``_drop_completed_inputs`` is skipped and pre-completed leaves are NOT dropped.
    pipeline = _build_pipeline([SourceStage(), FanOutStage(factor), _ActorPassThroughStage()])
    executor = RayDataExecutor()
    out = pipeline.run(executor, initial_tasks=initial, checkpoint_path=checkpoint_dir)

    # Per partition: 1 leaf already done, (factor - 1) leaves run this time.
    assert len(out) == len(initial) * (factor - 1)

    # The pre-completed _fan0 leaves must not appear in the surviving outputs.
    surviving_task_ids = {t.task_id for t in out}
    assert not any(tid.endswith("_fan0") for tid in surviving_task_ids), (
        f"pre-completed _fan0 leaves should have been dropped, got {surviving_task_ids}"
    )

    # Both partitions should now be fully finalized.
    mgr2 = CheckpointManager(checkpoint_dir)
    try:
        for task in initial:
            assert mgr2.is_task_completed(f"source_partition_{task.task_id}")
    finally:
        mgr2.close()


class _RaiseOncePerKeyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """1->1 stage that raises the first time it sees a target task, then succeeds.

    Uses a marker file on disk so "first time" is durable across runs: the first
    in-run encounter touches the marker and raises; the second encounter (or any
    later run) sees the marker and passes the task through.
    """

    name = "raise_once"
    resources = Resources(cpus=0.5)

    def __init__(self, marker_dir: str, target_task_id: str):
        self._marker_dir = marker_dir
        self._target = target_task_id

    def setup(self, _worker_metadata: object | None = None) -> None:
        Path(self._marker_dir).mkdir(parents=True, exist_ok=True)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if task.task_id.startswith(self._target):
            marker = Path(self._marker_dir) / f"{self._target}.raised"
            if not marker.exists():
                marker.touch()
                msg = f"injected failure for {self._target}"
                raise RuntimeError(msg)
        return task


def test_raise_mid_run_resumes(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """Stage raises once mid-run; resume must finalize every partition.

    A one-off error during run 1 (think: network blip, OOM) must not pollute the
    checkpoint, and the next run must complete the affected partition without
    re-doing the partitions that already finalized.
    """
    checkpoint_dir = str(tmp_path / "ckpt_raise")
    marker_dir = str(tmp_path / "markers")
    factor = 2
    initial = _make_tasks(3)
    target = "t1"

    # Run 1: must raise (the raise stage sits after fan-out, so it sees per-leaf inputs).
    pipeline = _build_pipeline(
        [
            SourceStage(),
            FanOutStage(factor),
            _RaiseOncePerKeyStage(marker_dir, target_task_id=target),
            AppendColumnStage("col_a", "A"),
        ],
    )
    executor = RayDataExecutor()
    with pytest.raises(Exception):  # noqa: B017, PT011 — backends wrap exceptions differently
        pipeline.run(executor, initial_tasks=initial, checkpoint_path=checkpoint_dir)

    # Run 2: marker exists, no raise; resume completes the affected partition.
    pipeline2 = _build_pipeline(
        [
            SourceStage(),
            FanOutStage(factor),
            _RaiseOncePerKeyStage(marker_dir, target_task_id=target),
            AppendColumnStage("col_a", "A"),
        ],
    )
    executor2 = RayDataExecutor()
    pipeline2.run(executor2, initial_tasks=_make_tasks(3), checkpoint_path=checkpoint_dir)

    # Every source partition must be finalized after resume.
    mgr = CheckpointManager(checkpoint_dir)
    try:
        for task in initial:
            key = f"source_partition_{task.task_id}"
            assert mgr.is_task_completed(key), f"partition {key!r} not finalized after resume"
    finally:
        mgr.close()


def test_missing_resumability_key_raises(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001
    """A source stage that doesn't set resumability_key must cause a clear error."""

    class BadSourceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
        name = "bad_source"
        resources = Resources(cpus=0.5)

        def is_source_stage(self) -> bool:
            return True

        def process(self, task: DocumentBatch) -> DocumentBatch:
            return task  # does NOT set resumability_key

    pipeline = _build_pipeline([BadSourceStage(), AppendColumnStage("col_a", "A")])
    executor = RayDataExecutor()
    with pytest.raises(Exception, match="resumability_key"):
        pipeline.run(executor, initial_tasks=_make_tasks(1), checkpoint_path=str(tmp_path / "ckpt_bad"))
