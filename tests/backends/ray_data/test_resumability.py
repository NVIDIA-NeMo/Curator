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
