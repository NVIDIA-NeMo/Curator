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

"""Regression tests for the resumability/checkpoint subsystem.

This file is the pytest version of a teammate's mock demo. Same 8-stage
shape -- ls (source) -> read mock -> passthrough -> passthrough (batched)
-> fanout -> passthrough -> passthrough (batched) -> write parquet --
exercised against both ``XennaExecutor`` and ``RayDataExecutor`` so we
have backend parity coverage.

Failure / filter modes:

  (1) TRANSIENT failure -- a stage returns ``TransientDrop`` on the
      first attempt and the task on retry (deterministic via on-disk
      marker; the demo used random 10% which would make pytest flaky).
      Intent: resume on a subsequent run should push these through.
      Exercised by ``test_transient_drop_not_marked_permanent``.
  (2) ALWAYS-FAIL -- a deterministic drop (the partition's ``mode`` tag
      is set to ``"always_fail"`` at source). Intent: resume must NOT
      make these succeed; the partition is permanently filtered.
  (3) FILTER -- keep half the rows; the partition completes with fewer
      rows.

Findings from the teammate's review (status under this PR):

  1. RayDataExecutor + ``_propagate_resumability_metadata`` hard-crashed
     because ``input_tasks`` arrived as a numpy ndarray from Ray Data's
     ``map_batches`` and ``if not input_tasks`` raised "truth value of an
     array with more than one element is ambiguous".  **FIXED** -- the
     adapter normalises to ``list`` before any truthiness check.
     Regression coverage: ``test_resumability_full_coverage_batch1[ray_data]``.

  2. ``_propagate_resumability_metadata`` attributed every output's
     task_key to ``input_tasks[0]`` and appended ``::{i}`` by output
     position.  Correct for 1->N fan-out, wrong for an N->N batched
     stage -- output ``i`` should derive from input ``i``, not input 0.
     **FIXED** -- the function is now per-(parent, result) and the
     default ``process_batch`` loop calls it once per parent. Stages
     that override ``process_batch`` for vectorized execution must
     invoke ``self._propagate_resumability_metadata(parent, result)``
     themselves per parent. Regression coverage:
     ``test_resumability_full_coverage_batch4`` on both executors.

  3. The framework couldn't distinguish "filtered out" from
     "transiently failed".  ``None`` returned from ``process`` marks the
     task COMPLETED in the checkpoint DB, so resume saw the partition
     as done and never retried.  **FIXED** -- ``process`` may return
     ``TransientDrop`` (see ``nemo_curator.tasks.TransientDrop``) to
     drop the task this run without marking it complete; resume
     retries. ``None`` keeps its existing "permanent filter" meaning.
     Regression coverage: ``test_transient_drop_not_marked_permanent``.

Tests in this file:

  * ``test_resumability_full_coverage_batch1`` -- modes 2/3 + fan-out at
    ``batch_size=1``. Runs the pipeline twice (initial + resume); asserts
    always-fail produces nothing, filter halves rows, all-mode partitions
    fan out correctly, and the resume run does not re-write any
    partition.
  * ``test_resumability_full_coverage_batch4`` -- same flow with
    ``batch_size=4`` on the batched stage. Validates finding #2's fix
    (per-parent lineage at batched stages).
  * ``test_raise_mid_run_resumes`` -- injects a stage that raises once
    for a target partition; asserts run 1 raises, run 2 completes the
    partition.
  * ``test_transient_drop_not_marked_permanent`` -- the regression for
    finding #3: a stage returning ``TransientDrop`` must NOT finalise
    the partition; resume rescues it.
  * ``test_metadata_strip_partitions_emits_warning`` -- documents
    that stripping ``resumability_*`` keys from ``task._metadata``
    mid-pipeline is rescued by the ``_uuid`` fallback for in-place
    mutation (so partitions still finalize), but the framework emits
    a WARNING at the stripping point so operators have a breadcrumb.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task, TransientDrop, _EmptyTask
from nemo_curator.utils.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor

# --------------------------------------------------------------------------- #
# Constants and mode tags
# --------------------------------------------------------------------------- #

MODE_ALL = "all"
MODE_ALWAYS_FAIL = "always_fail"
MODE_FILTER = "filter"
MODE_TRANSIENT = "transient"

ROWS_PER_TASK = 12
FANOUT_FACTOR = 3
# After fanout: rows_per_task // fanout_factor = 4 children per surviving partition.
CHILDREN_PER_PARTITION = ROWS_PER_TASK // FANOUT_FACTOR

# Source-partition modes. Index = partition id.
DEFAULT_MODES = (MODE_ALWAYS_FAIL, MODE_FILTER, MODE_ALL, MODE_ALL)


# --------------------------------------------------------------------------- #
# Task + stages
# --------------------------------------------------------------------------- #


@dataclass
class RowTask(Task[pd.DataFrame]):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class LsStage(ProcessingStage[_EmptyTask, RowTask]):
    """Source stage: emits one ``RowTask`` per configured partition.

    The mode tag is embedded in ``_metadata['mode']`` so downstream stages
    deterministically choose between drop / filter / pass without any
    random or environment state.
    """

    name = "1_ls"
    resources = Resources(cpus=0.5)

    def __init__(self, modes: tuple[str, ...] = DEFAULT_MODES, rows_per_task: int = ROWS_PER_TASK):
        self._modes = modes
        self._rows_per_task = rows_per_task

    def is_source_stage(self) -> bool:
        return True

    def process(self, _: _EmptyTask) -> list[RowTask]:
        out: list[RowTask] = []
        for i, mode in enumerate(self._modes):
            key = hashlib.sha256(f"partition::{i}::{mode}".encode()).hexdigest()
            out.append(
                RowTask(
                    task_id=f"part_{i}_{mode}",
                    dataset_name="demo",
                    data=pd.DataFrame(
                        {"row_idx": range(self._rows_per_task), "src": [i] * self._rows_per_task},
                    ),
                    _metadata={"resumability_key": key, "resumability_task_key": key, "mode": mode},
                ),
            )
        return out


class ReadMockStage(ProcessingStage[RowTask, RowTask]):
    name = "2_read_mock"
    resources = Resources(cpus=0.5)

    def process(self, task: RowTask) -> RowTask:
        task.data = task.data.assign(value=list(range(len(task.data))))
        return task


class PassThroughStage(ProcessingStage[RowTask, RowTask]):
    """1->1 stage that drops on ``always_fail`` and trims rows on ``filter``."""

    resources = Resources(cpus=0.5)

    def __init__(self, name: str):
        self.name = name

    def process(self, task: RowTask) -> RowTask | None:
        mode = task._metadata.get("mode", MODE_ALL)
        if mode == MODE_ALWAYS_FAIL:
            return None
        if mode == MODE_FILTER:
            n = len(task.data)
            task.data = task.data.iloc[: max(1, n // 2)].reset_index(drop=True)
        return task


class PassThroughBatchedStage(ProcessingStage[RowTask, RowTask]):
    """Same logic as PassThroughStage but overrides ``process_batch``.

    Stages that override ``process_batch`` for vectorized execution must
    call ``self._propagate_resumability_metadata(parent, result)`` per
    parent so the checkpoint system can attribute outputs to inputs.
    """

    resources = Resources(cpus=0.5)

    def __init__(self, name: str, batch_size: int = 1):
        self.name = name
        self.batch_size = batch_size

    def process(self, task: RowTask) -> RowTask:  # required by ABC; unused
        return task

    def process_batch(self, tasks: list[RowTask]) -> list[RowTask]:
        out: list[RowTask] = []
        for task in tasks:
            mode = task._metadata.get("mode", MODE_ALL)
            if mode == MODE_ALWAYS_FAIL:
                self._propagate_resumability_metadata(task, None)
                continue
            if mode == MODE_FILTER:
                n = len(task.data)
                task.data = task.data.iloc[: max(1, n // 2)].reset_index(drop=True)
            self._propagate_resumability_metadata(task, task)
            out.append(task)
        return out


class FanOutStage(ProcessingStage[RowTask, RowTask]):
    name = "5_fanout"
    resources = Resources(cpus=0.5)

    def __init__(self, factor: int):
        self._factor = factor

    def process(self, task: RowTask) -> list[RowTask]:
        rows = len(task.data)
        n_out = max(1, rows // self._factor)
        chunk = max(1, -(-rows // n_out))  # ceil
        out: list[RowTask] = []
        for i in range(n_out):
            sub = task.data.iloc[i * chunk : (i + 1) * chunk].reset_index(drop=True)
            if len(sub) == 0:
                continue
            out.append(
                RowTask(
                    task_id=f"{task.task_id}_fan{i}",
                    dataset_name=task.dataset_name,
                    data=sub,
                    _metadata=dict(task._metadata),
                ),
            )
        return out


class WriteParquetStage(ProcessingStage[RowTask, RowTask]):
    name = "8_write"
    resources = Resources(cpus=0.5)

    def __init__(self, out_dir: str):
        self._out_dir = out_dir

    def setup(self, _worker_metadata: object | None = None) -> None:
        # Overriding setup forces is_actor_stage()=True so the last-user-stage
        # adapter wires up _drop_completed_inputs (mirrors the user's mock).
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)

    def process(self, task: RowTask) -> RowTask:
        path = Path(self._out_dir) / f"{task.task_id}.parquet"
        task.data.to_parquet(path, index=False)
        return task


# --------------------------------------------------------------------------- #
# Helpers for executor parametrization
# --------------------------------------------------------------------------- #


def _make_executor(kind: str) -> BaseExecutor:
    if kind == "xenna":
        return XennaExecutor()
    if kind == "ray_data":
        return RayDataExecutor()
    msg = f"unknown executor kind {kind!r}"
    raise ValueError(msg)


def _build_pipeline(
    out_dir: Path,
    batched_size: int = 1,
    modes: tuple[str, ...] = DEFAULT_MODES,
    extra_stages_after_fanout: list[ProcessingStage] | None = None,
) -> Pipeline:
    p = Pipeline(name="resumability_mock")
    p.add_stage(LsStage(modes=modes))
    p.add_stage(ReadMockStage())
    p.add_stage(PassThroughStage(name="3_passthrough"))
    p.add_stage(PassThroughBatchedStage(name="4_passthrough_batched", batch_size=batched_size))
    p.add_stage(FanOutStage(factor=FANOUT_FACTOR))
    for s in extra_stages_after_fanout or []:
        p.add_stage(s)
    p.add_stage(PassThroughStage(name="6_passthrough"))
    p.add_stage(PassThroughBatchedStage(name="7_passthrough_batched", batch_size=batched_size))
    p.add_stage(WriteParquetStage(out_dir=str(out_dir)))
    return p


def _partition_keys(modes: tuple[str, ...] = DEFAULT_MODES) -> dict[str, str]:
    """Return {mode_name_with_index: resumability_key}, matching ``LsStage``."""
    return {
        f"part_{i}_{mode}": hashlib.sha256(f"partition::{i}::{mode}".encode()).hexdigest()
        for i, mode in enumerate(modes)
    }


EXECUTORS = [
    pytest.param("xenna", id="xenna"),
    pytest.param("ray_data", id="ray_data"),
]


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_resumability_full_coverage_batch1(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """All three modes + fan-out at batch_size=1; converges in 2 runs.

    Run 1 writes parquet for every partition except ``always_fail``.
    Run 2 must not write anything new (all surviving partitions already
    recorded as completed). The checkpoint DB must show every partition
    as ``finalized`` after run 2 — including ``always_fail``, which was
    permanently filtered.
    """
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"
    executor = _make_executor(executor_kind)

    # ---- Run 1 -----------------------------------------------------------
    out1 = _build_pipeline(out_dir, batched_size=1).run(executor, checkpoint_path=str(ckpt_dir))
    assert out1 is not None
    files1 = sorted(out_dir.glob("*.parquet"))

    # MODE_ALWAYS_FAIL: 0 outputs.
    # MODE_FILTER:  fanout produces children, each child further filtered.
    # MODE_ALL:     fanout produces ``CHILDREN_PER_PARTITION`` outputs at full rows.
    assert not any("always_fail" in f.name for f in files1), "always_fail must produce no output"

    n_all = sum(1 for m in DEFAULT_MODES if m == MODE_ALL)
    n_filter = sum(1 for m in DEFAULT_MODES if m == MODE_FILTER)
    files_all = [f for f in files1 if "_all_" in f.name]
    files_filter = [f for f in files1 if "_filter_" in f.name]
    assert len(files_all) == n_all * CHILDREN_PER_PARTITION, (
        f"each MODE_ALL partition should fan out to {CHILDREN_PER_PARTITION} files, "
        f"got {len(files_all)} (expected {n_all * CHILDREN_PER_PARTITION})"
    )
    # MODE_FILTER halves rows in stages 3 and 4 (12 -> 3) before fan-out; the fan-out
    # then yields exactly 1 child per filter partition (3 // FANOUT_FACTOR = 1).
    assert len(files_filter) == n_filter, (
        f"each MODE_FILTER partition should yield 1 surviving child, got {len(files_filter)} (expected {n_filter})"
    )
    assert len(files1) == n_all * CHILDREN_PER_PARTITION + n_filter, (
        f"unexpected total file count after run 1: {len(files1)}"
    )

    # ---- Inspect checkpoint state ----------------------------------------
    keys_by_partition = _partition_keys()
    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for partition_id, key in keys_by_partition.items():
            assert mgr.is_task_completed(key), (
                f"partition {partition_id!r} (key={key!r}) was not finalized after run 1"
            )
    finally:
        mgr.close()

    # ---- Run 2: resume — no new files should be written ------------------
    files_before_run2 = set(out_dir.glob("*.parquet"))
    out2 = _build_pipeline(out_dir, batched_size=1).run(
        _make_executor(executor_kind),
        checkpoint_path=str(ckpt_dir),
    )
    assert out2 is not None
    assert len(out2) == 0, f"resume should produce 0 tasks, got {len(out2)}"
    files_after_run2 = set(out_dir.glob("*.parquet"))
    assert files_before_run2 == files_after_run2, "resume must not write new parquet files"


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_resumability_full_coverage_batch4(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """Same as the batch1 test but the batched stages run at batch_size=4."""
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"

    _build_pipeline(out_dir, batched_size=4).run(
        _make_executor(executor_kind),
        checkpoint_path=str(ckpt_dir),
    )

    keys_by_partition = _partition_keys()
    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for partition_id, key in keys_by_partition.items():
            assert mgr.is_task_completed(key), (
                f"partition {partition_id!r} (key={key!r}) was not finalized after run 1 at batch_size=4"
            )
    finally:
        mgr.close()

    # Resume — no new files.
    files_before = set(out_dir.glob("*.parquet"))
    out2 = _build_pipeline(out_dir, batched_size=4).run(
        _make_executor(executor_kind),
        checkpoint_path=str(ckpt_dir),
    )
    assert out2 is not None
    assert len(out2) == 0
    assert set(out_dir.glob("*.parquet")) == files_before


# --------------------------------------------------------------------------- #
# Raise mid-run, resume completes the rest
# --------------------------------------------------------------------------- #


class _RaiseOncePerKeyStage(ProcessingStage[RowTask, RowTask]):
    """1->1 stage that raises the first time it sees a given partition key, then succeeds.

    Uses a marker directory on disk so the "first time" is durable across runs.
    """

    name = "raise_once"
    resources = Resources(cpus=0.5)

    def __init__(self, marker_dir: str, target_partition_id: str):
        self._marker_dir = marker_dir
        self._target = target_partition_id

    def setup(self, _worker_metadata: object | None = None) -> None:
        Path(self._marker_dir).mkdir(parents=True, exist_ok=True)

    def process(self, task: RowTask) -> RowTask:
        if task.task_id.startswith(self._target):
            marker = Path(self._marker_dir) / f"{self._target}.raised"
            if not marker.exists():
                marker.touch()
                msg = f"injected failure for {self._target}"
                raise RuntimeError(msg)
        return task


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_raise_mid_run_resumes(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """Stage raises once for a specific partition; resume must complete it.

    A one-off error during run 1 (think:
    network blip, OOM) must not pollute the checkpoint and must let the
    next run process the affected partition.
    """
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"
    marker_dir = tmp_path / "markers"

    modes = (MODE_ALL, MODE_ALL, MODE_ALL, MODE_ALL)  # disable always_fail/filter for clarity
    target_partition = "part_2_all"

    # Insert the raise stage after FanOutStage so it gets per-leaf inputs.
    extra = [_RaiseOncePerKeyStage(str(marker_dir), target_partition)]

    pipeline_factory = lambda: _build_pipeline(  # noqa: E731
        out_dir,
        batched_size=1,
        modes=modes,
        extra_stages_after_fanout=extra,
    )

    # ---- Run 1: must raise ----------------------------------------------
    with pytest.raises(Exception):  # noqa: B017, PT011 — backends wrap exceptions differently
        pipeline_factory().run(_make_executor(executor_kind), checkpoint_path=str(ckpt_dir))

    # ---- Run 2: marker exists, no raise; resume completes the rest ------
    pipeline_factory().run(_make_executor(executor_kind), checkpoint_path=str(ckpt_dir))

    keys_by_partition = _partition_keys(modes)
    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for partition_id, key in keys_by_partition.items():
            assert mgr.is_task_completed(key), f"partition {partition_id!r} (key={key!r}) not finalized after resume"
    finally:
        mgr.close()


# --------------------------------------------------------------------------- #
# Transient drop sentinel: dropped on run 1 but rescued on run 2
# --------------------------------------------------------------------------- #


class _TransientDropStage(ProcessingStage[RowTask, RowTask]):
    """Returns ``TransientDrop`` on the first attempt per task, then the task on retry.

    Models a flaky external dependency (e.g. inference server timeout).
    The framework must NOT mark the task complete on the first attempt so
    that resume picks it up on the second attempt.
    """

    name = "transient_drop"
    resources = Resources(cpus=0.5)

    def __init__(self, marker_dir: str):
        self._marker_dir = marker_dir

    def setup(self, _worker_metadata: object | None = None) -> None:
        Path(self._marker_dir).mkdir(parents=True, exist_ok=True)

    def process(self, task: RowTask) -> RowTask | TransientDrop:
        if task._metadata.get("mode") != MODE_TRANSIENT:
            return task
        marker = Path(self._marker_dir) / f"{task.task_id}.seen"
        if not marker.exists():
            marker.touch()
            return TransientDrop(reason="simulated transient failure")
        return task


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_transient_drop_not_marked_permanent(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
    executor_kind: str,
) -> None:
    """A stage returning ``TransientDrop`` must not finalise the partition; resume retries it."""
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"
    marker_dir = tmp_path / "markers"

    modes = (MODE_TRANSIENT, MODE_TRANSIENT, MODE_ALL, MODE_ALL)
    extra = [_TransientDropStage(str(marker_dir))]

    # Run 1: every transient partition is dropped this run via TransientDrop.
    _build_pipeline(out_dir, batched_size=1, modes=modes, extra_stages_after_fanout=extra).run(
        _make_executor(executor_kind),
        checkpoint_path=str(ckpt_dir),
    )

    # Run 2: markers exist → transient stage no longer drops → expect new files.
    files_before_run2 = set(out_dir.glob("*.parquet"))
    _build_pipeline(out_dir, batched_size=1, modes=modes, extra_stages_after_fanout=extra).run(
        _make_executor(executor_kind),
        checkpoint_path=str(ckpt_dir),
    )
    files_after_run2 = set(out_dir.glob("*.parquet"))
    new_files = files_after_run2 - files_before_run2

    transient_partition_ids = [f"part_{i}_{m}" for i, m in enumerate(modes) if m == MODE_TRANSIENT]
    assert new_files, (
        f"resume should have produced new parquet files for transient partitions "
        f"{transient_partition_ids}, got 0 new files"
    )


# --------------------------------------------------------------------------- #
# Stripping resumability_* from _metadata is unsupported (documented + warned)
# --------------------------------------------------------------------------- #


class _MetadataStripperStage(ProcessingStage[RowTask, RowTask]):
    """Pops resumability_* keys from ``task._metadata`` in-place.

    The framework's ``_uuid`` fallback re-stamps ``resumability_key`` on
    in-place outputs (same Task object as input), so partitions still
    finalize. To give operators a breadcrumb, the framework logs a
    WARNING when it detects the strip via ``_propagate_resumability_metadata``.
    """

    name = "metadata_stripper"
    resources = Resources(cpus=0.5)

    def process(self, task: RowTask) -> RowTask:
        task._metadata.pop("resumability_key", None)
        task._metadata.pop("resumability_task_key", None)
        return task


@pytest.mark.parametrize("executor_kind", EXECUTORS)
def test_metadata_strip_partitions_emits_warning(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
    caplog: pytest.LogCaptureFixture,
    executor_kind: str,
) -> None:
    """Stripping resumability_* mid-pipeline must emit a warning.

    The ``_uuid`` fallback in ``BaseStageAdapter._fallback_stamp_parent_task_key``
    rescues in-place mutation stages by re-stamping ``resumability_key`` on
    outputs that share ``Task._uuid`` with the snapshot parent, so partitions
    still finalize. But silently rescuing is dangerous — a stage that creates
    NEW tasks (different ``_uuid``) gets no rescue and resumability silently
    breaks downstream. The framework therefore logs a WARNING at the strip
    point so operators see the breadcrumb.
    """
    out_dir = tmp_path / "out"
    ckpt_dir = tmp_path / "ckpt"

    modes = (MODE_ALL, MODE_ALL, MODE_ALL, MODE_ALL)
    extra = [_MetadataStripperStage()]

    with caplog.at_level("WARNING"):
        _build_pipeline(out_dir, batched_size=1, modes=modes, extra_stages_after_fanout=extra).run(
            _make_executor(executor_kind),
            checkpoint_path=str(ckpt_dir),
        )

    # _uuid fallback rescues in-place mutation, so partitions DO finalize.
    keys_by_partition = _partition_keys(modes)
    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for partition_id, key in keys_by_partition.items():
            assert mgr.is_task_completed(key), (
                f"partition {partition_id!r} should have finalized via the _uuid fallback "
                f"despite the mid-pipeline strip"
            )
    finally:
        mgr.close()

    # And the framework should have logged a warning at the strip point.
    assert any("resumability_key" in rec.message for rec in caplog.records), (
        "expected a WARNING mentioning 'resumability_key' from the strip detection path"
    )
