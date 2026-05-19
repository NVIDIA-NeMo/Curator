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
"""End-to-end resumability test with three failure modes.

Drives an 8-stage pipeline twice through ``pipeline.run(executor,
checkpoint_path=...)`` on the same LMDB checkpoint, parametrized over both
``XennaExecutor`` and ``RayDataExecutor``, and asserts:

* **transient** failures injected on run 1 are rescued on run 2,
* **always-fail** branches leave some records ``completed=False`` even after
  resume, and
* **filter** decisions produce parquet files with the halved row count.

Failure modes are keyed on each task's framework-assigned ``_udid`` so the
decisions are stable across runs. Transient drops are gated by a per-stage
``is_resume_run`` flag — flipping it between runs lets us assert resume
behavior without relying on RNG.
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

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage, assign_child_lineage
from nemo_curator.tasks import Task, _EmptyTask
from nemo_curator.utils.lineage_store import (
    LineageRecord,
    LineageStore,
    mark_leaves_completed,
    record_lineage,
)

NUM_TASKS = 4
ROWS_PER_TASK = 12
FANOUT_FACTOR = 4  # fanout splits each parent into chunks of FANOUT_FACTOR rows


@dataclass
class _RowTask(Task[pd.DataFrame]):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


def _bucket(udid: str) -> int:
    return int(hashlib.sha256(udid.encode()).hexdigest(), 16) % 6


def _decision(udid: str) -> str:
    """Deterministic per-task outcome — stable across runs."""
    return {0: "always_fail", 1: "filter"}.get(_bucket(udid), "all")


def _is_transient(udid: str) -> bool:
    """Drops ~1/6 of tasks on the run that has transients enabled."""
    return _bucket(udid) == 2


@dataclass
class _LsStage(ProcessingStage[_EmptyTask, _RowTask]):
    name: str = "1_ls"
    num_tasks: int = NUM_TASKS
    rows_per_task: int = ROWS_PER_TASK

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, _: _EmptyTask) -> list[_RowTask]:
        return [
            _RowTask(
                task_id=f"part_{i}",
                dataset_name="demo",
                data=pd.DataFrame({"row_idx": list(range(self.rows_per_task)), "src": [i] * self.rows_per_task}),
            )
            for i in range(self.num_tasks)
        ]


@dataclass
class _ReadMockStage(ProcessingStage[_RowTask, _RowTask]):
    name: str = "2_read_mock"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _RowTask) -> _RowTask:
        task.data = task.data.assign(value=list(range(len(task.data))))
        return task


def _apply_failure_modes(task: _RowTask, is_resume_run: bool) -> _RowTask | None:
    """Shared 3-mode logic for the four passthrough stages."""
    d = _decision(task._udid)
    if d == "always_fail":
        return None
    if not is_resume_run and _is_transient(task._udid):
        return None
    if d == "filter":
        n = len(task.data)
        task.data = task.data.iloc[: max(1, n // 2)].reset_index(drop=True)
    return task


@dataclass
class _PassThroughStage(ProcessingStage[_RowTask, _RowTask]):
    name: str = "passthrough"
    is_resume_run: bool = False

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _RowTask) -> _RowTask | None:
        return _apply_failure_modes(task, self.is_resume_run)


@dataclass
class _PassThroughBatchedStage(ProcessingStage[_RowTask, _RowTask]):
    """Overrides ``process_batch`` — must call the four lineage helpers itself.

    See [_resumability_runner.py:109-121] for the same contract on a real
    multi-parent stage.
    """

    name: str = "passthrough_batched"
    is_resume_run: bool = False
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _RowTask) -> _RowTask:
        msg = "_PassThroughBatchedStage only supports batched execution"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[_RowTask]) -> list[_RowTask]:
        tasks = self._filter_completed_tasks(tasks)
        results: list[_RowTask] = []
        for task in tasks:
            result = _apply_failure_modes(task, self.is_resume_run)
            children = assign_child_lineage([task._lineage_path], result)
            record_lineage([task._udid], [c._udid for c in children])
            if self._is_terminal_stage and children:
                mark_leaves_completed([c._udid for c in children])
            results.extend(children)
        return results


@dataclass
class _FanOutStage(ProcessingStage[_RowTask, _RowTask]):
    name: str = "5_fanout"
    factor: int = FANOUT_FACTOR

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _RowTask) -> list[_RowTask]:
        rows = len(task.data)
        n_out = max(1, rows // self.factor)
        chunk = max(1, -(-rows // n_out))  # ceil(rows / n_out)
        out: list[_RowTask] = []
        for i in range(n_out):
            sub = task.data.iloc[i * chunk : (i + 1) * chunk].reset_index(drop=True)
            if len(sub) == 0:
                continue
            out.append(
                _RowTask(
                    task_id=f"{task.task_id}_fan{i}",
                    dataset_name=task.dataset_name,
                    data=sub,
                )
            )
        return out


@dataclass
class _WriteParquetStage(ProcessingStage[_RowTask, _RowTask]):
    name: str = "8_write"
    out_dir: str = ""

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _RowTask) -> _RowTask:
        # Filename keyed on _udid so file existence ↔ LMDB completion 1:1.
        path = Path(self.out_dir) / f"{task._udid}.parquet"
        task.data.to_parquet(path, index=False)
        return task


def _build_pipeline(out_dir: Path, is_resume_run: bool, batched_size: int) -> Pipeline:
    return Pipeline(
        name="resumability_three_modes",
        stages=[
            _LsStage(),
            _ReadMockStage(),
            _PassThroughStage(name="3_passthrough", is_resume_run=is_resume_run),
            _PassThroughBatchedStage(
                name="4_passthrough_batched",
                is_resume_run=is_resume_run,
                batch_size=batched_size,
            ),
            _FanOutStage(),
            _PassThroughStage(name="6_passthrough", is_resume_run=is_resume_run),
            _PassThroughBatchedStage(
                name="7_passthrough_batched",
                is_resume_run=is_resume_run,
                batch_size=batched_size,
            ),
            _WriteParquetStage(out_dir=str(out_dir)),
        ],
    )


def _read_lmdb(path: Path) -> dict[str, LineageRecord]:
    """Open the checkpoint after the executor killed the writer actor."""
    store = LineageStore(str(path))
    try:
        return dict(store.iter_records())
    finally:
        store.close()


@pytest.mark.parametrize(
    "executor_cls",
    [
        pytest.param(XennaExecutor, id="xenna"),
        pytest.param(RayDataExecutor, id="ray_data"),
    ],
)
@pytest.mark.parametrize("batched_size", [1, 4], ids=["batch1", "batch4"])
@pytest.mark.usefixtures("shared_ray_cluster")
def test_resumability_three_modes(executor_cls: type[BaseExecutor], batched_size: int, tmp_path: Path) -> None:
    """Two runs against the same LMDB: transient on run 1, disabled on run 2.

    Verifies the three resumability properties documented at the top of the
    module across:

    * Both ``XennaExecutor`` and ``RayDataExecutor``.
    * ``batch_size`` 1 and 4 on the two ``process_batch``-override stages —
      the ``batch4`` axis exercises the multi-task-per-call path that the
      demo's finding #1 flagged as broken on RayData.
    """
    checkpoint = tmp_path / "lineage.mdb"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # --- Run 1: transient drops active ---
    _build_pipeline(out_dir, is_resume_run=False, batched_size=batched_size).run(
        executor_cls(), checkpoint_path=str(checkpoint)
    )
    run1 = _read_lmdb(checkpoint)
    run1_files = {p.stem for p in out_dir.glob("*.parquet")}

    # --- Run 2: same checkpoint, transient disabled ---
    _build_pipeline(out_dir, is_resume_run=True, batched_size=batched_size).run(
        executor_cls(), checkpoint_path=str(checkpoint)
    )
    run2 = _read_lmdb(checkpoint)
    run2_files = {p.stem for p in out_dir.glob("*.parquet")}

    # --- Transient rescued on resume ---
    run1_completed_leaves = {u for u, r in run1.items() if r.completed and r.task_type in ("leaf", "source_leaf")}
    run2_completed_leaves = {u for u, r in run2.items() if r.completed and r.task_type in ("leaf", "source_leaf")}
    assert run1_completed_leaves < run2_completed_leaves, (
        f"run 2 must complete strictly more leaves than run 1 "
        f"(run1={len(run1_completed_leaves)} run2={len(run2_completed_leaves)})"
    )

    # --- Always-fail branches never converge ---
    assert any(not r.completed for r in run2.values()), (
        "always-fail branches should leave some records incomplete after resume"
    )

    # --- Determinism + filename/udid coupling ---
    assert run1_files <= run2_files, "resume must not lose parquet outputs"
    assert run2_files == run2_completed_leaves, (
        f"parquet filenames (= _udid) must match the completed-leaf set; "
        f"files - completed = {run2_files - run2_completed_leaves}, "
        f"completed - files = {run2_completed_leaves - run2_files}"
    )

    # --- Filter recorded with halved rows ---
    # Post-fanout chunk size = FANOUT_FACTOR rows. A filter at stage 6 or 7
    # halves that. Any leaf with fewer than FANOUT_FACTOR rows must be a
    # filter-affected branch.
    row_counts = [len(pd.read_parquet(p)) for p in out_dir.glob("*.parquet")]
    assert any(c < FANOUT_FACTOR for c in row_counts), (
        f"expected at least one filter-affected parquet with fewer than "
        f"{FANOUT_FACTOR} rows; got row counts {sorted(set(row_counts))}"
    )
