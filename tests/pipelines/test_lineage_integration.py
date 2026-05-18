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
import os
import signal
import subprocess
import sys
import time
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
    LineageStore,
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
class _Writer(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Stand-in for a real sink stage: emits one child per input. Lineage-wise
    indistinguishable from a passthrough but named separately so the 4-stage
    end-to-end test reads like a real ``passthrough → fanout → fanin → writer``
    pipeline."""

    name: str = "writer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        return _SimpleTask(task_id=f"{task.task_id}_w", dataset_name=task.dataset_name, data=task.data)


@dataclass
class _FailAfterN(ProcessingStage[_SimpleTask, _SimpleTask]):
    """Test-only stage that emits children for the first ``fail_after`` inputs and
    raises on the next one. Lets us drive a pipeline to a known partial-DAG state
    so we can assert "no spurious completions" after a mid-run abort."""

    fail_after: int = 1
    name: str = "fail_after_n"
    _seen: int = 0

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        if self._seen >= self.fail_after:
            msg = f"_FailAfterN exploding after {self.fail_after} inputs"
            raise RuntimeError(msg)
        self._seen += 1
        return _SimpleTask(task_id=f"{task.task_id}_x", dataset_name=task.dataset_name, data=task.data)


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


def test_full_run_marks_entire_dag_completed(actor: tuple[object, Path]) -> None:
    """End-to-end ``passthrough → fanout → fanin → writer``: after a successful run,
    every DAG node must be marked completed.

    ``_Writer`` (terminal) calls :func:`mark_leaves_completed` from inside its
    default ``process_batch``, which rolls completion up to the full DAG before
    ``_drive`` returns."""
    actor_handle, _ = actor
    pipeline = Pipeline(
        name="four_stage",
        stages=[_Passthrough(name="pt0"), _FanOut(times=3), _FanIn(), _Writer()],
    )
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])
    _drive(pipeline, [root])

    records = dict(ray.get(actor_handle.iter_records.remote()))
    assert all(rec.completed for rec in records.values())


def test_kill_midrun_leaves_partial_dag_with_no_completions(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    """Drive ``passthrough → fanout → fail_after_n → writer`` so the third stage
    explodes mid-batch, before the terminal ``_Writer`` stage runs at all.
    Verify that:

    1. The actor persisted edges for the stages that did run (passthrough + the
       partial fanout outputs the failing stage consumed before raising).
    2. No node is marked ``completed`` — the terminal stage never ran, so
       incremental marking never fired, and no leaves exist to seed the rollup.
       Companion test :func:`test_terminal_stage_partial_failure_marks_processed_leaves`
       covers the case where the terminal stage itself fails mid-batch."""
    _kill_actor_if_present()
    path = tmp_path / "lineage_midkill.mdb"
    actor_handle = LineageWriterActor.options(
        name=LINEAGE_ACTOR_NAME,
        get_if_exists=True,
    ).remote(path=str(path))

    try:
        pipeline = Pipeline(
            name="four_stage_flaky",
            stages=[
                _Passthrough(name="pt0"),
                _FanOut(times=5),
                _FailAfterN(fail_after=2),
                _Writer(),
            ],
        )
        pipeline.build()
        root = _SimpleTask(task_id="r", dataset_name="d", data=[1])

        with pytest.raises(RuntimeError, match="exploding"):
            _drive(pipeline, [root])

        # Close the actor cleanly so we can re-open the LMDB file from this process.
        ray.get(actor_handle.close.remote())
    finally:
        ray.kill(actor_handle)

    # Re-open the LMDB store directly to inspect the partial DAG.
    store = LineageStore(str(path))
    try:
        records = dict(store.iter_records())

        # We made progress: pt0 + at least 2 fanout outputs + 2 fail_after outputs were recorded.
        assert len(records) > 0
        # NOT every node should be present (writer never ran for some fanout outputs).
        # And critically, nothing must be marked completed — the terminal stage
        # (_Writer) never ran, so mark_leaves_completed was never invoked.
        assert all(not rec.completed for rec in records.values()), (
            f"unexpected completions in partial DAG: {[u for u, r in records.items() if r.completed]}"
        )
    finally:
        store.close()


def test_incremental_marking_inside_terminal_process_batch(
    actor: tuple[object, Path],
) -> None:
    """Drive ``passthrough → fanout → writer`` one stage at a time and peek at the
    LMDB between stages. Until the terminal ``_Writer`` stage runs, nothing is
    completed; once it does, every recorded node rolls up via BFS.

    This proves the marking happens inside the terminal stage's
    ``process_batch`` (via :func:`mark_leaves_completed`), not at end-of-pipeline.
    """
    actor_handle, _ = actor
    pipeline = Pipeline(
        name="incremental_marking",
        stages=[_Passthrough(name="pt0"), _FanOut(times=2), _Writer()],
    )
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])

    # Drive stage-by-stage, peeking after each.
    current: list[Task] = [root]
    completion_history: list[bool] = []
    for stage in pipeline.stages:
        current = BaseStageAdapter(stage).process_batch(current)
        records = dict(ray.get(actor_handle.iter_records.remote()))
        completion_history.append(any(rec.completed for rec in records.values()))

    # Stages 0 (passthrough) and 1 (fanout) record edges but never complete anything;
    # only stage 2 (terminal writer) triggers incremental marking.
    assert completion_history == [False, False, True]

    final_records = dict(ray.get(actor_handle.iter_records.remote()))
    assert all(rec.completed for rec in final_records.values())


def test_terminal_stage_partial_failure_marks_processed_leaves(
    actor: tuple[object, Path],
) -> None:
    """Place ``_FailAfterN`` as the terminal stage so it processes ``fail_after``
    inputs (marking each emitted leaf and rolling up its fully-completed ancestor
    chain) before exploding on the next input. Verify the partial completions
    landed in LMDB even though the pipeline aborted — the resumability story."""
    actor_handle, path = actor
    pipeline = Pipeline(
        name="terminal_fails_midbatch",
        stages=[_Passthrough(name="pt0"), _FanOut(times=5), _FailAfterN(fail_after=2)],
    )
    pipeline.build()
    root = _SimpleTask(task_id="r", dataset_name="d", data=[1])

    with pytest.raises(RuntimeError, match="exploding"):
        _drive(pipeline, [root])

    # Lineage paths for this topology (root has empty lineage_path):
    #   pt0 output:         "0"
    #   fanout(5) outputs:  "0_0" ... "0_4"
    #   fail_after outputs: "0_0_0", "0_1_0" (only first two processed successfully)
    pt0_udid = _path_to_udid("0")
    fanout_udids = [_path_to_udid(f"0_{i}") for i in range(5)]
    leaf_udids = [_path_to_udid("0_0_0"), _path_to_udid("0_1_0")]

    # Close the actor cleanly so we can re-open the LMDB file from this process.
    ray.get(actor_handle.close.remote())
    store = LineageStore(str(path))
    try:
        records = dict(store.iter_records())

        # Leaves emitted before the crash are completed.
        for udid in leaf_udids:
            assert records[udid].completed, f"leaf {udid} should be completed"

        # The two fanout outputs whose only child reached the terminal stage roll
        # up; the remaining three fanout outputs (no children produced) do not.
        completed_fanouts = [u for u in fanout_udids if records[u].completed]
        assert len(completed_fanouts) == 2

        # pt0 has 5 fanout children but only 2 completed — the BFS gate blocks
        # rollup at pt0, matching the partial-fan-in contract.
        assert not records[pt0_udid].completed
    finally:
        store.close()


def _launch_runner(checkpoint: Path, n_tasks: int, fanin_size: int, writer_sleep_s: float) -> subprocess.Popen[str]:
    runner = Path(__file__).parent / "_resumability_runner.py"
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": f"{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    return subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(runner),
            "--checkpoint-path",
            str(checkpoint),
            "--n-tasks",
            str(n_tasks),
            "--fanin-size",
            str(fanin_size),
            "--writer-sleep-s",
            str(writer_sleep_s),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


def _wait_for_completions_then_sigint(proc: subprocess.Popen[str], threshold: int, max_wait_s: float) -> None:
    """Read stdout until ``threshold`` ``completed`` lines have been emitted,
    then send SIGINT and wait for the runner to exit. Raises ``pytest.fail``
    on premature exit, no progress, or hang."""
    completed_seen = 0
    deadline = time.monotonic() + max_wait_s
    while completed_seen < threshold:
        line = proc.stdout.readline()
        if line == "":
            _, stderr_tail = proc.communicate(timeout=10)
            pytest.fail(
                f"runner exited before reaching threshold (saw {completed_seen} completions). "
                f"stderr:\n{stderr_tail}"
            )
        if line.strip() == "completed":
            completed_seen += 1
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            pytest.fail(f"runner only reached {completed_seen}/{threshold} completions in {max_wait_s}s")

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("runner did not exit within 60s of SIGINT")


def _drain_proc(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    with contextlib.suppress(Exception):
        proc.stdout.close()
    with contextlib.suppress(Exception):
        proc.stderr.close()


def _read_records(checkpoint: Path) -> dict[str, object]:
    _kill_actor_if_present()
    store = LineageStore(str(checkpoint))
    try:
        return dict(store.iter_records())
    finally:
        store.close()


def test_resumable_after_sigint(tmp_path: Path, shared_ray_cluster: str) -> None:  # noqa: ARG001
    """Drive a 4-stage 2000-task pipeline (fanout -> passthrough -> chunked-fanin
    -> slow_writer) in a subprocess, SIGINT it mid-run, and verify partial
    completion. Relaunch with the same checkpoint path and verify full
    completion. The shared Ray cluster (autouse session fixture) is what the
    runner subprocess connects to via ``RAY_ADDRESS``."""
    checkpoint = tmp_path / "lineage_resume.mdb"
    n_tasks = 2000
    fanin_size = 20
    writer_sleep_s = 0.05
    expected_total = 2 * n_tasks + 2 * (n_tasks // fanin_size)  # 4200
    threshold = 5

    # --- Run 1: launch, interrupt after some leaves complete ---
    _kill_actor_if_present()
    proc = _launch_runner(checkpoint, n_tasks, fanin_size, writer_sleep_s)
    try:
        _wait_for_completions_then_sigint(proc, threshold=threshold, max_wait_s=120)
    finally:
        _drain_proc(proc)

    records = _read_records(checkpoint)
    completed = [u for u, r in records.items() if r.completed]
    assert len(completed) >= threshold, (
        f"expected at least {threshold} completions after SIGINT, found {len(completed)}"
    )
    assert len(completed) < expected_total, (
        f"all {expected_total} nodes should not be completed after mid-run SIGINT (saw {len(completed)})"
    )

    # --- Run 2: relaunch with same checkpoint, run to natural completion ---
    _kill_actor_if_present()
    proc2 = _launch_runner(checkpoint, n_tasks, fanin_size, writer_sleep_s)
    try:
        _, stderr_tail = proc2.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc2.kill()
        proc2.wait()
        pytest.fail("runner did not finish within 300s on resume")
    assert proc2.returncode == 0, f"runner exited with {proc2.returncode}; stderr:\n{stderr_tail}"

    records = _read_records(checkpoint)
    assert len(records) == expected_total, (
        f"expected {expected_total} recorded nodes after full run, found {len(records)}"
    )
    unfinished = [u for u, r in records.items() if not r.completed]
    assert not unfinished, f"unfinished after resume: {unfinished[:5]} ({len(unfinished)} total)"
