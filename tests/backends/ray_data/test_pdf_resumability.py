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

"""End-to-end resumability test for the Nemotron-Parse PDF tutorial.

Downloads 2 real PDFs via the tutorial's ``download_data.py``, runs them
through a CPU-only pipeline (Partitioning -> Preprocess -> FanOut -> PassThrough),
verifies both source-task keys and per-leaf task_keys land in the LMDB
checkpoint DB, then reruns and verifies the partitions are skipped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytest.importorskip("pypdfium2")  # PDFPreprocessStage renders pages with pypdfium2

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch, _EmptyTask
from nemo_curator.utils.checkpoint import CheckpointManager

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOWNLOAD_DATA_PATH = _REPO_ROOT / "tutorials" / "interleaved" / "nemotron_parse_pdf" / "download_data.py"


def _load_download_data() -> ModuleType:
    if "tutorial_download_data" in sys.modules:
        return sys.modules["tutorial_download_data"]
    spec = importlib.util.spec_from_file_location("tutorial_download_data", _DOWNLOAD_DATA_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load download_data module from {_DOWNLOAD_DATA_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tutorial_download_data"] = module  # required for @dataclass annotation lookup
    spec.loader.exec_module(module)
    return module


class _FanOutInterleavedStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    name = "fanout_interleaved"
    resources = Resources(cpus=0.5)

    def __init__(self, factor: int = 3):
        self._factor = factor

    def process(self, task: InterleavedBatch) -> list[InterleavedBatch]:
        return [
            InterleavedBatch(
                task_id=f"{task.task_id}_fan{i}",
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=dict(task._metadata),
            )
            for i in range(self._factor)
        ]


class _WriteInterleavedStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Simulated writer at the tail of the pipeline.

    Mirrors the ``WriteParquetStage`` pattern in
    ``test_resumability_mock_pipeline.py``: overriding ``setup`` forces
    ``is_actor_stage()`` to True, which lets the adapter treat this as the
    last user stage and wire up ``_drop_completed_inputs``. The write itself
    is just a marker file per task so partitions are visible on disk.
    """

    name = "write_interleaved"
    resources = Resources(cpus=0.5)

    def __init__(self, out_dir: str):
        self._out_dir = out_dir

    def setup(self, _worker_metadata: object | None = None) -> None:
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        (Path(self._out_dir) / f"{task.task_id}.done").touch()
        return task


class _MarkFirstFanoutLeafStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Mark the i=0 fan-out leaf of each partition complete in the checkpoint DB.

    Inserted between the fan-out and the last user stage to simulate partial
    leaf completion: the next stage's adapter (``_is_last_user_stage=True``) will
    consult the DB via ``_drop_completed_inputs`` and skip the marked leaves,
    so only the unmarked ones reach the recorder.
    """

    name = "mark_first_fanout"
    resources = Resources(cpus=0.5)

    def __init__(self, checkpoint_path: str):
        self._checkpoint_path = checkpoint_path
        self._actor = None

    def setup(self, _worker_metadata: object | None = None) -> None:
        from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

        self._actor = get_or_create_checkpoint_actor(self._checkpoint_path)

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        if task.task_id.endswith("_fan0"):
            from nemo_curator.utils.checkpoint import _checkpoint_get

            key = task._metadata["resumability_key"]
            cur = task._metadata["resumability_task_key"]
            # This stage is 1->1, so the framework's _propagate_resumability_metadata
            # will append "::0" to task_key before the next stage sees it.  Mark the
            # post-propagation leaf key so _drop_completed_inputs at the last user
            # stage finds and drops the task.
            future_task_key = f"{cur}::0"
            _checkpoint_get(self._actor.mark_completed.remote(future_task_key, key))
        return task


def test_pdf_tutorial_resumability_with_fanout(tmp_path: Path, shared_ray_client: None) -> None:  # noqa: ARG001, PLR0915
    """Run the PDF tutorial pipeline twice; second run must skip both partitions."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "out"

    # ---- Download 2 real PDFs via the tutorial helper ----------------------
    download_data = _load_download_data()
    cfg = download_data.DownloadConfig(output_dir=data_dir, num_pdfs=2, force=False, workers=2)
    successes = download_data.download_all(cfg)
    if len(successes) < 2:
        pytest.skip("network unavailable — could not download 2 PDFs")
    manifest_path = data_dir / "manifest.jsonl"
    download_data.write_manifest(successes, manifest_path)
    pdfs_dir = data_dir / "pdfs"

    # ---- Compute the deterministic source-task keys -----------------------
    probe = PDFPartitioningStage(manifest_path=str(manifest_path), pdfs_per_task=1)
    probe_tasks = probe.process(_EmptyTask(task_id="probe", dataset_name="pdf_dataset", data=None))
    expected_source_keys = {t._metadata["resumability_key"] for t in probe_tasks}
    assert len(expected_source_keys) == 2

    # ---- Build a fresh pipeline (executors mutate stage attrs in place) ---
    fanout_factor = 3

    def build_pipeline() -> Pipeline:
        p = Pipeline(name="pdf_resumability_test")
        p.add_stage(PDFPartitioningStage(manifest_path=str(manifest_path), pdfs_per_task=1))
        p.add_stage(PDFPreprocessStage(pdf_dir=str(pdfs_dir), max_pages=2))
        p.add_stage(_FanOutInterleavedStage(factor=fanout_factor))
        p.add_stage(_WriteInterleavedStage(out_dir=str(out_dir)))
        return p

    # ---- First run: process both PDFs end-to-end --------------------------
    out1 = build_pipeline().run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))
    assert out1 is not None
    assert len(out1) == 2 * fanout_factor

    # The auto-injected _CheckpointRecorderStage runs after our last user stage and
    # is itself a 1->1 stage, so its adapter propagates a final "::0" suffix onto each
    # output's resumability_task_key.  The keys actually recorded in the DB are the
    # ones the recorder *received* (i.e. the last user stage's outputs), so strip that
    # trailing "::0" to recover them.
    def _strip_recorder_suffix(task_key: str) -> str:
        assert task_key.endswith("::0"), f"unexpected task_key shape: {task_key!r}"
        return task_key[: -len("::0")]

    leaf_pairs: list[tuple[str, str]] = [
        (t._metadata["resumability_key"], _strip_recorder_suffix(t._metadata["resumability_task_key"])) for t in out1
    ]
    observed_source_keys = {key for key, _ in leaf_pairs}

    # ---- Assert keys are in the DB ----------------------------------------
    mgr = CheckpointManager(str(ckpt_dir))
    try:
        assert observed_source_keys == expected_source_keys
        for key in expected_source_keys:
            assert mgr.is_task_completed(key), f"source key {key!r} not fully completed in checkpoint DB"

        leaf_flags = mgr.are_leaves_completed(leaf_pairs)
        missing = [pair for pair, ok in zip(leaf_pairs, leaf_flags, strict=True) if not ok]
        assert not missing, f"leaf keys missing from checkpoint DB: {missing}"
        assert len(set(leaf_pairs)) == 2 * fanout_factor, "expected unique leaf task_keys per fan-out output"
    finally:
        mgr.close()

    # ---- Second run: every partition must be skipped ----------------------
    out2 = build_pipeline().run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))
    assert out2 is not None
    assert len(out2) == 0, f"expected resume to skip all partitions, got {len(out2)} output tasks"

    # ---- Defensive: DB still reports completion for every key -------------
    mgr2 = CheckpointManager(str(ckpt_dir))
    try:
        for key in expected_source_keys:
            assert mgr2.is_task_completed(key)
        assert all(mgr2.are_leaves_completed(leaf_pairs))
    finally:
        mgr2.close()


def test_pdf_tutorial_leaf_resumability_drops_completed_leaves(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    """Leaves marked complete mid-pipeline are skipped by the last user stage.

    Pipeline: Partitioning -> Preprocess -> FanOut(3) -> MarkFirstFanout -> PassThrough.

    The marker pre-records the i=0 fan-out leaf of each partition in the checkpoint
    DB.  Partitions are NOT fully complete at the source filter (no entries existed
    when the run started, so init_partition + reset run normally), so the source
    filter passes both source tasks through.  Each partition then fans out to 3
    leaves; the marker records the i=0 leaf; the last user stage's adapter drops
    those marked leaves via ``_drop_completed_inputs``; only (factor - 1) leaves
    per partition reach the recorder.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "out"

    # ---- Download 2 real PDFs via the tutorial helper ---------------------
    download_data = _load_download_data()
    cfg = download_data.DownloadConfig(output_dir=data_dir, num_pdfs=2, force=False, workers=2)
    successes = download_data.download_all(cfg)
    if len(successes) < 2:
        pytest.skip("network unavailable — could not download 2 PDFs")
    manifest_path = data_dir / "manifest.jsonl"
    download_data.write_manifest(successes, manifest_path)
    pdfs_dir = data_dir / "pdfs"

    fanout_factor = 3

    pipeline = Pipeline(name="pdf_leaf_resumability_test")
    pipeline.add_stage(PDFPartitioningStage(manifest_path=str(manifest_path), pdfs_per_task=1))
    pipeline.add_stage(PDFPreprocessStage(pdf_dir=str(pdfs_dir), max_pages=2))
    pipeline.add_stage(_FanOutInterleavedStage(factor=fanout_factor))
    pipeline.add_stage(_MarkFirstFanoutLeafStage(str(ckpt_dir)))
    pipeline.add_stage(_WriteInterleavedStage(out_dir=str(out_dir)))

    out = pipeline.run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))

    # 2 partitions x (fanout - 1 dropped) = 4 surviving tasks reach the recorder
    assert out is not None
    expected_count = 2 * (fanout_factor - 1)
    assert len(out) == expected_count, f"expected {expected_count} surviving tasks, got {len(out)}"

    # No surviving task should correspond to the i=0 fan-out leaf
    surviving_task_ids = {t.task_id for t in out}
    assert not any(tid.endswith("_fan0") for tid in surviving_task_ids), (
        f"i=0 fan-out leaves should have been dropped at the last user stage, but found: {surviving_task_ids}"
    )

    # ---- Both partitions should be fully complete --------------------------
    # Per partition: 1 leaf recorded by the marker + (fanout-1) leaves recorded by
    # the recorder = fanout total, matching expected (set by fan-out's add_expected).
    source_keys = {t._metadata["resumability_key"] for t in out}
    assert len(source_keys) == 2

    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for key in source_keys:
            assert mgr.is_task_completed(key), f"partition {key!r} not fully complete after run"
    finally:
        mgr.close()
