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

"""End-to-end resumability tests for a realistic text pipeline.

Pipeline shape (mirrors the structure of test_pdf_resumability.py but for text and
with multiple fan-out points):

    FilePartitioningStage      # REAL source fan-out (one _EmptyTask -> N FileGroupTasks)
      -> JsonlReaderStage      # REAL 1->1 (FileGroupTask -> DocumentBatch)
      -> ScoreFilter           # REAL 1->1 (length filter, configured to keep all docs)
      -> Modify                # REAL 1->1 (unicode normalization)
      -> _FanOutDocumentBatchStage  # TEST helper fan-out (per-shard split)
      -> _SecondFanOutStage         # TEST helper fan-out (per-chunk re-split)
      -> _ActorPassThroughStage     # forces actor-style so _checkpoint_actor wires up

No network: synthetic JSONL files are written to ``tmp_path``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic.string import WordCountFilter
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter
from nemo_curator.tasks import DocumentBatch, _EmptyTask
from nemo_curator.utils.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl_files(data_dir: Path, num_files: int, docs_per_file: int = 4) -> None:
    """Write deterministic JSONL files (one doc per line) under ``data_dir``.

    Each text has well over the ``min_words`` threshold so every doc passes
    ``WordCountFilter`` — keeps the test focused on the fan-out paths instead of
    full-drop bookkeeping.
    """
    for f_idx in range(num_files):
        path = data_dir / f"part_{f_idx:03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for d_idx in range(docs_per_file):
                doc = {
                    "id": f"f{f_idx}_d{d_idx}",
                    "text": f"document {d_idx} in file {f_idx} has several words here for the filter",
                }
                fh.write(json.dumps(doc) + "\n")


class _FanOutDocumentBatchStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "fanout_document_batch_a"
    resources = Resources(cpus=0.5)

    def __init__(self, factor: int = 3):
        self._factor = factor

    def process(self, task: DocumentBatch) -> list[DocumentBatch]:
        df = task.to_pandas()
        return [
            DocumentBatch(
                task_id=f"{task.task_id}_fanA{i}",
                dataset_name=task.dataset_name,
                data=df.copy(),
                _metadata=dict(task._metadata),
            )
            for i in range(self._factor)
        ]


class _SecondFanOutStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "fanout_document_batch_b"
    resources = Resources(cpus=0.5)

    def __init__(self, factor: int = 2):
        self._factor = factor

    def process(self, task: DocumentBatch) -> list[DocumentBatch]:
        df = task.to_pandas()
        return [
            DocumentBatch(
                task_id=f"{task.task_id}_fanB{i}",
                dataset_name=task.dataset_name,
                data=df.copy(),
                _metadata=dict(task._metadata),
            )
            for i in range(self._factor)
        ]


class _ActorPassThroughStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Pass-through with an overridden ``setup`` so ``is_actor_stage()`` is True.

    Ray-Data task-style stages skip ``BaseStageAdapter.setup``, which leaves
    ``_checkpoint_actor`` unset and disables ``_drop_completed_inputs`` on the
    last user stage. Forcing actor-style execution wires the actor up so the
    leaf-skip path actually runs in the second test.
    """

    name = "actor_passthrough_document"
    resources = Resources(cpus=0.5)

    def setup(self, _worker_metadata: object | None = None) -> None:
        return

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return task


class _MarkFanoutLeafStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Mark the i_B=0 leaves of the second fan-out complete in the checkpoint DB.

    Inserted between the second fan-out and the last user stage to simulate
    partial leaf completion. The next stage's adapter (``_is_last_user_stage =
    True``) will consult the DB via ``_drop_completed_inputs`` and skip the
    marked leaves, so only the unmarked ones reach the recorder.
    """

    name = "mark_fanout_leaf"
    resources = Resources(cpus=0.5)

    def __init__(self, checkpoint_path: str, target_suffix: str = "_fanB0"):
        self._checkpoint_path = checkpoint_path
        self._target_suffix = target_suffix
        self._actor = None

    def setup(self, _worker_metadata: object | None = None) -> None:
        from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

        self._actor = get_or_create_checkpoint_actor(self._checkpoint_path)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if task.task_id.endswith(self._target_suffix):
            from nemo_curator.utils.checkpoint import _checkpoint_get

            key = task._metadata["resumability_key"]
            cur = task._metadata["resumability_task_key"]
            # This stage is 1->1, so _propagate_resumability_metadata will append
            # "::0" to task_key before the next stage sees it. Mark the post-
            # propagation leaf key so _drop_completed_inputs at the last user
            # stage finds and drops the task.
            future_task_key = f"{cur}::0"
            _checkpoint_get(self._actor.mark_completed.remote(future_task_key, key))
        return task


def _strip_recorder_suffix(task_key: str) -> str:
    """The auto-injected _CheckpointRecorderStage is itself 1->1, so its adapter
    propagates a trailing ``::0`` onto each output's ``resumability_task_key``.
    Strip it to recover the key the recorder actually wrote to the DB.
    """
    assert task_key.endswith("::0"), f"unexpected task_key shape: {task_key!r}"
    return task_key[: -len("::0")]


def test_text_pipeline_resumability_with_multiple_fanouts(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    """Run a multi-fan-out text pipeline twice; second run must skip all partitions."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"

    num_files = 3
    _write_jsonl_files(data_dir, num_files=num_files, docs_per_file=4)

    # ---- Compute the deterministic source-task keys -----------------------
    probe = FilePartitioningStage(file_paths=str(data_dir), files_per_partition=1)
    probe_tasks = probe.process(_EmptyTask(task_id="probe", dataset_name="text_dataset", data=None))
    expected_source_keys = {t._metadata["resumability_key"] for t in probe_tasks}
    assert len(expected_source_keys) == num_files

    factor_a = 3
    factor_b = 2

    def build_pipeline() -> Pipeline:
        p = Pipeline(name="text_resumability_test")
        p.add_stage(FilePartitioningStage(file_paths=str(data_dir), files_per_partition=1))
        p.add_stage(JsonlReaderStage())
        p.add_stage(ScoreFilter(filter_obj=WordCountFilter(min_words=2, max_words=100000), text_field="text"))
        p.add_stage(Modify(modifier_fn=UnicodeReformatter(), input_fields="text"))
        p.add_stage(_FanOutDocumentBatchStage(factor=factor_a))
        p.add_stage(_SecondFanOutStage(factor=factor_b))
        p.add_stage(_ActorPassThroughStage())
        return p

    # ---- First run: process every partition end-to-end --------------------
    out1 = build_pipeline().run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))
    assert out1 is not None
    expected_leaves = num_files * factor_a * factor_b
    assert len(out1) == expected_leaves

    leaf_pairs: list[tuple[str, str]] = [
        (t._metadata["resumability_key"], _strip_recorder_suffix(t._metadata["resumability_task_key"])) for t in out1
    ]
    observed_source_keys = {key for key, _ in leaf_pairs}

    mgr = CheckpointManager(str(ckpt_dir))
    try:
        assert observed_source_keys == expected_source_keys
        for key in expected_source_keys:
            assert mgr.is_task_completed(key), f"source key {key!r} not fully completed in checkpoint DB"

        leaf_flags = mgr.are_leaves_completed(leaf_pairs)
        missing = [pair for pair, ok in zip(leaf_pairs, leaf_flags, strict=True) if not ok]
        assert not missing, f"leaf keys missing from checkpoint DB: {missing}"
        assert len(set(leaf_pairs)) == expected_leaves, "expected unique leaf task_keys per fan-out output"
    finally:
        mgr.close()

    # ---- Second run: every partition must be skipped ----------------------
    out2 = build_pipeline().run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))
    assert out2 is not None
    assert len(out2) == 0, f"expected resume to skip all partitions, got {len(out2)} output tasks"

    mgr2 = CheckpointManager(str(ckpt_dir))
    try:
        for key in expected_source_keys:
            assert mgr2.is_task_completed(key)
        assert all(mgr2.are_leaves_completed(leaf_pairs))
    finally:
        mgr2.close()


def test_text_pipeline_leaf_resumability_drops_completed_leaves(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    """Leaves marked complete mid-pipeline are skipped by the last user stage.

    Pipeline:
        FilePartitioningStage -> JsonlReaderStage -> ScoreFilter -> Modify
          -> _FanOutDocumentBatchStage(factor_a)
          -> _SecondFanOutStage(factor_b)
          -> _MarkFanoutLeafStage(target_suffix="_fanB0")
          -> _ActorPassThroughStage

    The marker pre-records the i_B=0 fan-out leaf of every (partition, fanA_i)
    pair. The source filter still passes both source tasks through (the partition
    is not finalized at the start of the run). Both fan-outs fire and re-register
    expected; the marker records the leaves; the last user stage's adapter drops
    them via ``_drop_completed_inputs``; only ``factor_a * (factor_b - 1)`` leaves
    per partition reach the recorder.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"

    num_files = 2
    _write_jsonl_files(data_dir, num_files=num_files, docs_per_file=3)

    factor_a = 3
    factor_b = 2

    pipeline = Pipeline(name="text_leaf_resumability_test")
    pipeline.add_stage(FilePartitioningStage(file_paths=str(data_dir), files_per_partition=1))
    pipeline.add_stage(JsonlReaderStage())
    pipeline.add_stage(ScoreFilter(filter_obj=WordCountFilter(min_words=2, max_words=100000), text_field="text"))
    pipeline.add_stage(Modify(modifier_fn=UnicodeReformatter(), input_fields="text"))
    pipeline.add_stage(_FanOutDocumentBatchStage(factor=factor_a))
    pipeline.add_stage(_SecondFanOutStage(factor=factor_b))
    pipeline.add_stage(_MarkFanoutLeafStage(str(ckpt_dir), target_suffix="_fanB0"))
    pipeline.add_stage(_ActorPassThroughStage())

    out = pipeline.run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))

    expected_count = num_files * factor_a * (factor_b - 1)
    assert out is not None
    assert len(out) == expected_count, f"expected {expected_count} surviving tasks, got {len(out)}"

    surviving_task_ids = {t.task_id for t in out}
    assert not any(tid.endswith("_fanB0") for tid in surviving_task_ids), (
        f"_fanB0 leaves should have been dropped at the last user stage, but found: {surviving_task_ids}"
    )

    # Per partition, the marker records (factor_a) leaves and the recorder records
    # the remaining (factor_a * (factor_b - 1)) — total = factor_a * factor_b, the
    # expected count set by the two synchronous add_expected calls.
    source_keys = {t._metadata["resumability_key"] for t in out}
    assert len(source_keys) == num_files

    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for key in source_keys:
            assert mgr.is_task_completed(key), f"partition {key!r} not fully complete after run"
    finally:
        mgr.close()
