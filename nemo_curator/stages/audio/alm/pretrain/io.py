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

"""I/O stages for the pretrain pipeline.

* :class:`ReadLongFormManifestStage` reads the input JSONL manifest and
  fans out one ``AudioTask`` per row.
* :class:`SnippetManifestWriterStage` appends every non-stub snippet to
  a per-replica manifest shard for the driver to merge later.
* :class:`PretrainMetricsAggregatorStage` writes one JSONL record per
  task seen, also into a per-replica shard, so the driver merge can
  build a single metrics summary regardless of how many replicas ran.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.alm.pretrain.utils import (
    _MANIFEST_SHARD_EXT,
    _METRICS_SHARD_EXT,
    _PRETRAIN_META_KEY,
    _is_origin_stub,
    _make_shard_path,
    _resolve_audio_path,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, _EmptyTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


# ----------------------------------------------------------------------
# Stage: read JSONL manifest, fan out into AudioTasks
# ----------------------------------------------------------------------


@dataclass
class ReadLongFormManifestStage(ProcessingStage[_EmptyTask, AudioTask]):
    """Read a JSONL manifest of long-form audios; emit one AudioTask per row.

    Each line in ``input_manifest`` is parsed as JSON and re-emitted as
    an ``AudioTask`` whose ``data`` is the parsed dict with its audio
    path re-anchored to ``audio_dir``.

    This is the entry-point ``_EmptyTask -> list[AudioTask]`` fan-out
    stage following the same pattern as
    ``CreateInitialManifestReadSpeechStage``.

    Args:
        input_manifest: Path to the JSONL file.
        audio_dir: Directory containing the source audio files; the row's
            ``audio_filepath`` value is replaced with
            ``audio_dir / basename(audio_filepath)``.
        audio_filepath_key: JSONL field that holds the path to the audio
            file (default ``"audio_filepath"``).
        dataset_name: Optional dataset tag stamped on emitted tasks.
    """

    input_manifest: str
    audio_dir: str
    audio_filepath_key: str = "audio_filepath"
    dataset_name: str = "long_form_audio"

    name: str = "ReadLongFormManifest"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "id", "segments"]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}

    def process(self, _: _EmptyTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        if not os.path.isfile(self.input_manifest):
            msg = f"Manifest not found: {self.input_manifest}"
            raise FileNotFoundError(msg)

        tasks: list[AudioTask] = []
        with open(self.input_manifest, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] line {lineno}: invalid JSON ({e}); skipping")
                    continue

                original_path = entry.get(self.audio_filepath_key)
                if not original_path:
                    logger.warning(f"[{self.name}] line {lineno}: missing {self.audio_filepath_key!r}; skipping")
                    continue
                entry[self.audio_filepath_key] = _resolve_audio_path(self.audio_dir, original_path)

                tasks.append(
                    AudioTask(
                        task_id=f"{entry.get('id', f'line_{lineno}')}",
                        dataset_name=self.dataset_name,
                        data=entry,
                        filepath_key=self.audio_filepath_key,
                    )
                )

        self._log_metrics(
            {
                "manifest_load_time": time.perf_counter() - t0,
                "manifest_rows": float(len(tasks)),
            }
        )
        logger.info(f"[{self.name}] loaded {len(tasks)} rows from {self.input_manifest}")
        return tasks


# ----------------------------------------------------------------------
# Stage: append snippet records to a JSONL manifest
# ----------------------------------------------------------------------


@dataclass
class SnippetManifestWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Append each (non-stub) snippet's ``data`` as a JSONL line.

    Single-replica writer; the file is truncated once on driver setup
    so reruns produce a clean output.  Origin-stub tasks (no
    ``snippet_id``) are passed through unchanged so the metrics
    aggregator can still see them.
    """

    output_path: str

    name: str = "SnippetManifestWriter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self._shard_path: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Each replica writes its own shard; finalize_audio_pretrain_outputs
        # merges them after pipeline.run().
        self._shard_path = _make_shard_path(self.output_path, _MANIFEST_SHARD_EXT)
        logger.info(f"[{self.name}] writing manifest shard to {self._shard_path}")

    def process(self, task: AudioTask) -> AudioTask:
        if not _is_origin_stub(task) and self._shard_path is not None:
            with open(self._shard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        return task


# ----------------------------------------------------------------------
# Stage: aggregate metrics across all snippets/originals
# ----------------------------------------------------------------------


@dataclass
class PretrainMetricsAggregatorStage(ProcessingStage[AudioTask, AudioTask]):
    """Per-replica metrics aggregator.

    Each ``process()`` call appends one JSONL record to a per-replica
    shard.  ``finalize_audio_pretrain_outputs`` reads every shard after
    ``pipeline.run()`` returns and aggregates the records into the final
    summary JSON.

    The per-task append shape (vs. accumulating in memory and flushing in
    ``teardown()``) is required for correctness under Xenna: Xenna kills
    stage actors with ``ray.kill()`` and never invokes any teardown hook,
    so an in-memory-only aggregator silently produces an empty summary.

    Record schema (one line per task seen):

    * ``id`` -- original audio id
    * ``in_segments``, ``in_duration_sec``, ``dropped`` -- per-original
      input-side counters; written on every record (identical across
      records for the same original); the merger keeps the first.
    * ``is_stub`` -- True iff this is the extractor's zero-snippet stub.
    * ``out_segments``, ``out_duration_sec`` -- this snippet's
      contribution; zero for stubs.
    * ``filtered_texts`` -- example texts of snippets dropped by the
      repetition filter; written only on the first record we see for a
      given ``id`` per replica (so the shard stays small even when many
      fan-out tasks share the same source).

    The merger sums ``out_*`` across non-stub records per id and counts
    them as ``out_snippets``.
    """

    output_path: str

    name: str = "PretrainMetricsAggregator"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self._shard_path: str | None = None
        self._seen_ids: set[str] = set()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._shard_path = _make_shard_path(self.output_path, _METRICS_SHARD_EXT)
        logger.info(f"[{self.name}] writing metrics shard to {self._shard_path}")

    def process(self, task: AudioTask) -> AudioTask:
        if self._shard_path is None:
            return task
        original_id = str(task.data.get("id") or "")
        if not original_id:
            return task
        meta = task._metadata.get(_PRETRAIN_META_KEY, {})
        is_stub = _is_origin_stub(task)
        record: dict[str, Any] = {
            "id": original_id,
            "in_segments": int(meta.get("original_seg_count", 0)),
            "in_duration_sec": float(meta.get("original_seg_duration", 0.0)),
            "dropped": {
                "empty": int(meta.get("dropped_empty", 0)),
                "overlap": int(meta.get("dropped_overlap", 0)),
                "too_long": int(meta.get("dropped_too_long", 0)),
                "too_short": int(meta.get("dropped_too_short", 0)),
                "no_text": int(meta.get("dropped_no_text", 0)),
                "repetition": int(meta.get("dropped_repetition", 0)),
            },
            "is_stub": is_stub,
            "out_segments": 0 if is_stub else len(task.data.get("segments") or []),
            "out_duration_sec": 0.0 if is_stub else float(task.data.get("duration", 0.0)),
        }
        if original_id not in self._seen_ids:
            self._seen_ids.add(original_id)
            record["filtered_texts"] = list(meta.get("filtered_repetition_texts") or [])
        with open(self._shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return task
