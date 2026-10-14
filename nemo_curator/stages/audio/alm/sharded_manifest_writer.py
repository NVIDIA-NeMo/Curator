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

"""Sharded Manifest Writer -- writes per-shard JSONL files mirroring input paths with .done markers."""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io.manifest_writer_utils import (
    AudioManifestWriterMetrics,
    TerminalAudioPerformanceWriterMixin,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask


@dataclass
class ShardedManifestWriterStage(TerminalAudioPerformanceWriterMixin, ProcessingStage[AudioTask, FileGroupTask]):
    """Write AudioTasks to per-shard JSONL files mirroring the input manifest path structure.

    Output structure mirrors the input manifest paths::

        output_dir/
          yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl
          yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl.done

    The shard key is extracted from ``task._metadata["_shard_key"]``
    which is set by ``NemoTarShardReaderStage`` as a relative path
    (e.g. ``yodas/0_from_captions/en/sharded_manifests/manifest_42``).

    Args:
        output_dir: Root directory for output manifests.
    """

    name: str = "sharded_manifest_writer"
    output_dir: str = ""
    # Batch so process_batch() does one open+write+close per (batch, shard)
    # instead of one per row, lifting the Lustre per-row fsync ceiling.
    batch_size: int = 256
    ephemeral_keys: tuple[str, ...] = ("waveform",)
    write_perf_stats: bool = False
    duration_key: str = "duration"
    perf_summary_path: str | None = None
    perf_run_id: str = ""
    perf_executor: str = ""
    perf_pipeline_metadata: dict[str, Any] | None = None
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _writer_metrics: AudioManifestWriterMetrics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for ShardedManifestWriterStage"
            raise ValueError(msg)
        self._reset_writer_metrics()

    def _default_perf_summary_path(self) -> str:
        return os.path.join(self.output_dir, "perf_summary.json")

    def prepare_performance_summary(self) -> None:
        """Reset driver-owned metrics and remove the previous run summary."""
        self._reset_writer_metrics()
        self._remove_existing_perf_summary()

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ShardedManifestWriterStage: output_dir={self.output_dir}")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Recover ``_shard_counts`` from disk on every (re)start.

        Ray and Xenna may kill and replace this actor at any time
        (preemption, OOM, autoscaler scale-down, exception retry). A
        fresh actor would otherwise start with an empty counter and
        ``.done`` markers would never fire for shards that were partially
        processed before the crash.

        Strategy: walk ``output_dir`` once and seed
        ``_shard_counts[shard_key]`` from the line count of every
        ``*.jsonl`` that does not yet have a sibling ``*.jsonl.done``.
        Shards with a ``.done`` marker are skipped — they are already
        finalized and the reader will skip them on resume.
        """
        self._reset_writer_metrics()
        if not os.path.isdir(self.output_dir):
            return

        recovered = 0
        for root, _dirs, files in os.walk(self.output_dir):
            for fname in files:
                if not fname.endswith(".jsonl"):
                    continue
                jsonl_path = os.path.join(root, fname)
                if os.path.exists(jsonl_path + ".done"):
                    continue
                rel = os.path.relpath(jsonl_path, self.output_dir)
                shard_key = rel[: -len(".jsonl")]
                try:
                    with open(jsonl_path, "rb") as f:
                        self._shard_counts[shard_key] = sum(1 for _ in f)
                except OSError as exc:
                    logger.warning(f"ShardedManifestWriter: failed to recover line count for {jsonl_path}: {exc}")
                    continue
                recovered += 1

        if recovered:
            logger.info(
                f"ShardedManifestWriter: recovered partial counts for {recovered} shard(s) from {self.output_dir}"
            )

    @staticmethod
    def _json_default(value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, np.generic):
            return value.item()
        msg = f"Manifest value is not JSON serializable: {type(value).__name__}"
        raise TypeError(msg)

    def _serialize(self, task: AudioTask) -> str:
        row = {key: value for key, value in task.data.items() if key not in self.ephemeral_keys}
        return json.dumps(row, ensure_ascii=False, default=self._json_default)

    def process(self, task: AudioTask) -> FileGroupTask:
        shard_key = task._metadata.get("_shard_key", "unknown/shard_0")

        out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        write_t0 = time.perf_counter()
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(self._serialize(task) + "\n")
        manifest_write_time_s = time.perf_counter() - write_t0

        self._shard_counts[shard_key] += 1

        shard_total = task._metadata.get("_shard_total", 0)
        done_write_time_s = 0.0
        if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
            done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
            done_t0 = time.perf_counter()
            with open(done_path, "w") as f:
                f.write(f"{self._shard_counts[shard_key]}\n")
            done_write_time_s = time.perf_counter() - done_t0
            logger.info(f"Shard {shard_key} complete: {self._shard_counts[shard_key]} utterances, wrote {done_path}")

        if self.write_perf_stats:
            self._log_metrics(
                self._writer_metrics.record_output_invocation(
                    [task],
                    manifest_write_time_s=manifest_write_time_s,
                    extra_metrics={"done_marker_write_time_s": done_write_time_s},
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[out_path],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def process_batch(self, tasks: list[AudioTask]) -> list[FileGroupTask]:
        """One open+write+close per (batch, shard) instead of one per row."""
        # Ray Data passes tasks as an ndarray, so use len() not `if not tasks`.
        if len(tasks) == 0:
            return []

        by_shard: dict[str, list[AudioTask]] = defaultdict(list)
        for task in tasks:
            shard_key = task._metadata.get("_shard_key", "unknown/shard_0")
            by_shard[shard_key].append(task)

        results: list[FileGroupTask] = []
        manifest_write_time_s = 0.0
        done_write_time_s = 0.0
        for shard_key, shard_tasks in by_shard.items():
            out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            write_t0 = time.perf_counter()
            with open(out_path, "a", encoding="utf-8") as f:
                for task in shard_tasks:
                    f.write(self._serialize(task) + "\n")
            manifest_write_time_s += time.perf_counter() - write_t0

            # Update count only after data is on disk, so .done reflects it.
            self._shard_counts[shard_key] += len(shard_tasks)

            shard_total = shard_tasks[0]._metadata.get("_shard_total", 0)
            if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
                done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
                done_t0 = time.perf_counter()
                with open(done_path, "w") as f:
                    f.write(f"{self._shard_counts[shard_key]}\n")
                done_write_time_s += time.perf_counter() - done_t0
                logger.info(
                    f"Shard {shard_key} complete: {self._shard_counts[shard_key]} utterances, wrote {done_path}"
                )

            for task in shard_tasks:
                results.append(
                    FileGroupTask(
                        dataset_name=task.dataset_name,
                        data=[out_path],
                        _metadata=task._metadata,
                        _stage_perf=task._stage_perf,
                    )
                )

        if self.write_perf_stats:
            self._log_metrics(
                self._writer_metrics.record_output_invocation(
                    tasks,
                    manifest_write_time_s=manifest_write_time_s,
                    extra_metrics={"done_marker_write_time_s": done_write_time_s},
                )
            )
        return results

    def teardown(self) -> None:
        super().teardown()
        total = sum(self._shard_counts.values())
        done = sum(1 for k in self._shard_counts if os.path.exists(os.path.join(self.output_dir, f"{k}.jsonl.done")))
        logger.info(
            f"ShardedManifestWriter: {total} utterances across {len(self._shard_counts)} shards, {done} completed with .done"
        )

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}

    def ray_stage_spec(self) -> dict[str, Any]:
        # Force a single persistent actor so the in-memory `_shard_counts`
        # accumulator sees every row for each shard. Without this, Ray Data
        # runs the writer as parallel stateless tasks with fresh per-task
        # state, and `.done` markers never get written.
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}
