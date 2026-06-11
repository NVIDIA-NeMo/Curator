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
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.audio.metrics.performance import AudioPerformanceSummary, serialize_stage_perf
from nemo_curator.tasks import AudioTask, FileGroupTask


@dataclass
class ShardedManifestWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Write AudioTasks to per-shard JSONL files mirroring the input manifest path structure.

    Output structure mirrors the input manifest paths::

        output_dir/
          yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl
          yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl.done
          yodas/0_from_captions/en/sharded_manifests/manifest_42_perf.jsonl
        perf_summary.json

    The shard key is extracted from ``task._metadata["_shard_key"]``
    which is set by ``NemoTarShardReaderStage`` as a relative path
    (e.g. ``yodas/0_from_captions/en/sharded_manifests/manifest_42``).

    Args:
        output_dir: Root directory for output manifests.
        final_manifest_path: Optional JSONL path that also receives every
            output row. This is useful for single-rank tutorial runs; the
            sharded files remain the primary output.
        write_perf_stats: If True, write per-task stage perf to a sibling
            ``_perf.jsonl`` file and refresh an aggregate
            ``perf_summary.json`` whenever a shard completes.
    """

    output_dir: str
    name: str = "sharded_manifest_writer"
    final_manifest_path: str | None = None
    write_perf_stats: bool = True
    duration_key: str = "duration"
    drop_manifest_keys: tuple[str, ...] = ("waveform",)
    _perf_summary: AudioPerformanceSummary = field(init=False, repr=False)
    _writer_manifest_write_time_s: float = field(default=0.0, repr=False)
    _writer_perf_write_time_s: float = field(default=0.0, repr=False)
    _writer_done_write_time_s: float = field(default=0.0, repr=False)
    _writer_process_calls: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._perf_summary = AudioPerformanceSummary(duration_key=self.duration_key)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.final_manifest_path:
            final_parent = os.path.dirname(self.final_manifest_path)
            if final_parent:
                os.makedirs(final_parent, exist_ok=True)
            if os.path.exists(self.final_manifest_path):
                os.remove(self.final_manifest_path)
        self._perf_summary.reset_wall_timer()
        logger.info(f"ShardedManifestWriterStage: output_dir={self.output_dir}")

    def _manifest_data(self, task: AudioTask) -> dict[str, Any]:
        data: dict[str, Any] = {}
        drop_keys = set(self.drop_manifest_keys)
        for key, value in task.data.items():
            if key in drop_keys:
                continue
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                logger.debug("Dropping array-like manifest key {} from writer output", key)
                continue
            try:
                json.dumps(value, ensure_ascii=False)
            except TypeError as exc:
                msg = f"Task {task.task_id} contains non-JSON-serializable manifest key {key!r}"
                raise TypeError(msg) from exc
            data[key] = value
        return data

    def _write_perf_line(self, task: AudioTask, shard_key: str) -> None:
        """Append one task's stage perf chain to the shard's perf JSONL."""
        perf_path = os.path.join(self.output_dir, f"{shard_key}_perf.jsonl")
        line = {
            "task_id": task.task_id,
            "stages": serialize_stage_perf(task._stage_perf or []),
        }
        t0 = time.perf_counter()
        with open(perf_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._writer_perf_write_time_s += time.perf_counter() - t0

    def process(self, task: AudioTask) -> FileGroupTask:
        self._writer_process_calls += 1
        shard_key = task._metadata.get("_shard_key", "unknown/shard_0")

        out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        manifest_data = self._manifest_data(task)
        write_t0 = time.perf_counter()
        with open(out_path, "a", encoding="utf-8") as f:
            line = json.dumps(manifest_data, ensure_ascii=False) + "\n"
            f.write(line)
        if self.final_manifest_path:
            with open(self.final_manifest_path, "a", encoding="utf-8") as f:
                f.write(line)
        self._writer_manifest_write_time_s += time.perf_counter() - write_t0

        self._perf_summary.record_task(task, shard_key=shard_key, include_stage_perf=self.write_perf_stats)
        if self.write_perf_stats:
            self._write_perf_line(task, shard_key)

        shard_total = task._metadata.get("_shard_total", 0)
        if shard_total > 0 and self._perf_summary.shard_count(shard_key) >= shard_total:
            done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
            done_t0 = time.perf_counter()
            with open(done_path, "w") as f:
                f.write(f"{self._perf_summary.shard_count(shard_key)}\n")
            self._writer_done_write_time_s += time.perf_counter() - done_t0
            logger.info(
                f"Shard {shard_key} complete: "
                f"{self._perf_summary.shard_count(shard_key)} utterances, wrote {done_path}"
            )
            if self.write_perf_stats:
                self._write_perf_summary()

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[out_path],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def process_batch(self, tasks: list[AudioTask]) -> list[FileGroupTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        return [self.process(task) for task in tasks]

    def _write_perf_summary(self) -> None:
        """Write aggregate perf_summary.json at the output root."""
        writer_total_time = (
            self._writer_manifest_write_time_s
            + self._writer_perf_write_time_s
            + self._writer_done_write_time_s
        )
        writer_summary = {
            "total_process_time_s": writer_total_time,
            "total_items_processed": float(self._writer_process_calls),
            "invocation_count": float(self._writer_process_calls),
            "throughput_items_per_s": (
                float(self._writer_process_calls) / writer_total_time if writer_total_time > 0 else 0.0
            ),
            "custom_metrics_sum": {
                "manifest_write_time_s": self._writer_manifest_write_time_s,
                "perf_write_time_s": self._writer_perf_write_time_s,
                "done_marker_write_time_s": self._writer_done_write_time_s,
                "writer_process_calls": float(self._writer_process_calls),
            },
        }

        summary = self._perf_summary.build_summary(extra_stage_summaries={self.name: writer_summary})
        summary_path = os.path.join(self.output_dir, "perf_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote perf_summary.json: {summary_path}")

    def teardown(self) -> None:
        total = self._perf_summary.total_utterances
        done = sum(
            1 for k in self._perf_summary.shard_keys
            if os.path.exists(os.path.join(self.output_dir, f"{k}.jsonl.done"))
        )
        logger.info(
            f"ShardedManifestWriter: {total} utterances across "
            f"{len(self._perf_summary.shard_keys)} shards, {done} completed with .done"
        )

        if self.write_perf_stats:
            self._write_perf_summary()

    def num_workers(self) -> int | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
