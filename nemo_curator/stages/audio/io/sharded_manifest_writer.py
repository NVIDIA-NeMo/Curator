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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask


def _serialize_stage_perf(stage_perf_list: list) -> list[dict]:
    """Serialize a list of StagePerfStats to JSON-friendly dicts."""
    result = []
    for perf in stage_perf_list:
        entry: dict[str, Any] = {
            "stage_name": perf.stage_name,
            "process_time": perf.process_time,
            "actor_idle_time": perf.actor_idle_time,
            "input_data_size_mb": perf.input_data_size_mb,
            "num_items_processed": perf.num_items_processed,
        }
        if perf.custom_metrics:
            entry["custom_metrics"] = dict(perf.custom_metrics)
        result.append(entry)
    return result


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
        write_perf_stats: If True, write per-task stage perf to a sibling
            ``_perf.jsonl`` file and an aggregate ``perf_summary.json``.
    """

    name: str = "sharded_manifest_writer"
    output_dir: str = ""
    write_perf_stats: bool = True
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _stage_totals: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)), repr=False)
    _total_utterances: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for ShardedManifestWriterStage"
            raise ValueError(msg)

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
        logger.info(f"ShardedManifestWriterStage: output_dir={self.output_dir}")

    def _accumulate_perf(self, task: AudioTask) -> None:
        """Accumulate per-stage metrics from the task for the summary."""
        for perf in (task._stage_perf or []):
            totals = self._stage_totals[perf.stage_name]
            totals["process_time"] += perf.process_time
            totals["num_items_processed"] += perf.num_items_processed
            totals["batch_count"] += 1
            for k, v in (perf.custom_metrics or {}).items():
                totals[f"custom.{k}"] += v
        self._total_utterances += 1

    def _write_perf_line(self, task: AudioTask, shard_key: str) -> None:
        """Append one task's stage perf chain to the shard's perf JSONL."""
        perf_path = os.path.join(self.output_dir, f"{shard_key}_perf.jsonl")
        line = {
            "task_id": task.task_id,
            "stages": _serialize_stage_perf(task._stage_perf or []),
        }
        with open(perf_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def process(self, task: AudioTask) -> FileGroupTask:
        shard_key = task._metadata.get("_shard_key", "unknown/shard_0")

        out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")

        if self.write_perf_stats:
            self._accumulate_perf(task)
            self._write_perf_line(task, shard_key)

        self._shard_counts[shard_key] += 1

        shard_total = task._metadata.get("_shard_total", 0)
        if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
            done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
            with open(done_path, "w") as f:
                f.write(f"{self._shard_counts[shard_key]}\n")
            logger.info(f"Shard {shard_key} complete: {self._shard_counts[shard_key]} utterances, wrote {done_path}")

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
        stages_summary: dict[str, dict[str, float]] = {}
        for stage_name, totals in self._stage_totals.items():
            entry: dict[str, float] = {
                "total_process_time_s": totals.get("process_time", 0.0),
                "total_items_processed": totals.get("num_items_processed", 0.0),
                "batch_count": totals.get("batch_count", 0.0),
            }
            batch_count = totals.get("batch_count", 0.0)
            if batch_count > 0:
                entry["avg_batch_time_s"] = totals.get("process_time", 0.0) / batch_count
            total_time = totals.get("process_time", 0.0)
            total_items = totals.get("num_items_processed", 0.0)
            if total_time > 0 and total_items > 0:
                entry["throughput_items_per_s"] = total_items / total_time
            for k, v in totals.items():
                if k.startswith("custom."):
                    entry[f"{k}_sum"] = v
            stages_summary[stage_name] = entry

        summary = {
            "total_utterances": self._total_utterances,
            "stages": stages_summary,
        }
        summary_path = os.path.join(self.output_dir, "perf_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote perf_summary.json: {summary_path}")

    def teardown(self) -> None:
        total = sum(self._shard_counts.values())
        done = sum(
            1 for k in self._shard_counts
            if os.path.exists(os.path.join(self.output_dir, f"{k}.jsonl.done"))
        )
        logger.info(f"ShardedManifestWriter: {total} utterances across {len(self._shard_counts)} shards, {done} completed with .done")

        if self.write_perf_stats:
            self._write_perf_summary()

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
