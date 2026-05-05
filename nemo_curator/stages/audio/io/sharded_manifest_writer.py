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

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask


def _serialize_stage_perf(stage_perf_list: list) -> list[dict]:
    """Serialize a list of StagePerfStats to JSON-friendly dicts."""
    result = []
    for perf in stage_perf_list:
        entry: dict[str, Any] = {
            "invocation_id": getattr(perf, "invocation_id", ""),
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
            ``_perf.jsonl`` file and refresh an aggregate
            ``perf_summary.json`` whenever a shard completes.
    """

    name: str = "sharded_manifest_writer"
    output_dir: str = ""
    write_perf_stats: bool = True
    duration_key: str = "duration"
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _shard_audio_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float), repr=False)
    _stage_totals: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)), repr=False)
    _stage_custom_totals: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)), repr=False)
    _seen_perf_invocations: set[str] = field(default_factory=set, repr=False)
    _total_utterances: int = field(default=0, repr=False)
    _total_audio_seconds: float = field(default=0.0, repr=False)
    _wall_start_s: float = field(default=0.0, repr=False)
    _writer_manifest_write_time_s: float = field(default=0.0, repr=False)
    _writer_perf_write_time_s: float = field(default=0.0, repr=False)
    _writer_done_write_time_s: float = field(default=0.0, repr=False)
    _writer_process_calls: int = field(default=0, repr=False)

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
        self._wall_start_s = time.perf_counter()
        logger.info(f"ShardedManifestWriterStage: output_dir={self.output_dir}")

    def _task_audio_seconds(self, task: AudioTask) -> float:
        value = task.data.get(self.duration_key, 0.0)
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = 0.0
        return seconds if seconds > 0 else 0.0

    def _accumulate_task_totals(self, task: AudioTask, shard_key: str) -> None:
        audio_seconds = self._task_audio_seconds(task)
        self._total_utterances += 1
        self._total_audio_seconds += audio_seconds
        self._shard_audio_seconds[shard_key] += audio_seconds

    def _accumulate_perf(self, task: AudioTask) -> None:
        """Accumulate per-stage metrics from the task for the summary."""
        for perf in (task._stage_perf or []):
            invocation_id = getattr(perf, "invocation_id", "")
            if invocation_id:
                if invocation_id in self._seen_perf_invocations:
                    continue
                self._seen_perf_invocations.add(invocation_id)
            totals = self._stage_totals[perf.stage_name]
            totals["process_time"] += perf.process_time
            totals["actor_idle_time"] += perf.actor_idle_time
            totals["input_data_size_mb"] += perf.input_data_size_mb
            totals["num_items_processed"] += perf.num_items_processed
            totals["invocation_count"] += 1
            for k, v in (perf.custom_metrics or {}).items():
                self._stage_custom_totals[perf.stage_name][k] += v

    def _write_perf_line(self, task: AudioTask, shard_key: str) -> None:
        """Append one task's stage perf chain to the shard's perf JSONL."""
        perf_path = os.path.join(self.output_dir, f"{shard_key}_perf.jsonl")
        line = {
            "task_id": task.task_id,
            "stages": _serialize_stage_perf(task._stage_perf or []),
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

        write_t0 = time.perf_counter()
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        self._writer_manifest_write_time_s += time.perf_counter() - write_t0

        if self.write_perf_stats:
            self._accumulate_task_totals(task, shard_key)
            self._accumulate_perf(task)
            self._write_perf_line(task, shard_key)

        self._shard_counts[shard_key] += 1

        shard_total = task._metadata.get("_shard_total", 0)
        if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
            done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
            done_t0 = time.perf_counter()
            with open(done_path, "w") as f:
                f.write(f"{self._shard_counts[shard_key]}\n")
            self._writer_done_write_time_s += time.perf_counter() - done_t0
            logger.info(f"Shard {shard_key} complete: {self._shard_counts[shard_key]} utterances, wrote {done_path}")
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
        stages_summary: dict[str, dict[str, Any]] = {}
        for stage_name, totals in self._stage_totals.items():
            entry: dict[str, Any] = {
                "total_process_time_s": totals.get("process_time", 0.0),
                "total_actor_idle_time_s": totals.get("actor_idle_time", 0.0),
                "total_input_data_size_mb": totals.get("input_data_size_mb", 0.0),
                "total_items_processed": totals.get("num_items_processed", 0.0),
                "invocation_count": totals.get("invocation_count", 0.0),
            }
            invocation_count = totals.get("invocation_count", 0.0)
            if invocation_count > 0:
                entry["avg_invocation_time_s"] = totals.get("process_time", 0.0) / invocation_count
            total_time = totals.get("process_time", 0.0)
            total_items = totals.get("num_items_processed", 0.0)
            if total_time > 0 and total_items > 0:
                entry["throughput_items_per_s"] = total_items / total_time
            custom_sums = dict(self._stage_custom_totals.get(stage_name, {}))
            if custom_sums:
                entry["custom_metrics_sum"] = custom_sums
                audio_seconds = custom_sums.get("audio_duration_s", 0.0)
                if audio_seconds > 0 and total_time > 0:
                    entry["throughput_audio_s_per_process_s"] = audio_seconds / total_time
                if audio_seconds > 0 and total_items > 0:
                    entry["avg_audio_s_per_item"] = audio_seconds / total_items
                inference_time = custom_sums.get("inference_time_s", custom_sums.get("inference_time", 0.0))
                if audio_seconds > 0 and inference_time > 0:
                    entry["throughput_audio_s_per_inference_s"] = audio_seconds / inference_time
                output_tokens = custom_sums.get("output_tokens", 0.0)
                if output_tokens > 0 and total_time > 0:
                    entry["throughput_output_tokens_per_process_s"] = output_tokens / total_time
                if output_tokens > 0 and inference_time > 0:
                    entry["throughput_output_tokens_per_inference_s"] = output_tokens / inference_time
                output_chars = custom_sums.get("output_chars", 0.0)
                if output_chars > 0 and total_time > 0:
                    entry["throughput_output_chars_per_process_s"] = output_chars / total_time
                if output_chars > 0 and inference_time > 0:
                    entry["throughput_output_chars_per_inference_s"] = output_chars / inference_time
                waveform_bytes = custom_sums.get("waveform_bytes", 0.0)
                if waveform_bytes > 0 and total_time > 0:
                    entry["throughput_waveform_mb_per_process_s"] = (waveform_bytes / 1024.0 / 1024.0) / total_time
                input_tasks = custom_sums.get("input_tasks", 0.0)
                output_tasks = custom_sums.get("output_tasks", 0.0)
                if input_tasks > 0:
                    entry["output_tasks_per_input_task"] = output_tasks / input_tasks
                utterances_input = custom_sums.get("utterances_input", custom_sums.get("input_tasks", 0.0))
                if utterances_input > 0:
                    for metric_name in (
                        "utterances_selected",
                        "utterances_skipped",
                        "utterances_processed",
                        "utterances_eligible",
                        "utterances_restored",
                        "utterances_kept_as_is",
                        "utterances_filtered",
                        "utterances_newly_flagged",
                        "utterances_recovered",
                        "pnc_rejected",
                        "empty_after_regex",
                        "wrong_language",
                        "low_probability",
                    ):
                        metric_value = custom_sums.get(metric_name, 0.0)
                        if metric_value:
                            entry[f"{metric_name}_per_input_utterance"] = metric_value / utterances_input
            stages_summary[stage_name] = entry

        writer_total_time = (
            self._writer_manifest_write_time_s
            + self._writer_perf_write_time_s
            + self._writer_done_write_time_s
        )
        stages_summary[self.name] = {
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

        wall_time = max(time.perf_counter() - self._wall_start_s, 0.0) if self._wall_start_s else 0.0
        summary = {
            "total_utterances": self._total_utterances,
            "total_audio_seconds": self._total_audio_seconds,
            "total_audio_hours": self._total_audio_seconds / 3600.0,
            "writer_wall_time_s": wall_time,
            "pipeline_audio_s_per_wall_s": self._total_audio_seconds / wall_time if wall_time > 0 else 0.0,
            "pipeline_utterances_per_wall_s": self._total_utterances / wall_time if wall_time > 0 else 0.0,
            "perf_invocations_counted": len(self._seen_perf_invocations),
            "shards": {
                shard: {
                    "utterances": count,
                    "audio_seconds": self._shard_audio_seconds.get(shard, 0.0),
                    "audio_hours": self._shard_audio_seconds.get(shard, 0.0) / 3600.0,
                }
                for shard, count in sorted(self._shard_counts.items())
            },
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
