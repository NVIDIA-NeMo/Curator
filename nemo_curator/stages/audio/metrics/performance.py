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

"""Reusable audio pipeline performance summary helpers.

Audio stages should record processor-specific counters and timings through
``ProcessingStage._log_metrics()``. Backends attach those metrics to
``Task._stage_perf`` as ``StagePerfStats``. Terminal audio stages can use the
helpers in this module to serialize per-task perf chains and build aggregate
pipeline throughput summaries without adding backend-specific log scraping.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StagePerfStats


def serialize_stage_perf(stage_perf_list: list[StagePerfStats]) -> list[dict[str, Any]]:
    """Serialize a task's stage performance chain to JSON-friendly dicts."""
    result: list[dict[str, Any]] = []
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


def _task_audio_seconds(task: Task, duration_key: str) -> float:
    data = getattr(task, "data", {})
    if not isinstance(data, dict):
        return 0.0
    try:
        seconds = float(data.get(duration_key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return seconds if seconds > 0 else 0.0


def _add_ratio(entry: dict[str, Any], name: str, numerator: float, denominator: float) -> None:
    if numerator > 0 and denominator > 0:
        entry[name] = numerator / denominator


def _build_stage_summary(stage_totals: dict[str, float], custom_totals: dict[str, float]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "total_process_time_s": stage_totals.get("process_time", 0.0),
        "total_actor_idle_time_s": stage_totals.get("actor_idle_time", 0.0),
        "total_input_data_size_mb": stage_totals.get("input_data_size_mb", 0.0),
        "total_items_processed": stage_totals.get("num_items_processed", 0.0),
        "invocation_count": stage_totals.get("invocation_count", 0.0),
    }

    invocation_count = stage_totals.get("invocation_count", 0.0)
    total_time = stage_totals.get("process_time", 0.0)
    total_items = stage_totals.get("num_items_processed", 0.0)

    _add_ratio(entry, "avg_invocation_time_s", total_time, invocation_count)
    _add_ratio(entry, "throughput_items_per_s", total_items, total_time)

    custom_sums = dict(custom_totals)
    if not custom_sums:
        return entry

    entry["custom_metrics_sum"] = custom_sums

    audio_seconds = custom_sums.get("audio_duration_s", 0.0)
    inference_time = custom_sums.get("inference_time_s", custom_sums.get("inference_time", 0.0))
    output_tokens = custom_sums.get("output_tokens", 0.0)
    output_chars = custom_sums.get("output_chars", 0.0)
    waveform_bytes = custom_sums.get("waveform_bytes", 0.0)
    waveform_mb = waveform_bytes / 1024.0 / 1024.0

    _add_ratio(entry, "throughput_audio_s_per_process_s", audio_seconds, total_time)
    _add_ratio(entry, "throughput_audio_s_per_inference_s", audio_seconds, inference_time)
    _add_ratio(entry, "avg_audio_s_per_item", audio_seconds, total_items)
    _add_ratio(entry, "throughput_output_tokens_per_process_s", output_tokens, total_time)
    _add_ratio(entry, "throughput_output_tokens_per_inference_s", output_tokens, inference_time)
    _add_ratio(entry, "throughput_output_chars_per_process_s", output_chars, total_time)
    _add_ratio(entry, "throughput_output_chars_per_inference_s", output_chars, inference_time)
    _add_ratio(entry, "throughput_waveform_mb_per_process_s", waveform_mb, total_time)

    input_tasks = custom_sums.get("input_tasks", 0.0)
    output_tasks = custom_sums.get("output_tasks", 0.0)
    _add_ratio(entry, "output_tasks_per_input_task", output_tasks, input_tasks)

    input_shards = custom_sums.get("input_shards", 0.0)
    utterances_emitted = custom_sums.get("utterances_emitted", custom_sums.get("output_utterances", 0.0))
    _add_ratio(entry, "utterances_emitted_per_input_shard", utterances_emitted, input_shards)

    utterances_input = custom_sums.get("utterances_input", input_tasks)
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
            _add_ratio(entry, f"{metric_name}_per_input_utterance", metric_value, utterances_input)

    return entry


@dataclass
class AudioPerformanceSummary:
    """Accumulate and summarize audio task performance metrics.

    This class is intentionally independent of any writer implementation. A
    terminal audio stage can call ``record_task`` for each output task and then
    write ``build_summary()`` wherever its output contract requires.
    """

    duration_key: str = "duration"
    _stage_totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
        repr=False,
    )
    _stage_custom_totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
        repr=False,
    )
    _seen_perf_invocations: set[str] = field(default_factory=set, repr=False)
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _shard_audio_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float), repr=False)
    _total_utterances: int = field(default=0, repr=False)
    _total_audio_seconds: float = field(default=0.0, repr=False)
    _wall_start_s: float = field(default_factory=time.perf_counter, repr=False)

    @property
    def total_utterances(self) -> int:
        return self._total_utterances

    @property
    def shard_keys(self) -> list[str]:
        return sorted(self._shard_counts)

    def shard_count(self, shard_key: str) -> int:
        return self._shard_counts.get(shard_key, 0)

    def reset_wall_timer(self) -> None:
        self._wall_start_s = time.perf_counter()

    def record_task(self, task: Task, shard_key: str | None = None, *, include_stage_perf: bool = True) -> None:
        """Record one audio task and optionally its attached stage perf chain."""
        audio_seconds = _task_audio_seconds(task, self.duration_key)
        self._total_utterances += 1
        self._total_audio_seconds += audio_seconds

        if shard_key is not None:
            self._shard_counts[shard_key] += 1
            self._shard_audio_seconds[shard_key] += audio_seconds

        if include_stage_perf:
            self.record_stage_perf(getattr(task, "_stage_perf", []) or [])

    def record_stage_perf(self, stage_perf_list: list[StagePerfStats]) -> None:
        """Accumulate a list of ``StagePerfStats``, deduplicating batch invocations."""
        for perf in stage_perf_list:
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

            for key, value in (perf.custom_metrics or {}).items():
                self._stage_custom_totals[perf.stage_name][key] += value

    def build_stage_summaries(self) -> dict[str, dict[str, Any]]:
        """Build per-stage aggregate summaries from accumulated metrics."""
        return {
            stage_name: _build_stage_summary(
                dict(totals),
                dict(self._stage_custom_totals.get(stage_name, {})),
            )
            for stage_name, totals in self._stage_totals.items()
        }

    def build_summary(
        self,
        *,
        extra_stage_summaries: dict[str, dict[str, Any]] | None = None,
        wall_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Build the full audio pipeline performance summary."""
        resolved_wall_time_s = (
            max(time.perf_counter() - self._wall_start_s, 0.0) if wall_time_s is None else max(wall_time_s, 0.0)
        )
        stages_summary = self.build_stage_summaries()
        if extra_stage_summaries:
            stages_summary.update(extra_stage_summaries)

        return {
            "total_utterances": self._total_utterances,
            "total_audio_seconds": self._total_audio_seconds,
            "total_audio_hours": self._total_audio_seconds / 3600.0,
            "writer_wall_time_s": resolved_wall_time_s,
            "pipeline_audio_s_per_wall_s": (
                self._total_audio_seconds / resolved_wall_time_s if resolved_wall_time_s > 0 else 0.0
            ),
            "pipeline_utterances_per_wall_s": (
                self._total_utterances / resolved_wall_time_s if resolved_wall_time_s > 0 else 0.0
            ),
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
