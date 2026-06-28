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

"""Shared helpers for audio JSONL manifest writers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.metrics.performance import AudioPerformanceSummary

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask
    from nemo_curator.utils.performance_utils import StagePerfStats


def manifest_data(
    task: AudioTask,
    drop_manifest_keys: tuple[str, ...] = (),
    *,
    drop_array_like_values: bool = False,
) -> dict[str, Any]:
    """Return the manifest row after applying explicit serialization policy."""
    if not drop_manifest_keys and not drop_array_like_values:
        return task.data

    data: dict[str, Any] = {}
    drop_keys = set(drop_manifest_keys)
    for key, value in task.data.items():
        if key in drop_keys:
            continue
        if drop_array_like_values and hasattr(value, "shape") and hasattr(value, "dtype"):
            logger.debug("Dropping array-like manifest key {} from writer output", key)
            continue
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            msg = f"Task {task.task_id} contains non-JSON-serializable manifest key {key!r}"
            raise TypeError(msg) from exc
        data[key] = value
    return data


def manifest_lines(
    tasks: list[AudioTask],
    drop_manifest_keys: tuple[str, ...] = (),
    *,
    drop_array_like_values: bool = False,
) -> list[str]:
    """Serialize ``tasks`` to JSONL lines using the shared audio writer rules."""
    return [
        json.dumps(
            manifest_data(task, drop_manifest_keys, drop_array_like_values=drop_array_like_values),
            ensure_ascii=False,
        )
        + "\n"
        for task in tasks
    ]


@dataclass
class AudioManifestWriterMetrics:
    """Writer-local metrics and terminal perf-summary accumulator."""

    stage_name: str
    duration_key: str = "duration"
    write_perf_stats: bool = False
    _perf_summary: AudioPerformanceSummary = field(init=False, repr=False)
    _writer_manifest_write_time_s: float = field(default=0.0, repr=False)
    _writer_done_write_time_s: float = field(default=0.0, repr=False)
    _writer_perf_write_time_s: float = field(default=0.0, repr=False)
    _writer_invocation_count: int = field(default=0, repr=False)
    _writer_items_processed: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._perf_summary = AudioPerformanceSummary(duration_key=self.duration_key)

    @property
    def total_utterances(self) -> int:
        return self._perf_summary.total_utterances

    @property
    def shard_keys(self) -> list[str]:
        return self._perf_summary.shard_keys

    @property
    def items_processed(self) -> int:
        return self._writer_items_processed

    def reset_wall_timer(self) -> None:
        self._perf_summary.reset_wall_timer()

    def record_invocation(self, item_count: int) -> None:
        self._writer_invocation_count += 1
        self._writer_items_processed += item_count

    def add_manifest_write_time(self, elapsed_s: float) -> None:
        self._writer_manifest_write_time_s += elapsed_s

    def add_done_write_time(self, elapsed_s: float) -> None:
        self._writer_done_write_time_s += elapsed_s

    def add_perf_write_time(self, elapsed_s: float) -> None:
        self._writer_perf_write_time_s += elapsed_s

    def record_task(self, task: AudioTask, shard_key: str | None = None) -> None:
        self._perf_summary.record_task(task, shard_key=shard_key, include_stage_perf=self.write_perf_stats)

    def shard_count(self, shard_key: str) -> int:
        return self._perf_summary.shard_count(shard_key)

    def build_writer_summary(self) -> dict[str, Any]:
        writer_total_time = (
            self._writer_manifest_write_time_s + self._writer_done_write_time_s + self._writer_perf_write_time_s
        )
        return {
            "total_process_time_s": writer_total_time,
            "total_items_processed": float(self._writer_items_processed),
            "invocation_count": float(self._writer_invocation_count),
            "throughput_items_per_s": (
                float(self._writer_items_processed) / writer_total_time if writer_total_time > 0 else 0.0
            ),
            "custom_metrics_sum": {
                "manifest_write_time_s": self._writer_manifest_write_time_s,
                "done_marker_write_time_s": self._writer_done_write_time_s,
                "perf_write_time_s": self._writer_perf_write_time_s,
                "writer_process_calls": float(self._writer_invocation_count),
                "writer_invocation_count": float(self._writer_invocation_count),
                "writer_items_processed": float(self._writer_items_processed),
            },
        }

    def build_perf_summary(self) -> dict[str, Any]:
        return self._perf_summary.build_summary(extra_stage_summaries={self.stage_name: self.build_writer_summary()})

    def build_external_stage_summary(self, perf_stats: StagePerfStats) -> dict[str, Any] | None:
        """Render one externally collected perf record in the normal stage-summary shape."""
        perf_summary = AudioPerformanceSummary(duration_key=self.duration_key)
        perf_summary.record_stage_perf([perf_stats])
        return perf_summary.build_stage_summaries().get(perf_stats.stage_name)
