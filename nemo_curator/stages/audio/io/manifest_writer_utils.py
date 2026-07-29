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

"""Run-scoped metrics for the terminal audio manifest writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.metrics.performance import AudioPerformanceSummary

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask
    from nemo_curator.utils.performance_utils import StagePerfStats


@dataclass
class AudioManifestWriterMetrics:
    """Writer-local metrics and terminal perf-summary accumulator."""

    stage_name: str
    duration_key: str = "duration"
    write_perf_stats: bool = False
    _perf_summary: AudioPerformanceSummary = field(init=False, repr=False)
    _writer_manifest_write_time_s: float = field(default=0.0, repr=False)
    _writer_perf_write_time_s: float = field(default=0.0, repr=False)
    _writer_invocation_count: int = field(default=0, repr=False)
    _writer_items_processed: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._perf_summary = AudioPerformanceSummary(duration_key=self.duration_key)

    @property
    def total_utterances(self) -> int:
        return self._perf_summary.total_utterances

    @property
    def total_audio_seconds(self) -> float:
        return self._perf_summary.total_audio_seconds

    @property
    def items_processed(self) -> int:
        return self._writer_items_processed

    def record_invocation(self, item_count: int) -> None:
        self._writer_invocation_count += 1
        self._writer_items_processed += item_count

    def add_manifest_write_time(self, elapsed_s: float) -> None:
        self._writer_manifest_write_time_s += elapsed_s

    def add_perf_write_time(self, elapsed_s: float) -> None:
        self._writer_perf_write_time_s += elapsed_s

    def record_task(self, task: AudioTask) -> None:
        self._perf_summary.record_task(task, include_stage_perf=self.write_perf_stats)

    def record_stage_perf(self, stage_perf: list[StagePerfStats]) -> None:
        """Record executor-published authoritative invocation telemetry."""
        self._perf_summary.record_stage_perf(stage_perf)

    def build_stage_summaries(self) -> dict[str, dict[str, Any]]:
        """Build only accumulated stage entries for external merge."""
        return self._perf_summary.build_stage_summaries()

    @property
    def perf_invocations_counted(self) -> int:
        return self._perf_summary.perf_invocations_counted

    def build_writer_summary(self) -> dict[str, Any]:
        writer_total_time = self._writer_manifest_write_time_s + self._writer_perf_write_time_s
        return {
            "total_process_time_s": writer_total_time,
            "total_items_processed": float(self._writer_items_processed),
            "invocation_count": float(self._writer_invocation_count),
            "throughput_items_per_s": (
                float(self._writer_items_processed) / writer_total_time if writer_total_time > 0 else 0.0
            ),
            "custom_metrics_sum": {
                "manifest_write_time_s": self._writer_manifest_write_time_s,
                "perf_write_time_s": self._writer_perf_write_time_s,
                "writer_process_calls": float(self._writer_invocation_count),
                "writer_invocation_count": float(self._writer_invocation_count),
                "writer_items_processed": float(self._writer_items_processed),
                "pipeline_output_rows": float(self._perf_summary.total_utterances),
                "pipeline_output_audio_s": self._perf_summary.total_audio_seconds,
            },
        }

    def build_perf_summary(
        self,
        *,
        run_id: str = "",
        executor: str = "",
        pipeline_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._perf_summary.build_summary(
            extra_stage_summaries={self.stage_name: self.build_writer_summary()},
            run_id=run_id,
            executor=executor,
            pipeline_metadata=pipeline_metadata,
        )

    def build_external_stage_summary(self, perf_stats: StagePerfStats) -> dict[str, Any] | None:
        """Render one externally collected perf record in the normal stage-summary shape."""
        perf_summary = AudioPerformanceSummary(duration_key=self.duration_key)
        perf_summary.record_stage_perf([perf_stats])
        stage_key = str(getattr(perf_stats, "stage_id", "") or perf_stats.stage_name)
        return perf_summary.build_stage_summaries().get(stage_key)
