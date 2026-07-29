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

"""Run-scoped performance accounting for the terminal audio manifest writer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fsspec.core import url_to_fs

from nemo_curator.stages.audio.metrics.performance import AudioPerformanceSummary

if TYPE_CHECKING:
    from nemo_curator.tasks import Task
    from nemo_curator.utils.performance_utils import StagePerfStats


@dataclass(repr=False)
class AudioManifestWriterMetrics:
    """Writer-local metrics and terminal perf-summary accumulator."""

    stage_name: str
    duration_key: str = "duration"
    write_perf_stats: bool = False
    _perf_summary: AudioPerformanceSummary = field(init=False)
    _writer_manifest_write_time_s: float = 0.0
    _writer_invocation_count: int = 0
    _writer_items_processed: int = 0
    _writer_custom_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._perf_summary = AudioPerformanceSummary(duration_key=self.duration_key)

    def record_output_invocation(
        self,
        tasks: list[Task],
        *,
        manifest_write_time_s: float,
        extra_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Record one writer call and return its task-carried metric delta."""
        previous_audio_s = self._perf_summary.total_audio_seconds
        previous_duration_rows = self._perf_summary.duration_utterances
        self._writer_invocation_count += 1
        self._writer_items_processed += len(tasks)
        self._writer_manifest_write_time_s += manifest_write_time_s
        for task in tasks:
            self._perf_summary.record_task(task, include_stage_perf=self.write_perf_stats)
        metrics = {
            "manifest_write_time_s": float(manifest_write_time_s),
            "writer_process_calls": 1.0,
            "writer_invocation_count": 1.0,
            "writer_items_processed": float(len(tasks)),
            "pipeline_output_rows": float(len(tasks)),
            "pipeline_output_audio_s": self._perf_summary.total_audio_seconds - previous_audio_s,
            "pipeline_output_duration_rows": float(self._perf_summary.duration_utterances - previous_duration_rows),
        }
        for name, value in (extra_metrics or {}).items():
            metric_value = float(value)
            metrics[name] = metrics.get(name, 0.0) + metric_value
            self._writer_custom_metrics[name] = self._writer_custom_metrics.get(name, 0.0) + metric_value
        return metrics

    def build_writer_summary(self) -> dict[str, Any]:
        writer_total_time = self._writer_manifest_write_time_s
        return {
            "total_process_time_s": writer_total_time,
            "total_items_processed": float(self._writer_items_processed),
            "invocation_count": float(self._writer_invocation_count),
            "throughput_items_per_s": (
                float(self._writer_items_processed) / writer_total_time if writer_total_time > 0 else 0.0
            ),
            "custom_metrics_sum": {
                "manifest_write_time_s": self._writer_manifest_write_time_s,
                "writer_process_calls": float(self._writer_invocation_count),
                "writer_invocation_count": float(self._writer_invocation_count),
                "writer_items_processed": float(self._writer_items_processed),
                "pipeline_output_rows": float(self._perf_summary.total_utterances),
                "pipeline_output_audio_s": self._perf_summary.total_audio_seconds,
                "pipeline_output_duration_rows": float(self._perf_summary.duration_utterances),
                **self._writer_custom_metrics,
            },
        }

    def build_perf_summary(
        self,
        *,
        stage_id: str = "",
        wall_time_s: float | None = None,
        run_id: str = "",
        executor: str = "",
        pipeline_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        writer_key = stage_id or self.stage_name
        resolved_writer_summary = self.build_writer_summary()
        if writer_key != self.stage_name:
            resolved_writer_summary.setdefault("stage_name", self.stage_name)
        recorded_stages = self._perf_summary.build_stage_summaries()
        extra_stage_summaries = None
        if writer_key not in recorded_stages:
            extra_stage_summaries = {writer_key: resolved_writer_summary}
        return self._perf_summary.build_summary(
            extra_stage_summaries=extra_stage_summaries,
            wall_time_s=wall_time_s,
            run_id=run_id,
            executor=executor,
            pipeline_metadata=pipeline_metadata,
        )


class TerminalAudioPerformanceWriterMixin:
    """Shared terminal-summary lifecycle for audio writer stages."""

    name: str
    write_perf_stats: bool
    duration_key: str
    perf_summary_path: str | None
    perf_run_id: str
    perf_executor: str
    perf_pipeline_metadata: dict[str, Any] | None
    _curator_run_id: str
    _curator_executor: str
    _curator_stage_id: str
    _curator_pipeline_metadata: dict[str, Any] | None
    _writer_metrics: AudioManifestWriterMetrics
    _external_perf_stats: list[StagePerfStats]

    def _reset_writer_metrics(self) -> None:
        self._writer_metrics = AudioManifestWriterMetrics(
            stage_name=self.name,
            duration_key=self.duration_key,
            write_perf_stats=self.write_perf_stats,
        )
        self._external_perf_stats = []

    def _default_perf_summary_path(self) -> str:
        raise NotImplementedError

    def _resolved_perf_summary_path(self) -> str:
        return self.perf_summary_path or self._default_perf_summary_path()

    def _remove_existing_perf_summary(self) -> None:
        if not self.write_perf_stats:
            return
        perf_fs, perf_path = url_to_fs(self._resolved_perf_summary_path())
        try:
            if perf_fs.exists(perf_path):
                perf_fs.rm(perf_path)
        except OSError:
            # Summary cleanup is best effort on shared/object filesystems.
            return

    def _resolved_perf_context(self) -> tuple[str, str, dict[str, Any]]:
        pipeline_metadata = dict(self._curator_pipeline_metadata or {})
        pipeline_metadata.update(self.perf_pipeline_metadata or {})
        return (
            self.perf_run_id or self._curator_run_id,
            self.perf_executor or self._curator_executor,
            pipeline_metadata,
        )

    def _write_perf_summary(
        self,
        *,
        wall_time_s: float | None = None,
        status: str = "completed",
        preserve_existing_stages: bool = True,
    ) -> None:
        perf_fs, perf_path = url_to_fs(self._resolved_perf_summary_path())
        parent_dir = "/".join(perf_path.split("/")[:-1])
        if parent_dir:
            perf_fs.makedirs(parent_dir, exist_ok=True)
        run_id, executor, pipeline_metadata = self._resolved_perf_context()
        summary = self._writer_metrics.build_perf_summary(
            stage_id=self._curator_stage_id,
            wall_time_s=wall_time_s,
            run_id=run_id,
            executor=executor,
            pipeline_metadata=pipeline_metadata,
        )
        if preserve_existing_stages:
            try:
                with perf_fs.open(perf_path, encoding="utf-8") as f:
                    existing = json.load(f)
                existing_stages = existing.get("stages", {})
                if isinstance(existing_stages, dict):
                    stages = summary.setdefault("stages", {})
                    for stage_key, stage_summary in existing_stages.items():
                        stages.setdefault(stage_key, stage_summary)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                pass
        summary["status"] = status
        with perf_fs.open(perf_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def record_external_stage_perf(self, perf_stats: StagePerfStats) -> bool:
        return self.record_external_stage_perfs([perf_stats])

    def record_external_stage_perfs(self, perf_stats: list[StagePerfStats]) -> bool:
        if not self.write_perf_stats:
            return False
        self._external_perf_stats.extend(perf_stats)
        return True

    def teardown(self) -> None:
        if self.write_perf_stats and not self._curator_run_id:
            self._writer_metrics._perf_summary.record_stage_perf(self._external_perf_stats)
            self._write_perf_summary()

    def finalize_performance_summary(
        self,
        tasks: list[Task],
        *,
        external_perf_stats: list[StagePerfStats],
        wall_time_s: float,
    ) -> None:
        if not self.write_perf_stats:
            return
        final_metrics = AudioManifestWriterMetrics(
            stage_name=self.name,
            duration_key=self.duration_key,
            write_perf_stats=True,
        )
        for task in tasks:
            final_metrics.record_output_invocation([task], manifest_write_time_s=0.0)
        final_metrics._perf_summary.record_stage_perf([*self._external_perf_stats, *external_perf_stats])
        self._writer_metrics = final_metrics
        self._write_perf_summary(wall_time_s=wall_time_s)
        self._external_perf_stats = []
