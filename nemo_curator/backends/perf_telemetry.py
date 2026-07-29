# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Executor-side collection for opt-in performance telemetry.

This mixin intentionally implements #2223's ``consume_external_perf_records``
contract instead of duplicating terminal-writer discovery or summary logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.utils.performance_utils import StagePerfStats

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import Task


class PerformanceTelemetryExecutorMixin:
    """Collect backend-owned invocation and hardware records for #2223."""

    config: dict[str, Any]
    _external_perf_records: list[StagePerfStats]

    def _start_pipeline_hardware_sampler(self) -> list[Any]:
        if not bool(self.config.get("pipeline_hardware_sampler_enabled", False)):
            return []
        try:
            from nemo_curator.utils.pipeline_hardware_sampler import start_pipeline_hardware_samplers

            return start_pipeline_hardware_samplers(
                interval_s=float(self.config.get("pipeline_hardware_sampler_interval_s", 0.5)),
                startup_timeout_s=float(self.config.get("pipeline_hardware_sampler_startup_timeout_s", 5.0)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler disabled: {}", exc)
            return []

    def _stop_pipeline_hardware_sampler(self, sampler_actors: list[Any]) -> StagePerfStats | None:
        if not sampler_actors:
            return None
        try:
            from nemo_curator.utils.pipeline_hardware_sampler import stop_pipeline_hardware_samplers

            metrics = stop_pipeline_hardware_samplers(
                sampler_actors,
                stop_timeout_s=float(self.config.get("pipeline_hardware_sampler_stop_timeout_s", 10.0)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler stop failed: {}", exc)
            return None
        return StagePerfStats(
            stage_name="pipeline_hardware_sampler",
            process_time=float(metrics.pop("pipeline_hardware_wall_time_s", 0.0)),
            num_items_processed=1,
            custom_metrics=metrics,
        )

    @staticmethod
    def _start_stage_perf_collector(stages: list[ProcessingStage]) -> Any | None:  # noqa: ANN401
        try:
            from nemo_curator.utils.stage_perf_collector import start_stage_perf_collector

            return start_stage_perf_collector(stages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stage performance collector disabled: {}", exc)
            return None

    @staticmethod
    def _stop_stage_perf_collector(collector: object | None, stages: list[ProcessingStage]) -> list[Any]:
        try:
            from nemo_curator.utils.stage_perf_collector import stop_stage_perf_collector

            return stop_stage_perf_collector(collector, stages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stage performance collector stop failed: {}", exc)
            return []

    def _finalize_performance_telemetry(
        self,
        *,
        stages: list[ProcessingStage],
        tasks: list[Task],
        stage_perf_collector: object | None,
        hardware_sampler: list[Any],
    ) -> None:
        records = self._stop_stage_perf_collector(stage_perf_collector, stages)
        external = [record.perf_stats for record in records]

        # Direct executor callers still receive records that could not travel on
        # a surviving output. Pipeline's terminal summary deduplicates them by
        # invocation id when it consumes the authoritative external list.
        unattached = [record.perf_stats for record in records if not record.attached_to_output]
        for task in tasks:
            for perf_stats in unattached:
                task.add_stage_perf(perf_stats)

        hardware_perf = self._stop_pipeline_hardware_sampler(hardware_sampler)
        if hardware_perf is not None:
            external.append(hardware_perf)
            for task in tasks:
                task.add_stage_perf(hardware_perf)
        self._external_perf_records = external

    def consume_external_perf_records(self) -> list[StagePerfStats]:
        """Return and clear records retained for #2223's driver finalizer."""
        records = list(getattr(self, "_external_perf_records", []))
        self._external_perf_records = []
        return records
