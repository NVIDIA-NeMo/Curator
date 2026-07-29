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

"""Run-scoped Ray actors and executor hooks for opt-in hardware telemetry."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import ray
from loguru import logger

from nemo_curator.utils.gpu_sampler import (
    GpuUtilSampler,
    aggregate_pipeline_hardware_metrics,
    pipeline_node_hardware_metrics,
)
from nemo_curator.utils.performance_utils import StagePerfStats


class _PipelineHardwareSampler:
    def __init__(self, interval_s: float) -> None:
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.started_at = time.time()
        self.sampler = GpuUtilSampler(interval_s=interval_s, sample_all_visible=True, aggregate_only=True)
        self.sampler.start()

    def get_node_id(self) -> str:
        return self.node_id

    def stop(self) -> dict[str, float]:
        diagnostics = self.sampler.diagnostics()
        self.sampler.stop()
        return pipeline_node_hardware_metrics(
            node_id=self.node_id,
            wall_time_s=time.time() - self.started_at,
            aggregate_stats=self.sampler.aggregate_stats(),
            diagnostics=diagnostics,
        )


def _node_affinity_strategy(node_id: str) -> object:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def start_pipeline_hardware_samplers(interval_s: float, timeout_s: float) -> list[Any]:
    remote_cls = ray.remote(num_cpus=0)(_PipelineHardwareSampler)
    pending: dict[Any, Any] = {}
    for node in ray.nodes():
        node_id = str(node.get("NodeID", ""))
        if not node.get("Alive") or not node_id:
            continue
        try:
            actor = remote_cls.options(scheduling_strategy=_node_affinity_strategy(node_id)).remote(interval_s)
            pending[actor.get_node_id.remote()] = actor
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to start pipeline hardware sampler on node {}: {}", node_id, exc)
    if not pending:
        return []

    ready, waiting = ray.wait(list(pending), num_returns=len(pending), timeout=max(0.0, timeout_s))
    actors = []
    for ref in ready:
        try:
            ray.get(ref)
            actors.append(pending[ref])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler actor failed during startup: {}", exc)
            with contextlib.suppress(Exception):
                ray.kill(pending[ref], no_restart=True)
    for ref in waiting:
        logger.debug("Skipping pipeline hardware sampler actor that did not start within {}s", timeout_s)
        with contextlib.suppress(Exception):
            ray.kill(pending[ref], no_restart=True)
    return actors


def stop_pipeline_hardware_samplers(actors: list[Any], timeout_s: float) -> dict[str, float]:
    pending = {}
    for actor in actors:
        try:
            pending[actor.stop.remote()] = actor
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to request pipeline hardware sampler stop: {}", exc)
    if not pending:
        return {}

    ready, waiting = ray.wait(list(pending), num_returns=len(pending), timeout=max(0.0, timeout_s))
    for ref in waiting:
        logger.debug("Killing pipeline hardware sampler actor that did not stop within {}s", timeout_s)
        with contextlib.suppress(Exception):
            ray.kill(pending[ref], no_restart=True)
    results = []
    for ref in ready:
        try:
            results.append(ray.get(ref))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler stop failed: {}", exc)
    return aggregate_pipeline_hardware_metrics(results)


class PerformanceTelemetryExecutorMixin:
    """Add backend-owned aggregate hardware records to shared telemetry."""

    config: dict[str, Any]
    _external_perf_records: list[StagePerfStats]

    def _start_pipeline_hardware_sampler(self) -> list[Any]:
        if not self.config.get("pipeline_hardware_sampler_enabled", False):
            return []
        try:
            return start_pipeline_hardware_samplers(
                float(self.config.get("pipeline_hardware_sampler_interval_s", 0.5)),
                float(self.config.get("pipeline_hardware_sampler_startup_timeout_s", 5.0)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler disabled: {}", exc)
            return []

    def _stop_pipeline_hardware_sampler(self, actors: list[Any]) -> StagePerfStats | None:
        if not actors:
            return None
        try:
            metrics = stop_pipeline_hardware_samplers(
                actors,
                float(self.config.get("pipeline_hardware_sampler_stop_timeout_s", 10.0)),
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

    def _finalize_pipeline_hardware_sampler(self, hardware_sampler: list[Any], *, keep_record: bool) -> None:
        """Stop pipeline samplers and optionally append their aggregate record."""
        hardware = self._stop_pipeline_hardware_sampler(hardware_sampler)
        if keep_record and hardware is not None:
            self._external_perf_records.append(hardware)
