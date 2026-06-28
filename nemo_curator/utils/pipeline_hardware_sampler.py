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
# ruff: noqa: C901, PLR0912

"""Run-level, observational hardware sampling helpers.

Stage-level GPU samples answer "was this actor busy during this invocation?"
This module samples visible GPUs over the whole pipeline run so benchmark
summaries can also answer "were the requested devices generally occupied?".
It is intentionally fail-open and does not influence placement or autoscaling.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from loguru import logger

from nemo_curator.utils.gpu_sampler import GpuUtilSampler


class _PipelineHardwareSamplerActor:
    def __init__(self, interval_s: float = 0.5) -> None:
        import ray

        self._node_id = str(ray.get_runtime_context().get_node_id())
        self._started_at = time.time()
        self._sampler = GpuUtilSampler(interval_s=interval_s, sample_all_visible=True)
        self._sampler.start()

    def node_id(self) -> str:
        return self._node_id

    def stop(self) -> dict[str, float]:
        stopped_at = time.time()
        stats = self._sampler.window_stats(self._started_at, stopped_at)
        diagnostics = self._sampler.diagnostics()
        self._sampler.stop()
        metrics: dict[str, float] = {
            "pipeline_hardware_wall_time_s": stopped_at - self._started_at,
            "pipeline_hardware_sampler_node_count": 1.0,
            "pipeline_hardware_sampler_active_node_count": float(diagnostics.get("gpu_sampler_active", 0.0) > 0),
            "pipeline_hardware_gpu_device_count": float(len(stats)),
            "pipeline_hardware_gpu_sampler_error_count": diagnostics.get("gpu_sampler_error_count", 0.0),
        }
        util_sum = 0.0
        mem_sum = 0.0
        for gpu_uuid, gpu_stats in sorted(stats.items()):
            safe_key = f"{self._node_id[:8]}_{gpu_uuid[:12]}"
            util = float(gpu_stats.get("gpu_util_pct", 0.0))
            mem = float(gpu_stats.get("gpu_mem_used_pct", 0.0))
            metrics[f"pipeline_hardware_gpu_util_pct_{safe_key}"] = util
            metrics[f"pipeline_hardware_gpu_mem_used_pct_{safe_key}"] = mem
            util_sum += util
            mem_sum += mem
        if stats:
            metrics["pipeline_hardware_gpu_util_pct_mean_all_sampled"] = util_sum / len(stats)
            metrics["pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"] = mem_sum / len(stats)
        return metrics


def start_pipeline_hardware_samplers(*, interval_s: float = 0.5, startup_timeout_s: float = 5.0) -> list[Any]:
    """Start one sampler actor per live Ray node, best effort."""

    import ray

    remote_cls = ray.remote(num_cpus=0)(_PipelineHardwareSamplerActor)
    pending: dict[Any, Any] = {}
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = str(node.get("NodeID", ""))
        if not node_id:
            continue
        resource_key = f"node:{node_id}"
        if resource_key not in node.get("Resources", {}):
            logger.debug("Skipping pipeline hardware sampler on node {} without resource {}", node_id, resource_key)
            continue
        resources = {resource_key: 0.001}
        try:
            actor = remote_cls.options(resources=resources).remote(interval_s)
            pending[actor.node_id.remote()] = actor
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to start pipeline hardware sampler on node {}: {}", node_id, exc)
    if not pending:
        return []

    ready_refs, pending_refs = ray.wait(list(pending), num_returns=len(pending), timeout=max(0.0, startup_timeout_s))
    actors: list[Any] = []
    for ref in ready_refs:
        actor = pending[ref]
        try:
            ray.get(ref)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler actor failed during startup: {}", exc)
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
            continue
        actors.append(actor)
    for ref in pending_refs:
        actor = pending[ref]
        logger.debug("Skipping pipeline hardware sampler actor that did not start within {}s", startup_timeout_s)
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)
    return actors


def stop_pipeline_hardware_samplers(actors: list[Any], *, stop_timeout_s: float = 10.0) -> dict[str, float]:
    """Stop sampler actors and aggregate scalar metrics."""

    if not actors:
        return {
            "pipeline_hardware_sampler_node_count": 0.0,
            "pipeline_hardware_sampler_active_node_count": 0.0,
            "pipeline_hardware_gpu_device_count": 0.0,
        }

    import ray

    metrics: dict[str, float] = {}
    pending: dict[Any, Any] = {}
    for actor in actors:
        try:
            pending[actor.stop.remote()] = actor
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to request pipeline hardware sampler stop: {}", exc)
    if not pending:
        return metrics
    ready_refs, pending_refs = ray.wait(list(pending), num_returns=len(pending), timeout=max(0.0, stop_timeout_s))
    for ref in pending_refs:
        logger.debug("Killing pipeline hardware sampler actor that did not stop within {}s", stop_timeout_s)
        with contextlib.suppress(Exception):
            ray.kill(pending[ref], no_restart=True)
    for ref in ready_refs:
        try:
            result = ray.get(ref)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pipeline hardware sampler stop failed: {}", exc)
            continue
        for key, value in result.items():
            if key == "pipeline_hardware_wall_time_s":
                metrics[key] = max(metrics.get(key, 0.0), float(value))
                continue
            if key.startswith(("pipeline_hardware_gpu_util_pct_", "pipeline_hardware_gpu_mem_used_pct_")):
                metrics[key] = float(value)
                continue
            metrics[key] = metrics.get(key, 0.0) + float(value)

    device_count = metrics.get("pipeline_hardware_gpu_device_count", 0.0)
    if device_count:
        # Node means were summed above; normalize them to a run-wide sampled-device mean.
        util_keys = [
            key
            for key in metrics
            if key.startswith("pipeline_hardware_gpu_util_pct_")
            and key != "pipeline_hardware_gpu_util_pct_mean_all_sampled"
        ]
        mem_keys = [
            key
            for key in metrics
            if key.startswith("pipeline_hardware_gpu_mem_used_pct_")
            and key != "pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"
        ]
        if util_keys:
            metrics["pipeline_hardware_gpu_util_pct_mean_all_sampled"] = sum(metrics[key] for key in util_keys) / len(
                util_keys
            )
        if mem_keys:
            metrics["pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"] = sum(
                metrics[key] for key in mem_keys
            ) / len(mem_keys)
    return metrics
