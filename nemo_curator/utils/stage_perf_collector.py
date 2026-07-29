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

"""Run-scoped transport for authoritative extended stage telemetry."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.utils.performance_utils import StagePerfStats

_COLLECTOR_NAME_ATTR = "_curator_stage_perf_collector_name"


@dataclass
class CollectedStagePerf:
    """One invocation record plus whether it also travelled on output tasks."""

    perf_stats: StagePerfStats
    attached_to_output: bool


@ray.remote(num_cpus=0)
class _StagePerfCollector:
    """Small run-scoped actor retaining opt-in invocation records."""

    def __init__(self) -> None:
        self._records: list[CollectedStagePerf] = []

    def ready(self) -> bool:
        return True

    def record(self, perf_stats: StagePerfStats, attached_to_output: bool) -> None:
        self._records.append(
            CollectedStagePerf(
                perf_stats=perf_stats,
                attached_to_output=attached_to_output,
            )
        )

    def drain(self) -> list[CollectedStagePerf]:
        records = self._records
        self._records = []
        return records


def start_stage_perf_collector(stages: list[ProcessingStage]) -> Any | None:  # noqa: ANN401
    """Start one collector when at least one stage enables extended metrics."""
    if not any(bool(getattr(stage, "extended_performance_metrics", False)) for stage in stages):
        return None
    name = f"curator-stage-perf-{uuid.uuid4().hex}"
    collector = _StagePerfCollector.options(name=name).remote()
    ray.get(collector.ready.remote())
    for stage in stages:
        setattr(stage, _COLLECTOR_NAME_ATTR, name)
    return collector


def record_stage_perf(
    stage: ProcessingStage,
    perf_stats: StagePerfStats,
    *,
    attached_to_output: bool,
) -> bool:
    """Synchronously publish one record so driver drain cannot overtake it."""
    collector_name = str(getattr(stage, _COLLECTOR_NAME_ATTR, "") or "")
    if not collector_name:
        return False
    try:
        collector = ray.get_actor(collector_name)
        ray.get(collector.record.remote(perf_stats, attached_to_output))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stage performance collector publish failed for {}: {}", stage.name, exc)
        return False
    return True


def stop_stage_perf_collector(
    collector: Any | None,  # noqa: ANN401
    stages: list[ProcessingStage],
) -> list[CollectedStagePerf]:
    """Drain and remove the collector, clearing its run-scoped stage routing."""
    if collector is None:
        return []
    try:
        return list(ray.get(collector.drain.remote()))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stage performance collector drain failed: {}", exc)
        return []
    finally:
        for stage in stages:
            if hasattr(stage, _COLLECTOR_NAME_ATTR):
                delattr(stage, _COLLECTOR_NAME_ATTR)
        try:
            ray.kill(collector, no_restart=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stage performance collector cleanup failed: {}", exc)
