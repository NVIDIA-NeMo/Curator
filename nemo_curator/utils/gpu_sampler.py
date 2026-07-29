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

"""Fail-open NVML sampling for stage windows and pipeline aggregates."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass

from loguru import logger

from nemo_curator.utils.performance_utils import norm_gpu_uuid

norm_uuid = norm_gpu_uuid
_WALL_TIME = "pipeline_hardware_wall_time_s"
_UTIL_MEAN = "pipeline_hardware_gpu_util_pct_mean_all_sampled"
_MEM_MEAN = "pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"
_SUM_KEYS = {
    "pipeline_hardware_sampler_node_count",
    "pipeline_hardware_sampler_active_node_count",
    "pipeline_hardware_gpu_device_count",
    "pipeline_hardware_gpu_sampler_error_count",
}
_GPU_METRICS = {
    "util_pct": "gpu_util_pct",
    "mem_used_pct": "gpu_mem_used_pct",
    "sample_count": "gpu_sample_count",
    "util_min_pct": "gpu_util_pct_min",
    "util_max_pct": "gpu_util_pct_max",
    "mem_used_min_pct": "gpu_mem_used_pct_min",
    "mem_used_max_pct": "gpu_mem_used_pct_max",
    "read_error_count": "gpu_read_error_count",
}


@dataclass
class _Aggregate:
    samples: int = 0
    errors: int = 0
    util_n: int = 0
    util_sum: float = 0.0
    util_min: float | None = None
    util_max: float | None = None
    mem_n: int = 0
    mem_sum: float = 0.0
    mem_min: float | None = None
    mem_max: float | None = None

    def add(self, util: float | None, mem: float | None, error: bool) -> None:
        self.samples += 1
        self.errors += int(error)
        if util is not None:
            self.util_n += 1
            self.util_sum += util
            self.util_min = util if self.util_min is None else min(self.util_min, util)
            self.util_max = util if self.util_max is None else max(self.util_max, util)
        if mem is not None:
            self.mem_n += 1
            self.mem_sum += mem
            self.mem_min = mem if self.mem_min is None else min(self.mem_min, mem)
            self.mem_max = mem if self.mem_max is None else max(self.mem_max, mem)

    def snapshot(self) -> dict[str, float]:
        if not self.util_n:
            return {}
        metrics = {
            "gpu_util_pct": self.util_sum / self.util_n,
            "gpu_mem_used_pct": self.mem_sum / self.mem_n if self.mem_n else 0.0,
            "gpu_sample_count": float(self.samples),
            "gpu_util_sample_count": float(self.util_n),
            "gpu_mem_sample_count": float(self.mem_n),
            "gpu_read_error_count": float(self.errors),
            "gpu_util_pct_min": float(self.util_min or 0.0),
            "gpu_util_pct_max": float(self.util_max or 0.0),
        }
        if self.mem_min is not None and self.mem_max is not None:
            metrics.update(gpu_mem_used_pct_min=self.mem_min, gpu_mem_used_pct_max=self.mem_max)
        return metrics


def actor_gpu_window_metrics(
    window_stats: dict[str, dict[str, float]],
    diagnostics: dict[str, float] | None = None,
) -> dict[str, float]:
    metrics = dict(diagnostics or {})
    metrics.update(
        (f"{name}::{gpu_uuid}", float(value))
        for gpu_uuid, gpu_metrics in window_stats.items()
        for name, value in gpu_metrics.items()
    )
    return metrics


def _set_means(metrics: dict[str, float]) -> None:
    for prefix, output in (
        ("pipeline_hardware_gpu_util_pct_", _UTIL_MEAN),
        ("pipeline_hardware_gpu_mem_used_pct_", _MEM_MEAN),
    ):
        values = [value for key, value in metrics.items() if key.startswith(prefix) and key != output]
        if values:
            metrics[output] = sum(values) / len(values)


def pipeline_node_hardware_metrics(
    *,
    node_id: str,
    wall_time_s: float,
    aggregate_stats: dict[str, dict[str, float]],
    diagnostics: dict[str, float],
) -> dict[str, float]:
    metrics = {
        _WALL_TIME: float(wall_time_s),
        "pipeline_hardware_sampler_node_count": 1.0,
        "pipeline_hardware_sampler_active_node_count": float(diagnostics.get("gpu_sampler_active", 0.0) > 0),
        "pipeline_hardware_gpu_device_count": float(len(aggregate_stats)),
        "pipeline_hardware_gpu_sampler_error_count": diagnostics.get("gpu_sampler_error_count", 0.0),
    }
    for gpu_uuid, stats in sorted(aggregate_stats.items()):
        key = f"{node_id[:8]}_{norm_uuid(gpu_uuid)[:12]}"
        metrics.update(
            {
                f"pipeline_hardware_gpu_{output}_{key}": float(stats.get(source, 0.0))
                for output, source in _GPU_METRICS.items()
            }
        )
    _set_means(metrics)
    return metrics


def aggregate_pipeline_hardware_metrics(node_results: list[dict[str, float]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for result in node_results:
        for key, value in result.items():
            metric_value = float(value)
            if key == _WALL_TIME:
                metrics[key] = max(metrics.get(key, 0.0), metric_value)
            elif key in (_UTIL_MEAN, _MEM_MEAN):
                continue
            elif key in _SUM_KEYS:
                metrics[key] = metrics.get(key, 0.0) + metric_value
            else:
                metrics[key] = metric_value
    _set_means(metrics)
    return metrics


class GpuUtilSampler:
    """Poll NVML into windowed samples or constant-memory aggregates."""

    def __init__(
        self,
        gpu_uuids: tuple[str, ...] = (),
        interval_s: float = 0.2,
        *,
        sample_all_visible: bool = True,
        aggregate_only: bool = False,
    ) -> None:
        self._target_uuids = {norm_uuid(uuid) for uuid in gpu_uuids if str(uuid).strip()}
        self._sample_all_visible = sample_all_visible
        self._aggregate_only = aggregate_only
        self._interval_s = max(float(interval_s), 0.02)
        self._handles: list[object] = []
        self._handle_keys: list[str] = []
        self._samples: deque[tuple[float, list[float | None], list[float | None]]] = deque()
        self._aggregates: list[_Aggregate] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._read_error_count = 0

    def _resolve_handles(self) -> None:
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            key = norm_uuid(pynvml.nvmlDeviceGetUUID(handle))
            if self._sample_all_visible or key in self._target_uuids:
                self._handles.append(handle)
                self._handle_keys.append(key)

    def start(self) -> None:
        try:
            self._resolve_handles()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU sampler disabled: NVML handle resolution failed: {}", exc)
            self._handles = []
        if not self._handles:
            logger.debug("GPU sampler disabled: no matching NVML devices")
            return
        if self._aggregate_only:
            self._aggregates = [_Aggregate() for _ in self._handles]
        self._thread = threading.Thread(target=self._loop, name="gpu-util-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            utils: list[float | None] = []
            mems: list[float | None] = []
            errors: list[bool] = []
            for index, handle in enumerate(self._handles):
                try:
                    util = float(self._pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem = 100.0 * float(memory.used) / float(memory.total) if memory.total else 0.0
                    error = False
                except Exception as exc:  # noqa: BLE001
                    util = mem = None
                    error = True
                    self._read_error_count += 1
                    if self._read_error_count == 1 or self._read_error_count % 100 == 0:
                        logger.debug("GPU sampler NVML read failed for handle {}: {}", index, exc)
                utils.append(util)
                mems.append(mem)
                errors.append(error)
            with self._lock:
                if self._aggregate_only:
                    for aggregate, util, mem, error in zip(self._aggregates, utils, mems, errors, strict=True):
                        aggregate.add(util, mem, error)
                else:
                    self._samples.append((time.time(), utils, mems))
            self._stop.wait(self._interval_s)

    def window_stats(self, t0: float, t1: float) -> dict[str, dict[str, float]]:
        totals = [[0.0, 0, 0.0, 0] for _ in self._handles]
        with self._lock:
            while self._samples and self._samples[0][0] < t0:
                self._samples.popleft()
            for timestamp, utils, mems in self._samples:
                if timestamp > t1:
                    break
                for total, util, mem in zip(totals, utils, mems, strict=True):
                    if util is not None:
                        total[0] += util
                        total[1] += 1
                    if mem is not None:
                        total[2] += mem
                        total[3] += 1
        return {
            key: {
                "gpu_util_pct": total[0] / total[1],
                "gpu_mem_used_pct": total[2] / total[3] if total[3] else 0.0,
            }
            for key, total in zip(self._handle_keys, totals, strict=True)
            if total[1]
        }

    def window_metrics(self, t0: float, t1: float) -> dict[str, float]:
        return actor_gpu_window_metrics(self.window_stats(t0, t1), self.diagnostics())

    def aggregate_stats(self) -> dict[str, dict[str, float]]:
        if not self._aggregate_only:
            return {}
        with self._lock:
            snapshots = [aggregate.snapshot() for aggregate in self._aggregates]
        return {key: value for key, value in zip(self._handle_keys, snapshots, strict=True) if value}

    def diagnostics(self) -> dict[str, float]:
        return {
            "gpu_sampler_active": float(self._thread is not None and bool(self._handles)),
            "gpu_sampler_handle_count": float(len(self._handles)),
            "gpu_sampler_target_uuid_count": float(len(self._target_uuids)),
            "gpu_sampler_sample_all_visible": float(self._sample_all_visible),
            "gpu_sampler_error_count": float(self._read_error_count),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
