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

"""Background NVML GPU-utilization sampler (adapter-agnostic).

A worker-local daemon thread polls NVML for SM utilization and memory-used
percent. By default it samples all NVML-visible devices on the node so hardware
metrics are independent of which processor/actor owns a GPU. Callers may pass
``gpu_uuids`` with ``sample_all_visible=False`` when they need actor-local
attribution only. Per physical GPU: ``window_stats`` returns a
``{normalized_uuid: {gpu_util_pct, gpu_mem_used_pct}}`` mean over ``[t0, t1]``.
Pipeline-wide callers can instead enable ``aggregate_only`` and read
``aggregate_stats`` without retaining timestamped samples.

No-op when ``pynvml`` is unavailable, no UUIDs match, or NVML raises (fields
simply omitted). ``gpu`` here is NVML SM duty-cycle percent, not FLOP efficiency.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from loguru import logger

PIPELINE_HARDWARE_WALL_TIME_KEY = "pipeline_hardware_wall_time_s"
PIPELINE_HARDWARE_UTIL_MEAN_KEY = "pipeline_hardware_gpu_util_pct_mean_all_sampled"
PIPELINE_HARDWARE_MEM_MEAN_KEY = "pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"

_PIPELINE_PER_GPU_METRIC_PREFIXES = (
    "pipeline_hardware_gpu_util_pct_",
    "pipeline_hardware_gpu_mem_used_pct_",
    "pipeline_hardware_gpu_sample_count_",
    "pipeline_hardware_gpu_util_min_pct_",
    "pipeline_hardware_gpu_util_max_pct_",
    "pipeline_hardware_gpu_mem_used_min_pct_",
    "pipeline_hardware_gpu_mem_used_max_pct_",
    "pipeline_hardware_gpu_read_error_count_",
)
_PIPELINE_MEAN_KEYS = {
    PIPELINE_HARDWARE_UTIL_MEAN_KEY,
    PIPELINE_HARDWARE_MEM_MEAN_KEY,
}
_PIPELINE_GPU_AGGREGATE_METRICS = (
    ("util_pct", "gpu_util_pct"),
    ("mem_used_pct", "gpu_mem_used_pct"),
    ("sample_count", "gpu_sample_count"),
    ("util_min_pct", "gpu_util_pct_min"),
    ("util_max_pct", "gpu_util_pct_max"),
    ("mem_used_min_pct", "gpu_mem_used_pct_min"),
    ("mem_used_max_pct", "gpu_mem_used_pct_max"),
    ("read_error_count", "gpu_read_error_count"),
)


@dataclass
class _GpuStreamingAggregate:
    sample_count: int = 0
    util_count: int = 0
    util_sum: float = 0.0
    util_min: float | None = None
    util_max: float | None = None
    mem_count: int = 0
    mem_sum: float = 0.0
    mem_min: float | None = None
    mem_max: float | None = None
    error_count: int = 0

    def add(self, util: float | None, mem: float | None, *, read_error: bool) -> None:
        self.sample_count += 1
        self.error_count += int(read_error)
        if util is not None:
            self.util_count += 1
            self.util_sum += util
            self.util_min = util if self.util_min is None else min(self.util_min, util)
            self.util_max = util if self.util_max is None else max(self.util_max, util)
        if mem is not None:
            self.mem_count += 1
            self.mem_sum += mem
            self.mem_min = mem if self.mem_min is None else min(self.mem_min, mem)
            self.mem_max = mem if self.mem_max is None else max(self.mem_max, mem)

    def snapshot(self) -> dict[str, float]:
        if not self.util_count or self.util_min is None or self.util_max is None:
            return {}
        metrics = {
            "gpu_util_pct": self.util_sum / self.util_count,
            "gpu_mem_used_pct": self.mem_sum / self.mem_count if self.mem_count else 0.0,
            "gpu_sample_count": float(self.sample_count),
            "gpu_util_sample_count": float(self.util_count),
            "gpu_mem_sample_count": float(self.mem_count),
            "gpu_read_error_count": float(self.error_count),
            "gpu_util_pct_min": self.util_min,
            "gpu_util_pct_max": self.util_max,
        }
        if self.mem_min is not None and self.mem_max is not None:
            metrics["gpu_mem_used_pct_min"] = self.mem_min
            metrics["gpu_mem_used_pct_max"] = self.mem_max
        return metrics


def norm_uuid(value: object) -> str:
    """Normalize a GPU UUID for comparison (drop ``GPU-`` prefix, lowercase)."""
    text = value.decode() if isinstance(value, bytes) else str(value)
    return text.strip().lower().removeprefix("gpu-")


def actor_gpu_window_metrics(
    window_stats: dict[str, dict[str, float]],
    diagnostics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Flatten one actor invocation's GPU window into StagePerf custom metrics."""

    metrics = dict(diagnostics or {})
    for uuid_key, gpu_metrics in window_stats.items():
        for metric, value in gpu_metrics.items():
            metrics[f"{metric}::{uuid_key}"] = float(value)
    return metrics


def pipeline_node_hardware_metrics(
    *,
    node_id: str,
    wall_time_s: float,
    aggregate_stats: dict[str, dict[str, float]],
    diagnostics: dict[str, float],
) -> dict[str, float]:
    """Build run-level hardware metrics for one sampler actor/node."""

    metrics: dict[str, float] = {
        PIPELINE_HARDWARE_WALL_TIME_KEY: float(wall_time_s),
        "pipeline_hardware_sampler_node_count": 1.0,
        "pipeline_hardware_sampler_active_node_count": float(diagnostics.get("gpu_sampler_active", 0.0) > 0),
        "pipeline_hardware_gpu_device_count": float(len(aggregate_stats)),
        "pipeline_hardware_gpu_sampler_error_count": diagnostics.get("gpu_sampler_error_count", 0.0),
    }
    for gpu_uuid, gpu_stats in sorted(aggregate_stats.items()):
        safe_key = _pipeline_gpu_metric_key(node_id, gpu_uuid)
        for output_name, input_name in _PIPELINE_GPU_AGGREGATE_METRICS:
            metrics[f"pipeline_hardware_gpu_{output_name}_{safe_key}"] = float(gpu_stats.get(input_name, 0.0))
    _set_pipeline_hardware_means(metrics)
    return metrics


def aggregate_pipeline_hardware_metrics(node_results: list[dict[str, float]]) -> dict[str, float]:
    """Merge per-node pipeline hardware sampler metrics into one run metric dict."""

    metrics: dict[str, float] = {}
    for result in node_results:
        for key, value in result.items():
            metric_value = float(value)
            if key == PIPELINE_HARDWARE_WALL_TIME_KEY:
                metrics[key] = max(metrics.get(key, 0.0), metric_value)
            elif key in _PIPELINE_MEAN_KEYS:
                continue
            elif _is_pipeline_per_gpu_metric(key):
                metrics[key] = metric_value
            else:
                metrics[key] = metrics.get(key, 0.0) + metric_value
    _set_pipeline_hardware_means(metrics)
    return metrics


def _pipeline_gpu_metric_key(node_id: str, gpu_uuid: str) -> str:
    return f"{node_id[:8]}_{norm_uuid(gpu_uuid)[:12]}"


def _is_pipeline_per_gpu_metric(key: str) -> bool:
    return key not in _PIPELINE_MEAN_KEYS and key.startswith(_PIPELINE_PER_GPU_METRIC_PREFIXES)


def _set_pipeline_hardware_means(metrics: dict[str, float]) -> None:
    util_values = [
        value
        for key, value in metrics.items()
        if key.startswith("pipeline_hardware_gpu_util_pct_") and key != PIPELINE_HARDWARE_UTIL_MEAN_KEY
    ]
    mem_values = [
        value
        for key, value in metrics.items()
        if key.startswith("pipeline_hardware_gpu_mem_used_pct_") and key != PIPELINE_HARDWARE_MEM_MEAN_KEY
    ]
    if util_values:
        metrics[PIPELINE_HARDWARE_UTIL_MEAN_KEY] = sum(util_values) / len(util_values)
    if mem_values:
        metrics[PIPELINE_HARDWARE_MEM_MEAN_KEY] = sum(mem_values) / len(mem_values)


class GpuUtilSampler:
    """Poll NVML and retain either invocation windows or constant-memory run aggregates."""

    def __init__(
        self,
        gpu_uuids: tuple[str, ...] = (),
        interval_s: float = 0.2,
        *,
        sample_all_visible: bool = True,
        aggregate_only: bool = False,
    ) -> None:
        self._target_uuids = {norm_uuid(u) for u in (gpu_uuids or ()) if str(u).strip()}
        self._sample_all_visible = bool(sample_all_visible)
        self._aggregate_only = bool(aggregate_only)
        self._interval_s = max(float(interval_s), 0.02)
        self._handles: list[object] = []
        # normalized UUID per handle -- the consumer's per-GPU attribution key.
        self._handle_keys: list[str] = []
        # (t, [util% per handle], [mem% per handle]) aligned to ``_handles``;
        # time-ordered and pruned by ``window_stats`` to stay bounded.
        self._samples: deque[tuple[float, list[float | None], list[float | None]]] = deque()
        self._aggregates: list[_GpuStreamingAggregate] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._read_error_count = 0

    def _resolve_handles(self) -> None:
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
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
            if self._target_uuids:
                logger.debug(
                    "GPU sampler disabled: no NVML handles matched target UUIDs {}", sorted(self._target_uuids)
                )
            else:
                logger.debug("GPU sampler disabled: no target GPU UUIDs were provided")
            return
        if self._aggregate_only:
            self._aggregates = [_GpuStreamingAggregate() for _ in self._handles]
        self._thread = threading.Thread(target=self._loop, name="gpu-util-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        pynvml = self._pynvml
        n = len(self._handles)
        while not self._stop.is_set():
            # Position-aligned to ``_handles`` (None on read error so a transient
            # failure on one GPU never shifts the others).
            utils: list[float | None] = [None] * n
            mems: list[float | None] = [None] * n
            read_errors = [False] * n
            for k, handle in enumerate(self._handles):
                try:
                    utils[k] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mems[k] = 100.0 * float(mem.used) / float(mem.total) if mem.total else 0.0
                except Exception as exc:  # noqa: BLE001
                    read_errors[k] = True
                    self._read_error_count += 1
                    if self._read_error_count == 1 or self._read_error_count % 100 == 0:
                        logger.debug("GPU sampler NVML read failed for handle {}: {}", k, exc)
                    continue
            with self._lock:
                if self._aggregate_only:
                    for k, aggregate in enumerate(self._aggregates):
                        aggregate.add(utils[k], mems[k], read_error=read_errors[k])
                else:
                    self._samples.append((time.time(), utils, mems))
            self._stop.wait(self._interval_s)

    def window_stats(self, t0: float, t1: float) -> dict[str, dict[str, float]]:
        """Per-GPU mean util/mem over ``[t0, t1]``, keyed by normalized UUID.

        Returns ``{uuid: {gpu_util_pct, gpu_mem_used_pct}}`` (empty if no samples
        landed in the window); the consumer maps each UUID back to a physical index.
        """
        n = len(self._handles)
        util_sum = [0.0] * n
        util_cnt = [0] * n
        mem_sum = [0.0] * n
        mem_cnt = [0] * n
        with self._lock:
            # Windows advance monotonically (batches run sequentially), so drop
            # anything older than ``t0`` -- never reused, keeps the deque bounded.
            while self._samples and self._samples[0][0] < t0:
                self._samples.popleft()
            for ts, utils, mems in self._samples:
                if ts > t1:
                    break  # time-ordered: no later sample falls in the window
                for k in range(n):
                    if utils[k] is not None:
                        util_sum[k] += utils[k]
                        util_cnt[k] += 1
                    if mems[k] is not None:
                        mem_sum[k] += mems[k]
                        mem_cnt[k] += 1
        result: dict[str, dict[str, float]] = {}
        for k, key in enumerate(self._handle_keys):
            if not util_cnt[k]:
                continue
            result[key] = {
                "gpu_util_pct": util_sum[k] / util_cnt[k],
                "gpu_mem_used_pct": (mem_sum[k] / mem_cnt[k]) if mem_cnt[k] else 0.0,
            }
        return result

    def window_metrics(self, t0: float, t1: float) -> dict[str, float]:
        """Return actor invocation custom metrics for ``[t0, t1]``."""
        return actor_gpu_window_metrics(self.window_stats(t0, t1), diagnostics=self.diagnostics())

    def aggregate_stats(self) -> dict[str, dict[str, float]]:
        """Return constant-memory per-GPU run aggregates.

        This is empty for the default windowed mode. Aggregate-only callers do
        not retain timestamped samples, so memory use is independent of run time.
        """
        if not self._aggregate_only:
            return {}
        with self._lock:
            snapshots = [aggregate.snapshot() for aggregate in self._aggregates]
        return {key: snapshot for key, snapshot in zip(self._handle_keys, snapshots, strict=True) if snapshot}

    def diagnostics(self) -> dict[str, float]:
        """Small scalar state so perf summaries explain missing GPU samples."""
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
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU sampler NVML shutdown failed: {}", exc)
