# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Observational resource metrics for audio pipeline execution."""

from __future__ import annotations

import sys
from typing import Any


def begin_resource_probe(stage: Any) -> bool:  # noqa: ANN401
    """Reset CUDA peak accounting for a GPU-reserving stage when available.

    Returns whether a matching VRAM reading can be collected after processing.
    Every failure is observational only; resource telemetry must never change
    whether a stage can execute.
    """
    try:
        resources = getattr(stage, "resources", None)
        if float(getattr(resources, "gpus", 0.0) or 0.0) <= 0:
            return False
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.reset_peak_memory_stats()
        return True  # noqa: TRY300 - trivial guard, no else needed
    except Exception:  # noqa: BLE001 - metrics are best-effort
        return False


def resource_probe_metrics(
    *,
    gpu_probe_started: bool,
    process_time: float,
    num_items: int,
) -> dict[str, float]:
    """Best-effort worker high-water RSS, CUDA peak, and throughput metrics."""
    metrics: dict[str, float] = {}
    try:
        import resource

        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. This process-lifetime diagnostic
        # must remain distinct from the per-stage keys consumed by calibration.
        peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
        if peak_rss_bytes >= 0:
            metrics["worker_lifetime_peak_host_mem_gb"] = peak_rss_bytes / (1024**3)
    except Exception:  # noqa: BLE001, S110 - unsupported platform/worker
        pass
    if gpu_probe_started:
        try:
            import torch

            metrics["peak_vram_gb"] = float(torch.cuda.max_memory_allocated()) / (1024**3)
        except Exception:  # noqa: BLE001, S110 - telemetry cannot fail execution
            pass
    if process_time > 0 and num_items >= 0:
        metrics["throughput"] = float(num_items) / process_time
    return metrics
