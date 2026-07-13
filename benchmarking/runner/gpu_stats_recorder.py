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

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar, Self

from loguru import logger

if TYPE_CHECKING:
    from types import ModuleType


class GPUStatsRecorder:
    """Background thread that polls all GPUs via gpustat and appends rows to a CSV.

    Polls every ``interval_s`` seconds and writes one row per (timestamp, GPU)
    pair to ``output_path``. Designed to wrap a benchmark subprocess:

        with GPUStatsRecorder(session_entry_path / "gpustats.csv", interval_s=1.0):
            run_benchmark_subprocess(...)

    The recorder polls all visible GPUs via NVML, independent of
    ``CUDA_VISIBLE_DEVICES`` — useful for verifying that workloads honor the
    visible-device mask.

    Failures during a single poll iteration are logged at WARNING and the
    thread keeps running; polling is best-effort and must not crash the
    benchmark.
    """

    HEADER: ClassVar[list[str]] = [
        "timestamp_utc",
        "gpu_id",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "fan_speed_pct",
        "processes",
    ]

    def __init__(self, output_path: Path, interval_s: float = 1.0) -> None:
        self.output_path = Path(output_path)
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._csv_file: IO[str] | None = None
        self._csv_writer: csv._writer | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            msg = "GPUStatsRecorder already started"
            raise RuntimeError(msg)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.output_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.HEADER)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="gpu-stats-recorder",
            daemon=True,
        )
        self._thread.start()
        logger.debug(f"GPUStatsRecorder started: {self.output_path} (interval={self.interval_s}s)")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        self._thread = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        logger.debug(f"GPUStatsRecorder stopped: {self.output_path}")

    def _poll_loop(self) -> None:
        # Import lazily so importing this module never requires gpustat at module load time.
        import gpustat

        while not self._stop_event.is_set():
            try:
                self._poll_once(gpustat)
            except Exception as e:
                # Best-effort poller — must not crash the benchmark.
                logger.warning(f"GPUStatsRecorder poll failed: {e}")
            # Wait returns True if the event was set during the wait, allowing prompt shutdown.
            self._stop_event.wait(self.interval_s)

    def _poll_once(self, gpustat_mod: ModuleType) -> None:
        query = gpustat_mod.new_query()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for gpu in query:
            mem_total = gpu.memory_total or 0
            mem_pct = (gpu.memory_used / mem_total * 100.0) if mem_total else 0.0
            procs = [{k: p.get(k) for k in ("pid", "username", "command", "gpu_memory_usage")} for p in gpu.processes]
            # power_draw / power_limit / fan_speed are part of the same NVML query
            # gpustat already issued — reading them is free. Datacenter SKUs often
            # return None for fan_speed (no controllable per-card fan); render
            # None as empty string for CSV cleanliness.
            self._csv_writer.writerow(
                [
                    ts,
                    gpu.index,
                    gpu.utilization,
                    round(mem_pct, 2),
                    gpu.temperature,
                    "" if gpu.power_draw is None else round(gpu.power_draw, 1),
                    "" if gpu.power_limit is None else int(gpu.power_limit),
                    "" if gpu.fan_speed is None else int(gpu.fan_speed),
                    json.dumps(procs, separators=(",", ":")),
                ]
            )
        if self._csv_file is not None:
            self._csv_file.flush()
