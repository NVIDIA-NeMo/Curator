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
percent on the GPUs this worker owns (matched by ``gpu_uuids``, so it samples
only its own devices). Per physical GPU: ``window_stats`` returns a
``{normalized_uuid: {gpu_util_pct, gpu_mem_used_pct}}`` mean over ``[t0, t1]``.

No-op when ``pynvml`` is unavailable, no UUIDs match, or NVML raises (fields
simply omitted). ``gpu`` here is NVML SM duty-cycle percent, not FLOP efficiency.
"""

from __future__ import annotations

import threading
import time
from collections import deque


def _norm_uuid(value: object) -> str:
    """Normalize a GPU UUID for comparison (drop ``GPU-`` prefix, lowercase)."""
    text = value.decode() if isinstance(value, bytes) else str(value)
    return text.strip().lower().removeprefix("gpu-")


class GpuUtilSampler:
    """Polls NVML in a background thread; reports windowed mean util/mem."""

    def __init__(self, gpu_uuids: tuple[str, ...] = (), interval_s: float = 0.2) -> None:
        self._target_uuids = {_norm_uuid(u) for u in (gpu_uuids or ()) if str(u).strip()}
        self._interval_s = max(float(interval_s), 0.02)
        self._handles: list[object] = []
        # normalized UUID per handle -- the consumer's per-GPU attribution key.
        self._handle_keys: list[str] = []
        # (t, [util% per handle], [mem% per handle]) aligned to ``_handles``;
        # time-ordered and pruned by ``window_stats`` to stay bounded.
        self._samples: deque[tuple[float, list[float | None], list[float | None]]] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None

    def _resolve_handles(self) -> None:
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            # No target UUIDs -> cannot safely attribute devices to this worker.
            key = _norm_uuid(pynvml.nvmlDeviceGetUUID(handle))
            if self._target_uuids and key in self._target_uuids:
                self._handles.append(handle)
                self._handle_keys.append(key)

    def start(self) -> None:
        try:
            self._resolve_handles()
        except Exception:  # noqa: BLE001
            self._handles = []
        if not self._handles:
            return
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
            for k, handle in enumerate(self._handles):
                try:
                    utils[k] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mems[k] = 100.0 * float(mem.used) / float(mem.total) if mem.total else 0.0
                except Exception:  # noqa: BLE001
                    continue
            with self._lock:
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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
