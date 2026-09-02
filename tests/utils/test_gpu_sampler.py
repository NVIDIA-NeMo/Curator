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

from types import SimpleNamespace

import pytest

from nemo_curator.utils import gpu_sampler
from nemo_curator.utils.gpu_sampler import (
    GpuUtilSampler,
    aggregate_pipeline_hardware_metrics,
    norm_uuid,
    pipeline_node_hardware_metrics,
)


class _OneSampleStop:
    def __init__(self) -> None:
        self._checks = 0

    def is_set(self) -> bool:
        self._checks += 1
        return self._checks > 1

    def wait(self, _interval_s: float) -> None:
        return None


def test_norm_uuid_is_public_normalizer() -> None:
    assert norm_uuid("GPU-ABCDEF") == "abcdef"
    assert norm_uuid(b"GPU-1234") == "1234"


def test_gpu_sampler_reports_inactive_diagnostics_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = GpuUtilSampler(gpu_uuids=("GPU-abc",))

    def _raise_unavailable() -> None:
        msg = "NVML unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(sampler, "_resolve_handles", _raise_unavailable)

    sampler.start()
    diagnostics = sampler.diagnostics()

    assert diagnostics["gpu_sampler_active"] == 0.0
    assert diagnostics["gpu_sampler_target_uuid_count"] == 1.0
    assert diagnostics["gpu_sampler_handle_count"] == 0.0
    assert diagnostics["gpu_sampler_sample_all_visible"] == 1.0


def test_gpu_sampler_uses_stage_window_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    wall_clock_s = 1_785_323_000.0
    monkeypatch.setattr(gpu_sampler.time, "time", lambda: wall_clock_s)
    monkeypatch.setattr(gpu_sampler.time, "perf_counter", lambda: 123.0)
    sampler = GpuUtilSampler(gpu_uuids=("GPU-abc",), sample_all_visible=False)
    sampler._handles = [object()]
    sampler._handle_keys = ["abc"]
    sampler._pynvml = SimpleNamespace(
        nvmlDeviceGetUtilizationRates=lambda _handle: SimpleNamespace(gpu=75),
        nvmlDeviceGetMemoryInfo=lambda _handle: SimpleNamespace(used=50, total=100),
    )
    sampler._stop = _OneSampleStop()

    sampler._loop()

    assert sampler.window_stats(wall_clock_s - 1.0, wall_clock_s + 1.0) == {
        "abc": {
            "gpu_util_pct": 75.0,
            "gpu_mem_used_pct": 50.0,
        }
    }


def test_pipeline_node_hardware_metrics_formats_one_node() -> None:
    metrics = pipeline_node_hardware_metrics(
        node_id="node-abcdef",
        wall_time_s=12.5,
        aggregate_stats={
            "GPU-aaaa1111bbbb2222": {
                "gpu_util_pct": 80.0,
                "gpu_mem_used_pct": 60.0,
                "gpu_sample_count": 10.0,
                "gpu_read_error_count": 1.0,
                "gpu_util_pct_min": 40.0,
                "gpu_util_pct_max": 90.0,
                "gpu_mem_used_pct_min": 30.0,
                "gpu_mem_used_pct_max": 70.0,
            }
        },
        diagnostics={"gpu_sampler_active": 1.0, "gpu_sampler_error_count": 1.0},
    )

    assert metrics["pipeline_hardware_wall_time_s"] == 12.5
    assert metrics["pipeline_hardware_sampler_node_count"] == 1.0
    assert metrics["pipeline_hardware_sampler_active_node_count"] == 1.0
    assert metrics["pipeline_hardware_gpu_device_count"] == 1.0
    assert metrics["pipeline_hardware_gpu_sampler_error_count"] == 1.0
    assert metrics["pipeline_hardware_gpu_util_pct_node-abc_aaaa1111bbbb"] == 80.0
    assert metrics["pipeline_hardware_gpu_mem_used_pct_node-abc_aaaa1111bbbb"] == 60.0
    assert metrics["pipeline_hardware_gpu_util_pct_mean_all_sampled"] == 80.0
    assert metrics["pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"] == 60.0


def test_aggregate_pipeline_hardware_metrics_preserves_gpu_metrics_and_recomputes_means() -> None:
    metrics = aggregate_pipeline_hardware_metrics(
        [
            {
                "pipeline_hardware_wall_time_s": 10.0,
                "pipeline_hardware_sampler_node_count": 1.0,
                "pipeline_hardware_gpu_device_count": 1.0,
                "pipeline_hardware_gpu_util_pct_node1_gpu1": 90.0,
                "pipeline_hardware_gpu_mem_used_pct_node1_gpu1": 70.0,
                "pipeline_hardware_gpu_util_pct_mean_all_sampled": 90.0,
            },
            {
                "pipeline_hardware_wall_time_s": 15.0,
                "pipeline_hardware_sampler_node_count": 1.0,
                "pipeline_hardware_gpu_device_count": 1.0,
                "pipeline_hardware_gpu_util_pct_node2_gpu1": 30.0,
                "pipeline_hardware_gpu_mem_used_pct_node2_gpu1": 50.0,
                "pipeline_hardware_gpu_util_pct_mean_all_sampled": 30.0,
            },
        ]
    )

    assert metrics["pipeline_hardware_wall_time_s"] == 15.0
    assert metrics["pipeline_hardware_sampler_node_count"] == 2.0
    assert metrics["pipeline_hardware_gpu_device_count"] == 2.0
    assert metrics["pipeline_hardware_gpu_util_pct_node1_gpu1"] == 90.0
    assert metrics["pipeline_hardware_gpu_util_pct_node2_gpu1"] == 30.0
    assert metrics["pipeline_hardware_gpu_util_pct_mean_all_sampled"] == 60.0
    assert metrics["pipeline_hardware_gpu_mem_used_pct_mean_all_sampled"] == 60.0
