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

"""Audio worker resource observations and calibration boundaries."""

from nemo_curator.stages.audio.audio_agent import calibration
from nemo_curator.stages.audio.utils import resource_probe_metrics


def test_runtime_resource_probe_keeps_worker_lifetime_memory_out_of_calibration() -> None:
    metrics = resource_probe_metrics(
        gpu_probe_started=False,
        process_time=2.0,
        num_items=5,
    )

    assert metrics["throughput"] == 2.5
    assert metrics["worker_lifetime_peak_host_mem_gb"] > 0
    assert "peak_host_mem_gb" not in metrics
    assert "peak_vram_gb" not in metrics
    measured = calibration.from_smoke(
        {
            "per_stage_metrics": {
                "scorer": {
                    "custom.worker_lifetime_peak_host_mem_gb": {"max": 8.0},
                    "custom.throughput": {"mean": 2.5},
                }
            }
        }
    )

    assert measured["scorer"] == {"throughput": 2.5, "source": "measured"}
