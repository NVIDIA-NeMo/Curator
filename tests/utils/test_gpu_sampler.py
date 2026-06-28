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

from nemo_curator.utils.gpu_sampler import GpuUtilSampler, norm_uuid


def test_norm_uuid_is_public_normalizer() -> None:
    assert norm_uuid("GPU-ABCDEF") == "abcdef"
    assert norm_uuid(b"GPU-1234") == "1234"


def test_gpu_sampler_reports_inactive_diagnostics_without_nvml() -> None:
    sampler = GpuUtilSampler(gpu_uuids=("GPU-abc",))

    sampler.start()
    diagnostics = sampler.diagnostics()

    assert diagnostics["gpu_sampler_active"] == 0.0
    assert diagnostics["gpu_sampler_target_uuid_count"] == 1.0
    assert diagnostics["gpu_sampler_handle_count"] == 0.0
    assert diagnostics["gpu_sampler_sample_all_visible"] == 1.0
