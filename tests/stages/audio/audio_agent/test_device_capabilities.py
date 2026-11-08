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

"""Malformed CPU constraints cannot certify an adapter for CPU execution."""

from types import SimpleNamespace

import pytest

from nemo_curator.stages.audio.audio_agent._device_capabilities import supported_devices


@pytest.mark.parametrize("requirements", [None, [], {"precision": "fp32"}, {"precision": {"value": "fp32"}}])
def test_malformed_cpu_constraints_preserve_only_declared_gpu_support(requirements: object) -> None:
    stage = SimpleNamespace(adapter_target="package.Adapter", adapter_kwargs={"precision": "fp32"})
    card = {
        "adapter_capabilities": {
            "package.Adapter": {
                "supported_devices": ["cpu", "cuda"],
                "cpu_requires": requirements,
            }
        }
    }

    assert supported_devices(stage, card) == {"cuda"}
