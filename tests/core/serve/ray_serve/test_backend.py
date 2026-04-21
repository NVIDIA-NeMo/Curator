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

import pytest

from nemo_curator.core.serve import RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

LLMConfig = pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed").LLMConfig


class TestRayServeBackend:
    def test_to_llm_config_reads_typed_model_config(self) -> None:
        model = RayServeModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
            runtime_env={
                "pip": ["my-package"],
                "env_vars": {"MY_VAR": "1", "VLLM_LOGGING_LEVEL": "DEBUG"},
            },
        )

        quiet_env = RayServeBackend._quiet_runtime_env()
        result = RayServeBackend._to_llm_config(model, quiet_runtime_env=quiet_env)

        assert isinstance(result, LLMConfig)
        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.deployment_config == {"autoscaling_config": {"min_replicas": 1}}
        assert result.engine_kwargs == {"tensor_parallel_size": 4}
        assert result.runtime_env["pip"] == ["my-package"]
        assert result.runtime_env["env_vars"]["MY_VAR"] == "1"
        assert result.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert result.runtime_env["env_vars"]["RAY_SERVE_LOG_TO_STDERR"] == "0"
