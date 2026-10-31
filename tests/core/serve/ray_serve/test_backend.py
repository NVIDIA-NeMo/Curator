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

from unittest import mock

import pytest

from nemo_curator.core.serve import InferenceServer, RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed")


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

        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.deployment_config == {"autoscaling_config": {"min_replicas": 1}}
        assert result.engine_kwargs == {"tensor_parallel_size": 4}
        assert result.runtime_env["pip"] == ["my-package"]
        assert result.runtime_env["env_vars"]["MY_VAR"] == "1"
        assert result.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert result.runtime_env["env_vars"]["RAY_SERVE_LOG_TO_STDERR"] == "0"

    def test_deploy_submits_without_waiting_and_shares_startup_deadline(self) -> None:
        server = InferenceServer(models=[RayServeModelConfig(model_identifier="model")], verbose=True)
        backend = RayServeBackend(server)

        with (
            mock.patch("ray.serve.start"),
            mock.patch("ray.serve.run_many") as run_many,
            mock.patch("ray.serve.RunTarget", side_effect=lambda **kwargs: kwargs),
            mock.patch("ray.serve.llm.build_openai_app", return_value="app"),
            mock.patch("nemo_curator.core.serve.ray_serve.backend.get_free_port", return_value=9000),
            mock.patch("nemo_curator.core.serve.ray_serve.backend.time.monotonic", return_value=100.0),
            mock.patch.object(backend, "_to_llm_config", return_value="config"),
            mock.patch.object(server, "_wait_for_healthy") as wait_for_healthy,
        ):
            backend._deploy()

        run_many.assert_called_once_with(
            [{"target": "app", "name": server.name, "logging_config": None}],
            blocking=False,
            wait_for_ingress_deployment_creation=False,
            wait_for_applications_running=False,
        )
        wait_for_healthy.assert_called_once_with(deadline=100.0 + server.health_check_timeout_s)
