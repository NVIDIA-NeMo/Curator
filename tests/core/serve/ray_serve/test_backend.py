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

import time

import pytest

from nemo_curator.core.serve import InferenceServer, RayServeModelConfig, is_inference_server_active
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


@pytest.mark.parametrize("stalled", [False, True])
def test_deployment_readiness_is_bounded(
    shared_ray_client: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    stalled: bool,
) -> None:
    from ray import serve

    @serve.deployment(ray_actor_options={"num_cpus": 1000 if stalled else 0}, graceful_shutdown_timeout_s=0.1)
    class Endpoint:
        def __call__(self) -> dict:
            return {"data": []}

    # Replace the GPU model with a real CPU Serve replica; keep deployment,
    # HTTP readiness, timeout, and cleanup on the real Ray integration path.
    monkeypatch.setattr("ray.serve.llm.build_openai_app", lambda _: Endpoint.bind())
    server = InferenceServer(models=[], health_check_timeout_s=5 if stalled else 60)
    started = time.monotonic()
    try:
        if stalled:
            with pytest.raises(TimeoutError, match="within 5s"):
                server.start()
            assert time.monotonic() - started < 60
            assert not is_inference_server_active()
            assert not serve.status().applications
        else:
            server.start()
            assert is_inference_server_active()
    finally:
        server.stop()
