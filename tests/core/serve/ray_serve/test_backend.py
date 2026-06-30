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

import contextlib
import sys
import types

import pytest

from nemo_curator.core.serve import RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

LLMConfig = pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed").LLMConfig


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:
    """Override the repository-wide Ray fixture for these pure unit tests."""


@pytest.fixture
def fake_ray_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, object]], list[str]]:
    ray_init_calls: list[dict[str, object]] = []
    serve_calls: list[str] = []

    def fail_private_client_reset(_: object) -> None:
        msg = "Ray 2.56 should not need Curator to reset ray.serve.context._global_client"
        raise AssertionError(msg)

    def fake_ray_init(**kwargs: object) -> contextlib.AbstractContextManager[None]:
        ray_init_calls.append(kwargs)
        return contextlib.nullcontext()

    ray_module = types.ModuleType("ray")
    ray_module.init = fake_ray_init  # type: ignore[attr-defined]

    serve_module = types.ModuleType("ray.serve")
    serve_module.shutdown = lambda: serve_calls.append("shutdown")  # type: ignore[attr-defined]
    ray_module.serve = serve_module  # type: ignore[attr-defined]

    context_module = types.ModuleType("ray.serve.context")
    context_module._set_global_client = fail_private_client_reset  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.serve", serve_module)
    monkeypatch.setitem(sys.modules, "ray.serve.context", context_module)

    return ray_init_calls, serve_calls


class TestRayServeBackend:
    def test_start_uses_public_ray_lifecycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_ray_modules: tuple[list[dict[str, object]], list[str]],
    ) -> None:
        ray_init_calls, _ = fake_ray_modules
        deploy_calls: list[str] = []
        backend = RayServeBackend(server=object())  # type: ignore[arg-type]

        monkeypatch.setattr(backend, "_deploy", lambda: deploy_calls.append("deploy"))

        backend.start()

        assert ray_init_calls == [{"ignore_reinit_error": True}]
        assert deploy_calls == ["deploy"]

    def test_stop_uses_public_ray_lifecycle(
        self,
        fake_ray_modules: tuple[list[dict[str, object]], list[str]],
    ) -> None:
        ray_init_calls, serve_calls = fake_ray_modules
        backend = RayServeBackend(server=object())  # type: ignore[arg-type]

        backend.stop()

        assert ray_init_calls == [{"ignore_reinit_error": True}]
        assert serve_calls == ["shutdown"]

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
