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
import os
import sys
import types

import pytest

from nemo_curator.core.serve import RayServeModelConfig
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:
    """Override the repository-wide Ray fixture for these pure unit tests."""


@pytest.fixture
def fake_ray_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], list[str], list[object], list[str]]:
    ray_init_calls: list[dict[str, object]] = []
    serve_calls: list[str] = []
    client_cache_resets: list[object] = []
    lifecycle_events: list[str] = []

    def fake_ray_init(**kwargs: object) -> contextlib.AbstractContextManager[None]:
        ray_init_calls.append(kwargs)
        lifecycle_events.append("ray.init")
        return contextlib.nullcontext()

    ray_module = types.ModuleType("ray")
    ray_module.init = fake_ray_init  # type: ignore[attr-defined]

    serve_module = types.ModuleType("ray.serve")

    def fake_serve_shutdown() -> None:
        serve_calls.append("shutdown")
        lifecycle_events.append("serve.shutdown")

    serve_module.shutdown = fake_serve_shutdown  # type: ignore[attr-defined]
    ray_module.serve = serve_module  # type: ignore[attr-defined]

    context_module = types.ModuleType("ray.serve.context")

    def fake_set_global_client(client: object) -> None:
        client_cache_resets.append(client)
        lifecycle_events.append("clear_client_cache")

    context_module._set_global_client = fake_set_global_client  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.serve", serve_module)
    monkeypatch.setitem(sys.modules, "ray.serve.context", context_module)

    return ray_init_calls, serve_calls, client_cache_resets, lifecycle_events


class TestRayServeBackend:
    def test_stop_uses_public_ray_lifecycle(
        self,
        fake_ray_modules: tuple[list[dict[str, object]], list[str], list[object], list[str]],
    ) -> None:
        ray_init_calls, serve_calls, _, _ = fake_ray_modules
        backend = RayServeBackend(server=object())  # type: ignore[arg-type]

        backend.stop()

        assert ray_init_calls == [{"ignore_reinit_error": True}]
        assert serve_calls == ["shutdown"]

    def test_stop_clears_stale_ray_serve_client_cache(
        self,
        fake_ray_modules: tuple[list[dict[str, object]], list[str], list[object], list[str]],
    ) -> None:
        _, _, client_cache_resets, lifecycle_events = fake_ray_modules
        backend = RayServeBackend(server=object())  # type: ignore[arg-type]

        backend.stop()

        assert client_cache_resets == [None, None]
        assert lifecycle_events == ["clear_client_cache", "ray.init", "serve.shutdown", "clear_client_cache"]

    def test_configure_ray_serve_haproxy_uses_pip_binary_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAY_SERVE_ENABLE_HA_PROXY", raising=False)
        monkeypatch.delenv("RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY", raising=False)
        monkeypatch.delenv("RAY_SERVE_HAPROXY_BINARY_PATH", raising=False)

        RayServeBackend._configure_ray_serve_haproxy()

        assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"
        assert os.environ["RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY"] == "1"

    def test_configure_ray_serve_haproxy_preserves_explicit_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAY_SERVE_ENABLE_HA_PROXY", raising=False)
        monkeypatch.delenv("RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY", raising=False)
        monkeypatch.setenv("RAY_SERVE_HAPROXY_BINARY_PATH", "/usr/bin/haproxy")

        RayServeBackend._configure_ray_serve_haproxy()

        assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"
        assert "RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY" not in os.environ

    def test_to_llm_config_reads_typed_model_config(self) -> None:
        llm_config_type = pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed").LLMConfig
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

        assert isinstance(result, llm_config_type)
        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.deployment_config == {"autoscaling_config": {"min_replicas": 1}}
        assert result.engine_kwargs == {"tensor_parallel_size": 4}
        assert result.runtime_env["pip"] == ["my-package"]
        assert result.runtime_env["env_vars"]["MY_VAR"] == "1"
        assert result.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert result.runtime_env["env_vars"]["RAY_SERVE_LOG_TO_STDERR"] == "0"
