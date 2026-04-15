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

import importlib
from unittest.mock import patch

import pytest
from pytest_httpserver import HTTPServer

import nemo_curator.core.serve as serve_module

LLMConfig = pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed").LLMConfig

from nemo_curator.core.serve import InferenceModelConfig, InferenceServer  # noqa: E402


class TestServePackageExports:
    def test_private_active_server_state_is_not_in_all(self) -> None:
        assert "is_inference_server_active" in serve_module.__all__
        assert "_active_servers" not in serve_module.__all__


class TestInferenceModelConfig:
    def test_to_llm_config_falls_back_to_identifier(self) -> None:
        config = InferenceModelConfig(model_identifier="meta-llama/Llama-3-8B")
        result = config.to_llm_config()
        assert isinstance(result, LLMConfig)
        assert result.model_loading_config.model_id == "meta-llama/Llama-3-8B"
        assert result.model_loading_config.model_source == "meta-llama/Llama-3-8B"

    def test_to_llm_config_with_model_name(self) -> None:
        custom = InferenceModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
        )
        result = custom.to_llm_config()
        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.engine_kwargs == {"tensor_parallel_size": 4}

    def test_to_llm_config_quiet_env_merges_with_user_runtime_env(self) -> None:
        """Quiet env vars override user's logging vars but preserve other runtime_env keys."""
        config = InferenceModelConfig(
            model_identifier="some-model",
            runtime_env={
                "pip": ["my-package"],
                "env_vars": {"MY_VAR": "1", "VLLM_LOGGING_LEVEL": "DEBUG"},
            },
        )
        try:
            backend_module = importlib.import_module("nemo_curator.core.serve.ray_serve.backend")
        except ModuleNotFoundError as exc:
            pytest.fail(f"Ray Serve backend module should exist: {exc}")

        quiet_env = backend_module.RayServeBackend._quiet_runtime_env()
        result = config.to_llm_config(quiet_runtime_env=quiet_env)

        assert result.runtime_env["pip"] == ["my-package"]
        assert result.runtime_env["env_vars"]["MY_VAR"] == "1"
        # quiet overrides the user's DEBUG with WARNING
        assert result.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert result.runtime_env["env_vars"]["RAY_SERVE_LOG_TO_STDERR"] == "0"
        assert not hasattr(InferenceServer, "_quiet_runtime_env")

        # Without quiet_env, user's runtime_env is passed through as-is
        result_verbose = config.to_llm_config()
        assert result_verbose.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "DEBUG"
        assert "RAY_SERVE_LOG_TO_STDERR" not in result_verbose.runtime_env["env_vars"]


class TestInferenceServer:
    def test_endpoint_uses_configured_port(self) -> None:
        assert InferenceServer(models=[], port=9999).endpoint == "http://localhost:9999/v1"

    def test_stop_before_start_is_noop(self) -> None:
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="some-model")])
        server.stop()
        assert server._started is False

    def test_start_stop_delegates_to_backend(self) -> None:
        class StubBackend:
            def __init__(self) -> None:
                self.started = False
                self.stopped = False

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

        server = InferenceServer(models=[InferenceModelConfig(model_identifier="some-model")])
        backend = StubBackend()
        from nemo_curator.core.serve.server import _active_servers

        with (
            patch("atexit.register"),
            patch("nemo_curator.core.serve.server.logger.info") as info_log,
            patch.object(InferenceServer, "_create_backend", return_value=backend, create=True),
        ):
            server.start()

        with (
            patch("atexit.unregister"),
        ):
            server.stop()

        assert backend.started is True
        assert backend.stopped is True
        info_log.assert_called_with(f"Inference server is ready at {server.endpoint}")
        assert server._started is False
        assert server.name not in _active_servers

    def test_wait_for_healthy(self, httpserver: HTTPServer) -> None:
        """Health check succeeds on 200, retries on failure, and times out on unreachable port."""
        # Immediate success
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = InferenceServer(models=[], port=httpserver.port, health_check_timeout_s=5)
        server._wait_for_healthy()

        # Timeout on unreachable port
        server = InferenceServer(models=[], port=19876, health_check_timeout_s=2)
        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()

    def test_start_raises_when_another_server_active(self) -> None:
        """start() raises RuntimeError if another InferenceServer is already active."""
        from nemo_curator.core.serve.server import _active_servers

        server = InferenceServer(models=[InferenceModelConfig(model_identifier="some-model")])

        _active_servers.add("other-app")
        try:
            with pytest.raises(RuntimeError, match="already active"):
                server.start()
        finally:
            _active_servers.discard("other-app")

    def test_stop_calls_shutdown(self) -> None:
        """stop() calls serve.shutdown() when the server was started."""
        from ray import serve

        from nemo_curator.core.serve.server import _active_servers

        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        server._started = True
        _active_servers.add(server.name)
        try:
            with patch.object(serve, "shutdown"):
                server.stop()
            assert server._started is False
            assert server.name not in _active_servers
        finally:
            _active_servers.discard(server.name)

    def test_stop_skips_shutdown_when_not_started(self) -> None:
        """stop() on a not-started server is a no-op — serve.shutdown() is not called."""
        from ray import serve

        fresh = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        fresh._started = False
        with patch.object(serve, "shutdown") as spy:
            fresh.stop()
            spy.assert_not_called()

    def test_stop_is_idempotent(self) -> None:
        """stop() called twice on a not-started server is safe (atexit double-call)."""
        fresh = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        assert fresh._started is False
        fresh.stop()
        fresh.stop()
        assert fresh._started is False
