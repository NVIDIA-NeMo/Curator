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

import json
import logging
import sys
import urllib.request
from collections.abc import Generator
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from nemo_curator.core import serve as serve_module
from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT
from nemo_curator.core.serve import ModelConfig, ModelServer, is_ray_serve_active

INTEGRATION_TEST_MODEL = "Qwen/Qwen2-0.5B"


# ---------------------------------------------------------------------------
# Fixture: mock ray.serve.llm for unit tests (not autouse — applied per-class)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ray_serve_llm() -> Generator[ModuleType]:
    """Inject a mock ray.serve.llm module so unit tests don't need the real ray[serve] extras."""
    mock_llm_module = ModuleType("ray.serve.llm")
    mock_llm_module.LLMConfig = Mock(name="LLMConfig")
    mock_llm_module.build_openai_app = Mock(name="build_openai_app")

    # Also ensure ray.serve has a mock 'run' and 'shutdown'
    ray_serve = sys.modules.get("ray.serve")
    if ray_serve is None:
        ray_serve = ModuleType("ray.serve")
        sys.modules["ray.serve"] = ray_serve
    if not hasattr(ray_serve, "run"):
        ray_serve.run = Mock(name="serve.run")
    if not hasattr(ray_serve, "shutdown"):
        ray_serve.shutdown = Mock(name="serve.shutdown")
    if not hasattr(ray_serve, "delete"):
        ray_serve.delete = Mock(name="serve.delete")
    if not hasattr(ray_serve, "status"):
        # Default: no existing apps
        mock_status = Mock()
        mock_status.applications = {}
        ray_serve.status = Mock(name="serve.status", return_value=mock_status)

    # Mock ray.serve.schema.LoggingConfig
    mock_schema_module = sys.modules.get("ray.serve.schema")
    if mock_schema_module is None:
        mock_schema_module = ModuleType("ray.serve.schema")
        sys.modules["ray.serve.schema"] = mock_schema_module
    if not hasattr(mock_schema_module, "LoggingConfig"):
        mock_schema_module.LoggingConfig = Mock(name="LoggingConfig")
    ray_serve.schema = mock_schema_module

    original = sys.modules.get("ray.serve.llm")
    sys.modules["ray.serve.llm"] = mock_llm_module
    ray_serve.llm = mock_llm_module
    yield mock_llm_module
    if original is not None:
        sys.modules["ray.serve.llm"] = original
    else:
        sys.modules.pop("ray.serve.llm", None)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("mock_ray_serve_llm")
class TestModelConfig:
    def test_defaults(self) -> None:
        config = ModelConfig(model_identifier="meta-llama/Llama-3-8B")
        assert config.model_identifier == "meta-llama/Llama-3-8B"
        assert config.model_name is None
        assert config.deployment_config == {}
        assert config.engine_kwargs == {}

    def test_custom_fields(self) -> None:
        config = ModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 2}},
            engine_kwargs={"tensor_parallel_size": 4},
        )
        assert config.model_identifier == "google/gemma-3-27b-it"
        assert config.model_name == "gemma-27b"
        assert config.deployment_config == {"autoscaling_config": {"min_replicas": 2}}
        assert config.engine_kwargs == {"tensor_parallel_size": 4}

    def test_to_llm_config_uses_model_name(self, mock_ray_serve_llm: ModuleType) -> None:
        mock_llm_cls = mock_ray_serve_llm.LLMConfig
        mock_llm_cls.reset_mock()
        sentinel = Mock()
        mock_llm_cls.return_value = sentinel

        config = ModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
        )
        result = config.to_llm_config()

        mock_llm_cls.assert_called_once_with(
            model_loading_config={
                "model_id": "gemma-27b",
                "model_source": "google/gemma-3-27b-it",
            },
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
        )
        assert result == sentinel

    def test_to_llm_config_falls_back_to_identifier(self, mock_ray_serve_llm: ModuleType) -> None:
        mock_llm_cls = mock_ray_serve_llm.LLMConfig
        mock_llm_cls.reset_mock()

        config = ModelConfig(model_identifier="meta-llama/Llama-3-8B")
        config.to_llm_config()

        mock_llm_cls.assert_called_once_with(
            model_loading_config={
                "model_id": "meta-llama/Llama-3-8B",
                "model_source": "meta-llama/Llama-3-8B",
            },
            deployment_config={},
            engine_kwargs={},
        )


@pytest.mark.usefixtures("mock_ray_serve_llm")
class TestModelServer:
    def test_defaults(self) -> None:
        config = ModelConfig(model_identifier="some-model")
        server = ModelServer(models=[config])
        assert server.name == "default"
        assert server.port == DEFAULT_SERVE_PORT
        assert server.health_check_timeout_s == DEFAULT_SERVE_HEALTH_TIMEOUT_S
        assert server.verbose is False
        assert len(server.models) == 1

    def test_endpoint_property(self) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], port=9999)
        assert server.endpoint == "http://localhost:9999/v1"

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_start_deploys_and_sets_active(self, mock_port: Mock, mock_ray_serve_llm: ModuleType) -> None:
        mock_app = Mock()
        mock_ray_serve_llm.build_openai_app.return_value = mock_app

        config = ModelConfig(model_identifier="some-model")
        server = ModelServer(models=[config])

        with patch.object(server, "_wait_for_healthy"), patch.object(server, "_teardown_existing_apps"):
            server.start()

        mock_port.assert_called_once_with(DEFAULT_SERVE_PORT, get_next_free_port=True)
        mock_ray_serve_llm.build_openai_app.assert_called_once()
        assert is_ray_serve_active()

        # Cleanup
        serve_module._ray_serve_active = False

    @patch("nemo_curator.core.serve.get_free_port", return_value=8001)
    def test_start_uses_custom_port(self, mock_port: Mock) -> None:
        config = ModelConfig(model_identifier="some-model")
        server = ModelServer(models=[config], port=8001)

        with patch.object(server, "_wait_for_healthy"), patch.object(server, "_teardown_existing_apps"):
            server.start()

        # Custom port != DEFAULT_SERVE_PORT, so get_next_free_port=False
        mock_port.assert_called_once_with(8001, get_next_free_port=False)

        # Cleanup
        serve_module._ray_serve_active = False

    def test_stop_shuts_down_and_clears_active(self) -> None:
        serve_module._ray_serve_active = True
        assert is_ray_serve_active()

        server = ModelServer(models=[ModelConfig(model_identifier="m")])

        with patch.object(server, "_wait_for_port_release"):
            server.stop()

        assert not is_ray_serve_active()

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_context_manager(self, mock_port: Mock) -> None:
        config = ModelConfig(model_identifier="some-model")
        server = ModelServer(models=[config])

        with (
            patch.object(server, "_wait_for_healthy"),
            patch.object(server, "_teardown_existing_apps"),
            patch.object(server, "_wait_for_port_release"),
            server as s,
        ):
            assert s is server
            assert is_ray_serve_active()

        mock_port.assert_called_once()
        assert not is_ray_serve_active()

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_start_multiple_models(self, mock_port: Mock, mock_ray_serve_llm: ModuleType) -> None:
        mock_ray_serve_llm.build_openai_app.reset_mock()
        configs = [
            ModelConfig(model_identifier="model-a", model_name="a"),
            ModelConfig(model_identifier="model-b", model_name="b"),
        ]
        server = ModelServer(models=configs)

        with patch.object(server, "_wait_for_healthy"), patch.object(server, "_teardown_existing_apps"):
            server.start()

        mock_port.assert_called_once()
        # build_openai_app should receive two LLMConfigs
        call_args = mock_ray_serve_llm.build_openai_app.call_args[0][0]
        assert len(call_args["llm_configs"]) == 2

        # Cleanup
        serve_module._ray_serve_active = False

    def test_wait_for_healthy_success(self) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], port=8000, health_check_timeout_s=5)

        mock_response = Mock()
        mock_response.status = 200

        with patch("urllib.request.urlopen", return_value=mock_response):
            server._wait_for_healthy()

    def test_wait_for_healthy_timeout(self) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], port=8000, health_check_timeout_s=2)

        with (
            patch("urllib.request.urlopen", side_effect=ConnectionError("refused")),
            patch("time.sleep"),
            pytest.raises(TimeoutError, match="did not become ready within 2s"),
        ):
            server._wait_for_healthy()


@pytest.mark.usefixtures("mock_ray_serve_llm")
class TestVerboseLogging:
    """Tests for the verbose flag controlling Ray Serve / vLLM log levels."""

    def teardown_method(self) -> None:
        serve_module._ray_serve_active = False
        serve_module._reset_ray_serve_loggers()

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_verbose_false_suppresses_loggers(self, mock_port: Mock) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], verbose=False)

        with patch.object(server, "_wait_for_healthy"), patch.object(server, "_teardown_existing_apps"):
            server.start()

        # Noisy loggers should be at WARNING or higher
        for name in serve_module._NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING

        serve_module._ray_serve_active = False

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_verbose_true_keeps_default_levels(self, mock_port: Mock) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], verbose=True)

        with patch.object(server, "_wait_for_healthy"), patch.object(server, "_teardown_existing_apps"):
            server.start()

        # Loggers should remain at NOTSET (default)
        for name in serve_module._NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.NOTSET

        serve_module._ray_serve_active = False

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_stop_restores_loggers(self, mock_port: Mock) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], verbose=False)

        with (
            patch.object(server, "_wait_for_healthy"),
            patch.object(server, "_teardown_existing_apps"),
            patch.object(server, "_wait_for_port_release"),
        ):
            server.start()
            # Loggers are suppressed during run
            for name in serve_module._NOISY_LOGGERS:
                assert logging.getLogger(name).level >= logging.WARNING

            server.stop()

        # After stop, loggers should be restored
        for name in serve_module._NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.NOTSET


@pytest.mark.usefixtures("mock_ray_serve_llm")
class TestTeardownExistingApps:
    """Tests for cleaning up stale Serve state before deploying."""

    def test_no_existing_apps_is_noop(self) -> None:
        from ray import serve

        mock_status = Mock()
        mock_status.applications = {}

        mock_delete = Mock()

        with patch.object(serve, "status", return_value=mock_status), patch.object(serve, "delete", mock_delete):
            server = ModelServer(models=[ModelConfig(model_identifier="m")])
            server._teardown_existing_apps()

        mock_delete.assert_not_called()

    def test_only_own_app_is_deleted(self) -> None:
        from ray import serve

        # Cluster has "default" and "other-app", but ModelServer only owns "default"
        mock_status = Mock()
        mock_status.applications = {"default": Mock(), "other-app": Mock()}

        mock_delete = Mock()

        with (
            patch.object(serve, "status", return_value=mock_status),
            patch.object(serve, "delete", mock_delete),
        ):
            server = ModelServer(models=[ModelConfig(model_identifier="m")])
            server._teardown_existing_apps()

        # Only "default" should be deleted — "other-app" is left alone
        mock_delete.assert_called_once_with("default", _blocking=True)

    def test_custom_name_only_deletes_own_app(self) -> None:
        from ray import serve

        mock_status = Mock()
        mock_status.applications = {"default": Mock(), "my-app": Mock()}

        mock_delete = Mock()

        with (
            patch.object(serve, "status", return_value=mock_status),
            patch.object(serve, "delete", mock_delete),
        ):
            server = ModelServer(models=[ModelConfig(model_identifier="m")], name="my-app")
            server._teardown_existing_apps()

        # Only "my-app" should be deleted — "default" is left alone
        mock_delete.assert_called_once_with("my-app", _blocking=True)

    def test_no_matching_app_is_noop(self) -> None:
        from ray import serve

        mock_status = Mock()
        mock_status.applications = {"other-app": Mock()}

        mock_delete = Mock()

        with patch.object(serve, "status", return_value=mock_status), patch.object(serve, "delete", mock_delete):
            server = ModelServer(models=[ModelConfig(model_identifier="m")])
            server._teardown_existing_apps()

        mock_delete.assert_not_called()

    def test_status_error_is_silently_ignored(self) -> None:
        from ray import serve

        with patch.object(serve, "status", side_effect=RuntimeError("no controller")):
            server = ModelServer(models=[ModelConfig(model_identifier="m")])
            # Should not raise
            server._teardown_existing_apps()


@pytest.mark.usefixtures("mock_ray_serve_llm")
class TestStopCleanup:
    """Tests for the two-phase stop (delete + shutdown) and port wait."""

    def teardown_method(self) -> None:
        serve_module._ray_serve_active = False

    def test_stop_calls_delete_then_shutdown(self) -> None:
        from ray import serve

        mock_delete = Mock()
        mock_shutdown = Mock()

        serve_module._ray_serve_active = True
        server = ModelServer(models=[ModelConfig(model_identifier="m")])

        with (
            patch.object(serve, "delete", mock_delete),
            patch.object(serve, "shutdown", mock_shutdown),
            patch.object(server, "_wait_for_port_release"),
        ):
            server.stop()

        mock_delete.assert_called_once_with("default", _blocking=True)
        mock_shutdown.assert_called_once()
        assert not is_ray_serve_active()

    def test_stop_uses_custom_name(self) -> None:
        from ray import serve

        mock_delete = Mock()
        mock_shutdown = Mock()

        serve_module._ray_serve_active = True
        server = ModelServer(models=[ModelConfig(model_identifier="m")], name="my-app")

        with (
            patch.object(serve, "delete", mock_delete),
            patch.object(serve, "shutdown", mock_shutdown),
            patch.object(server, "_wait_for_port_release"),
        ):
            server.stop()

        mock_delete.assert_called_once_with("my-app", _blocking=True)

    def test_stop_survives_delete_error(self) -> None:
        from ray import serve

        mock_delete = Mock(side_effect=RuntimeError("already gone"))
        mock_shutdown = Mock()

        serve_module._ray_serve_active = True
        server = ModelServer(models=[ModelConfig(model_identifier="m")])

        with (
            patch.object(serve, "delete", mock_delete),
            patch.object(serve, "shutdown", mock_shutdown),
            patch.object(server, "_wait_for_port_release"),
        ):
            server.stop()

        # shutdown still called even though delete failed
        mock_shutdown.assert_called_once()
        assert not is_ray_serve_active()

    def test_wait_for_port_release_returns_when_free(self) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="m")], port=19876)

        with patch("socket.socket") as mock_sock_cls:
            mock_sock = Mock()
            mock_sock_cls.return_value.__enter__ = Mock(return_value=mock_sock)
            mock_sock_cls.return_value.__exit__ = Mock(return_value=False)
            # Port not in use (connect fails)
            mock_sock.connect_ex.return_value = 1

            server._wait_for_port_release(timeout_s=3)

        # Should return immediately on first check
        mock_sock.connect_ex.assert_called_once()


@pytest.mark.usefixtures("mock_ray_serve_llm")
class TestIsRayServeActive:
    def setup_method(self) -> None:
        serve_module._ray_serve_active = False

    def teardown_method(self) -> None:
        serve_module._ray_serve_active = False

    def test_initially_false(self) -> None:
        assert not is_ray_serve_active()

    def test_reflects_global_state(self) -> None:
        serve_module._ray_serve_active = True
        assert is_ray_serve_active()


# ---------------------------------------------------------------------------
# Integration tests (require GPU + real Ray Serve)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestModelServerIntegration:
    def teardown_method(self) -> None:
        serve_module._ray_serve_active = False

    def test_serve_and_query_model(self) -> None:
        """Start ModelServer with a real model, query it, and verify the response."""
        config = ModelConfig(
            model_identifier=INTEGRATION_TEST_MODEL,
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,
                },
            },
            engine_kwargs={
                "tensor_parallel_size": 1,
                "max_model_len": 512,
            },
        )

        server = ModelServer(models=[config], health_check_timeout_s=600)

        try:
            server.start()

            assert is_ray_serve_active()

            # Verify /v1/models lists our model
            models_url = f"{server.endpoint}/models"
            resp = urllib.request.urlopen(models_url, timeout=10)  # noqa: S310
            models_data = json.loads(resp.read().decode())
            model_ids = [m["id"] for m in models_data["data"]]
            assert INTEGRATION_TEST_MODEL in model_ids

            # Send a chat completion request
            chat_url = f"{server.endpoint}/chat/completions"
            payload = json.dumps(
                {
                    "model": INTEGRATION_TEST_MODEL,
                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                    "max_tokens": 16,
                    "temperature": 0.0,
                }
            ).encode()
            req = urllib.request.Request(  # noqa: S310
                chat_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)  # noqa: S310
            chat_data = json.loads(resp.read().decode())

            assert "choices" in chat_data
            assert len(chat_data["choices"]) > 0
            content = chat_data["choices"][0]["message"]["content"]
            assert isinstance(content, str)
            assert len(content) > 0

        finally:
            server.stop()

        assert not is_ray_serve_active()

    def test_context_manager_lifecycle(self) -> None:
        """Verify the context manager starts and stops cleanly."""
        config = ModelConfig(
            model_identifier=INTEGRATION_TEST_MODEL,
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,
                },
            },
            engine_kwargs={
                "tensor_parallel_size": 1,
                "max_model_len": 512,
            },
        )

        with ModelServer(models=[config], health_check_timeout_s=600) as server:
            assert is_ray_serve_active()

            # Quick health check
            models_url = f"{server.endpoint}/models"
            resp = urllib.request.urlopen(models_url, timeout=10)  # noqa: S310
            assert resp.status == 200

        assert not is_ray_serve_active()
