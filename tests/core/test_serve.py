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
        assert server.port == DEFAULT_SERVE_PORT
        assert server.health_check_timeout_s == DEFAULT_SERVE_HEALTH_TIMEOUT_S
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

        with patch.object(server, "_wait_for_healthy"):
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

        with patch.object(server, "_wait_for_healthy"):
            server.start()

        # Custom port != DEFAULT_SERVE_PORT, so get_next_free_port=False
        mock_port.assert_called_once_with(8001, get_next_free_port=False)

        # Cleanup
        serve_module._ray_serve_active = False

    def test_stop_shuts_down_and_clears_active(self) -> None:
        serve_module._ray_serve_active = True
        assert is_ray_serve_active()

        server = ModelServer(models=[ModelConfig(model_identifier="m")])
        server.stop()

        assert not is_ray_serve_active()

    @patch("nemo_curator.core.serve.get_free_port", return_value=8000)
    def test_context_manager(self, mock_port: Mock) -> None:
        config = ModelConfig(model_identifier="some-model")
        server = ModelServer(models=[config])

        with patch.object(server, "_wait_for_healthy"), server as s:
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

        with patch.object(server, "_wait_for_healthy"):
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
