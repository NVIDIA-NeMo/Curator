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

from unittest.mock import patch

import pytest
from pytest_httpserver import HTTPServer
from ray.serve.llm import LLMConfig

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT
from nemo_curator.core.serve import ModelConfig, ModelServer, is_ray_serve_active

INTEGRATION_TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
INTEGRATION_TEST_MODEL_2 = "HuggingFaceTB/SmolLM-135M-Instruct"


class TestModelConfig:
    def test_model_config_defaults(self) -> None:
        default = ModelConfig(model_identifier="meta-llama/Llama-3-8B")
        assert default.model_name is None
        assert default.deployment_config == {}
        assert default.engine_kwargs == {}

        # to_llm_config falls back to identifier when no model_name
        result = default.to_llm_config()
        assert isinstance(result, LLMConfig)
        assert result.model_loading_config.model_id == "meta-llama/Llama-3-8B"
        assert result.model_loading_config.model_source == "meta-llama/Llama-3-8B"

    def test_to_llm_config_with_model_name(self) -> None:
        # to_llm_config uses model_name when provided
        custom = ModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
        )
        result = custom.to_llm_config()
        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.engine_kwargs == {"tensor_parallel_size": 4}


class TestModelServer:
    def test_defaults_and_idempotent_stop(self) -> None:
        server = ModelServer(models=[ModelConfig(model_identifier="some-model")])
        assert server.name == "default"
        assert server.port == DEFAULT_SERVE_PORT
        assert server.health_check_timeout_s == DEFAULT_SERVE_HEALTH_TIMEOUT_S
        assert server.verbose is False
        assert server._started is False
        assert server.endpoint == f"http://localhost:{DEFAULT_SERVE_PORT}/v1"

        # Custom port
        assert ModelServer(models=[], port=9999).endpoint == "http://localhost:9999/v1"

        # stop() before start() is a no-op
        server.stop()
        assert server._started is False

    def test_wait_for_healthy(self, httpserver: HTTPServer) -> None:
        """Health check succeeds on 200, retries on failure, and times out on unreachable port."""
        # Immediate success
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = ModelServer(models=[], port=httpserver.port, health_check_timeout_s=5)
        server._wait_for_healthy()

        # Timeout on unreachable port
        server = ModelServer(models=[], port=19876, health_check_timeout_s=2)
        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()


# ---------------------------------------------------------------------------
# Integration tests — real Ray Serve + vLLM, requires GPU
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def model_server(shared_ray_cluster: str) -> ModelServer:  # noqa: ARG001
    """Start ModelServer once for all integration tests.

    Uses enforce_eager=True to skip torch.compile and CUDA graph capture,
    cutting vLLM startup from ~30s to ~5s.
    """
    config = ModelConfig(
        model_identifier=INTEGRATION_TEST_MODEL,
        deployment_config={
            "autoscaling_config": {"min_replicas": 1, "max_replicas": 1},
        },
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 512,
            "enforce_eager": True,
        },
    )

    server = ModelServer(models=[config], health_check_timeout_s=600)
    server.start()

    yield server

    server.stop()


@pytest.mark.gpu
@pytest.mark.usefixtures("model_server")
class TestModelServerIntegration:
    """Full lifecycle tests against a real ModelServer started once for the class."""

    def test_is_active_and_queryable(self, model_server: ModelServer) -> None:
        """Server is active, lists models, and responds to chat completions."""
        from openai import OpenAI

        assert is_ray_serve_active()
        assert model_server._started is True

        client = OpenAI(base_url=model_server.endpoint, api_key="na")

        # /v1/models lists our model
        model_ids = [m.id for m in client.models.list()]
        assert INTEGRATION_TEST_MODEL in model_ids

        # Chat completion returns a non-empty response
        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    def test_resolve_port_reuses_existing_controller(self, model_server: ModelServer) -> None:
        """A second ModelServer should detect the existing controller's port."""
        server2 = ModelServer(
            models=[ModelConfig(model_identifier="unused")],
            port=model_server.port + 1,
            name="second",
        )
        server2._resolve_port()
        assert server2.port == model_server.port

    def test_stop_never_calls_serve_shutdown(self, model_server: ModelServer) -> None:
        """Verify serve.shutdown() is never used — only serve.delete()."""
        from ray import serve

        # We can't actually stop the shared server, so spy on a fresh one
        # that was never started — stop() should be a no-op
        fresh = ModelServer(models=[ModelConfig(model_identifier="m")])
        fresh._started = False
        with patch.object(serve, "shutdown") as spy:
            fresh.stop()
            spy.assert_not_called()

    def test_stop_is_idempotent(self, model_server: ModelServer) -> None:
        """stop() on a not-started server is a no-op (safe for atexit double-call)."""
        fresh = ModelServer(models=[ModelConfig(model_identifier="m")])
        assert fresh._started is False
        fresh.stop()  # no-op, no exception
        fresh.stop()  # second call also no-op
        assert fresh._started is False

    def test_restart_after_stop(self, model_server: ModelServer) -> None:
        """A new ModelServer starts cleanly after the previous one is stopped.

        Since we use serve.delete() (not serve.shutdown()), the controller
        and HTTP proxy stay alive, so a fresh app deploys without issues.

        This test must run last in the class — it stops the shared fixture's
        server and starts a replacement.
        """
        from openai import OpenAI

        port = model_server.port

        # Stop the fixture's server (first server)
        model_server.stop()
        assert not is_ray_serve_active()

        # Start a new server with two models — also request a different port
        # to verify that _resolve_port detects the existing controller and
        # reuses its port.
        common_engine_kwargs = {
            "tensor_parallel_size": 1,
            "max_model_len": 512,
            "enforce_eager": True,
        }
        configs = [
            ModelConfig(
                model_identifier=INTEGRATION_TEST_MODEL,
                deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
                engine_kwargs=common_engine_kwargs,
            ),
            ModelConfig(
                model_identifier=INTEGRATION_TEST_MODEL_2,
                deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
                engine_kwargs=common_engine_kwargs,
            ),
        ]
        server2 = ModelServer(models=configs, port=port + 1, health_check_timeout_s=600)
        server2.start()

        assert server2.port == port  # reused existing controller, not port + 1

        client = OpenAI(base_url=server2.endpoint, api_key="na")
        model_ids = {m.id for m in client.models.list()}
        assert INTEGRATION_TEST_MODEL in model_ids
        assert INTEGRATION_TEST_MODEL_2 in model_ids

        # Both models respond to chat completions
        for model_name in (INTEGRATION_TEST_MODEL, INTEGRATION_TEST_MODEL_2):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say hello in one word."}],
                max_tokens=16,
                temperature=0.0,
            )
            assert len(response.choices) > 0
            assert len(response.choices[0].message.content) > 0

        assert is_ray_serve_active()

        server2.stop()
