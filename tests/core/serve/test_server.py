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

from nemo_curator.core.serve import (
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    RayServeModelConfig,
    RayServeServerConfig,
)
from nemo_curator.core.serve.dynamo.backend import DynamoBackend
from nemo_curator.core.serve.ray_serve.backend import RayServeBackend


class TestInferenceServer:
    def test_endpoint_uses_configured_port(self) -> None:
        assert InferenceServer(models=[], port=9999).endpoint == "http://localhost:9999/v1"

    def test_stop_before_start_is_noop(self) -> None:
        server = InferenceServer(models=[RayServeModelConfig(model_identifier="some-model")])

        server.stop()

        assert server._started is False

    def test_dispatches_to_correct_backend(self) -> None:
        ray_server = InferenceServer(models=[RayServeModelConfig(model_identifier="ray-model")])
        dynamo_server = InferenceServer(
            models=[DynamoVLLMModelConfig(model_identifier="dynamo-model")],
            backend=DynamoServerConfig(),
        )

        assert isinstance(ray_server.backend, RayServeServerConfig)
        assert isinstance(ray_server._create_backend(), RayServeBackend)
        assert isinstance(dynamo_server._create_backend(), DynamoBackend)

    def test_init_rejects_backend_model_mismatch(self) -> None:
        with pytest.raises(TypeError, match="RayServeServerConfig accepts RayServeModelConfig"):
            InferenceServer(
                models=[DynamoVLLMModelConfig(model_identifier="some-model")],
                backend=RayServeServerConfig(),
            )

    def test_init_rejects_mixed_model_config_types(self) -> None:
        from dataclasses import dataclass
        from typing import ClassVar

        from nemo_curator.core.serve import BaseModelConfig, BaseServerConfig

        @dataclass
        class _EngineA(BaseModelConfig):
            pass

        @dataclass
        class _EngineB(BaseModelConfig):
            pass

        @dataclass
        class _MultiEngineServerConfig(BaseServerConfig):
            model_configs: ClassVar[tuple[type[BaseModelConfig], ...]] = (_EngineA, _EngineB)

        with pytest.raises(TypeError, match="must be the same config type"):
            InferenceServer(
                models=[_EngineA(model_identifier="model-a"), _EngineB(model_identifier="model-b")],
                backend=_MultiEngineServerConfig(),
            )

    def test_start_stop_delegates_to_backend(self) -> None:
        class StubBackend:
            def __init__(self) -> None:
                self.started = False
                self.stopped = False

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

        server = InferenceServer(models=[RayServeModelConfig(model_identifier="some-model")])
        backend = StubBackend()
        from nemo_curator.core.serve.server import _active_servers

        with (
            patch("atexit.register"),
            patch("nemo_curator.core.serve.server.logger.info") as info_log,
            patch.object(InferenceServer, "_create_backend", return_value=backend, create=True),
        ):
            server.start()

        with patch("atexit.unregister"):
            server.stop()

        assert backend.started is True
        assert backend.stopped is True
        info_log.assert_called_with(f"Inference server is ready at {server.endpoint}")
        assert server._started is False
        assert server.name not in _active_servers

    def test_wait_for_healthy(self, httpserver: HTTPServer) -> None:
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = InferenceServer(models=[], port=httpserver.port, health_check_timeout_s=5)
        server._wait_for_healthy()

        server = InferenceServer(models=[], port=19876, health_check_timeout_s=2)
        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()
