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

from __future__ import annotations

import http
import time
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.serve.llm import LLMConfig

from loguru import logger

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT
from nemo_curator.core.utils import get_free_port

_ray_serve_active: bool = False


def is_ray_serve_active() -> bool:
    """Check whether a ModelServer is currently running."""
    return _ray_serve_active


@dataclass
class ModelConfig:
    """Configuration for a single model to be served via Ray Serve.

    Args:
        model_identifier: HuggingFace model ID or local path (maps to model_source in LLMConfig).
        model_name: API-facing model name clients use in requests. Defaults to model_identifier.
        deployment_config: Ray Serve deployment configuration (autoscaling, replicas, etc.).
            Passed directly to LLMConfig.deployment_config.
        engine_kwargs: vLLM engine keyword arguments (tensor_parallel_size, etc.).
            Passed directly to LLMConfig.engine_kwargs.
    """

    model_identifier: str
    model_name: str | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_llm_config(self) -> LLMConfig:
        """Convert to a Ray Serve LLMConfig."""
        from ray.serve.llm import LLMConfig

        return LLMConfig(
            model_loading_config={
                "model_id": self.model_name or self.model_identifier,
                "model_source": self.model_identifier,
            },
            deployment_config=self.deployment_config,
            engine_kwargs=self.engine_kwargs,
        )


@dataclass
class ModelServer:
    """Serve one or more models via Ray Serve with an OpenAI-compatible endpoint.

    Requires a running Ray cluster (e.g. via RayClient or RAY_ADDRESS env var).

    Args:
        models: List of ModelConfig instances to deploy.
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for models to become healthy.

    Example::

        from nemo_curator.core.serve import ModelConfig, ModelServer

        config = ModelConfig(
            model_identifier="google/gemma-3-27b-it",
            engine_kwargs={"tensor_parallel_size": 4},
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,
                },
            },
        )

        with ModelServer(models=[config]) as server:
            print(server.endpoint)  # http://localhost:8000/v1
            # Use with OpenAI SDK or NeMo Curator stages
    """

    models: list[ModelConfig]
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy."""
        global _ray_serve_active  # noqa: PLW0603

        from ray import serve
        from ray.serve.llm import build_openai_app

        self.port = get_free_port(self.port, get_next_free_port=(self.port == DEFAULT_SERVE_PORT))

        llm_configs = [m.to_llm_config() for m in self.models]
        model_names = [m.model_name or m.model_identifier for m in self.models]
        logger.info(f"Starting Ray Serve with models: {model_names} on port {self.port}")

        app = build_openai_app({"llm_configs": llm_configs})
        serve.run(app, blocking=False)

        self._wait_for_healthy()
        _ray_serve_active = True
        logger.info(f"Ray Serve is ready at {self.endpoint}")

    def stop(self) -> None:
        """Shut down Ray Serve and all deployed models."""
        global _ray_serve_active  # noqa: PLW0603

        from ray import serve

        logger.info("Shutting down Ray Serve")
        serve.shutdown()
        _ray_serve_active = False

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models."""
        return f"http://localhost:{self.port}/v1"

    def _wait_for_healthy(self) -> None:
        """Poll the /v1/models endpoint until all models are ready."""
        models_url = f"http://localhost:{self.port}/v1/models"
        for attempt in range(self.health_check_timeout_s):
            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    logger.info(f"Model server ready after {attempt + 1}s")
                    return
            except Exception:  # noqa: BLE001
                logger.debug(f"Health check attempt {attempt + 1} failed, retrying...")
            time.sleep(1)
        msg = f"Model server did not become ready within {self.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
