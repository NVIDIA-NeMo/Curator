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
import logging
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

_NOISY_LOGGERS = [
    "ray.serve",
    "ray.serve._private",
    "ray.serve._private.http_proxy",
    "ray.serve._private.replica",
    "ray.serve._private.router",
    "ray.serve.controller",
    "vllm",
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
]


def is_ray_serve_active() -> bool:
    """Check whether a ModelServer is currently running."""
    return _ray_serve_active


def _quiet_ray_serve_loggers(log_level: int = logging.WARNING) -> None:
    """Suppress verbose standard-library loggers used by Ray Serve and vLLM."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(log_level)


def _reset_ray_serve_loggers() -> None:
    """Restore Ray Serve / vLLM loggers to their default level."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)


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
        name: Ray Serve application name. Defaults to "default". Only the
            application with this name is cleaned up on start/stop — other
            applications on the same cluster are left untouched.
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for models to become healthy.
        verbose: If True, keep Ray Serve / vLLM logging at default levels.
            If False (default), suppress noisy loggers to WARNING level and
            disable per-request access logs.

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
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy."""
        global _ray_serve_active  # noqa: PLW0603

        from ray import serve
        from ray.serve.llm import build_openai_app
        from ray.serve.schema import LoggingConfig

        self.port = get_free_port(self.port, get_next_free_port=(self.port == DEFAULT_SERVE_PORT))

        # --- (c) Clean up stale Serve state before deploying ---
        self._teardown_existing_apps()

        llm_configs = [m.to_llm_config() for m in self.models]
        model_names = [m.model_name or m.model_identifier for m in self.models]
        logger.info(f"Starting Ray Serve with models: {model_names} on port {self.port}")

        # --- (a) Control Ray Serve logging verbosity ---
        logging_config = None
        if not self.verbose:
            _quiet_ray_serve_loggers()
            logging_config = LoggingConfig(
                log_level="WARNING",
                enable_access_log=False,
            )

        # Configure the HTTP proxy port *before* deploying so that
        # serve.run() binds to the port we found via get_free_port().
        serve.start(http_options={"port": self.port})

        app = build_openai_app({"llm_configs": llm_configs})
        serve.run(app, name=self.name, blocking=False, logging_config=logging_config)

        self._wait_for_healthy()
        _ray_serve_active = True
        logger.info(f"Ray Serve is ready at {self.endpoint}")

    def stop(self) -> None:
        """Shut down Ray Serve and all deployed models."""
        global _ray_serve_active  # noqa: PLW0603

        from ray import serve

        logger.info("Shutting down Ray Serve")

        # Delete the application first and wait for it to drain, then shut down
        # the Serve system actors.  This two-phase approach gives replicas a
        # chance to finish in-flight requests before the controller is killed.
        try:
            serve.delete(self.name, _blocking=True)
        except Exception:  # noqa: BLE001
            logger.debug(f"Could not delete '{self.name}' app (may already be gone)")

        try:
            serve.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("serve.shutdown() failed (cluster may already be gone)")
        _ray_serve_active = False

        # Wait for the HTTP port to be released so the next start() can bind.
        self._wait_for_port_release()

        # Restore logger levels so other code is not affected after shutdown.
        if not self.verbose:
            _reset_ray_serve_loggers()

        logger.info("Ray Serve shut down successfully")

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models."""
        return f"http://localhost:{self.port}/v1"

    def _teardown_existing_apps(self) -> None:
        """Remove the Serve application with our name if it already exists.

        This prevents conflicts when the cluster already has a deployment
        with a different model loaded (e.g. from a crashed or non-cleanly-stopped
        previous session).  Only the application matching ``self.name`` is
        deleted — other applications on the same cluster are left untouched.
        """
        from ray import serve

        try:
            status = serve.status()
        except Exception:  # noqa: BLE001
            # No Serve controller running — nothing to clean up.
            return

        if self.name not in status.applications:
            return

        logger.info(f"Found existing Serve application '{self.name}', tearing it down before deploying")

        try:
            serve.delete(self.name, _blocking=True)
        except Exception:  # noqa: BLE001
            logger.warning(f"Failed to delete existing Serve application '{self.name}'")

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
                if self.verbose:
                    logger.debug(f"Health check attempt {attempt + 1} failed, retrying...")
            time.sleep(1)
        msg = f"Model server did not become ready within {self.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def _wait_for_port_release(self, timeout_s: int = 10) -> None:
        """Wait for the HTTP port to be released after shutdown."""
        import socket

        for _ in range(timeout_s):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("localhost", self.port)) != 0:
                    return
            time.sleep(1)
        logger.warning(f"Port {self.port} still in use after {timeout_s}s — next start() may need a different port")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
