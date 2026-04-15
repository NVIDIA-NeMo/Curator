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

import atexit
import http
import logging
import time
import urllib.request
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ray.serve.llm import LLMConfig

from loguru import logger

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT

# Track which application names are currently managed by an InferenceServer in
# this process. The public helpers below let other parts of the codebase
# detect possible GPU contention with active model serving.
_active_servers: set[str] = set()


class InferenceBackend(Protocol):
    """Internal backend contract used by ``InferenceServer`` lifecycle code."""

    def start(self) -> None:
        """Start the inference backend."""

    def stop(self) -> None:
        """Stop the inference backend."""


def is_inference_server_active() -> bool:
    """Check whether any inference server is currently running in this process."""
    return bool(_active_servers)


@dataclass
class InferenceModelConfig:
    """Configuration for a single model to be served via Ray Serve.

    Args:
        model_identifier: HuggingFace model ID or local path (maps to model_source in LLMConfig).
        model_name: API-facing model name clients use in requests. Defaults to model_identifier.
        deployment_config: Ray Serve deployment configuration (autoscaling, replicas, etc.).
            Passed directly to LLMConfig.deployment_config.
        engine_kwargs: vLLM engine keyword arguments (tensor_parallel_size, etc.).
            Passed directly to LLMConfig.engine_kwargs.
        runtime_env: Ray runtime environment configuration (pip packages, env_vars, working_dir, etc.).
            Merged with quiet logging overrides when ``verbose=False`` on the InferenceServer.
    """

    model_identifier: str
    model_name: str | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_env: dict[str, Any] = field(default_factory=dict)

    def to_llm_config(self, quiet_runtime_env: dict[str, Any] | None = None) -> "LLMConfig":
        """Convert to a Ray Serve LLMConfig.

        Args:
            quiet_runtime_env: Optional runtime environment with quiet/logging
                overrides. Merged on top of ``self.runtime_env`` so that quiet
                env vars take precedence while preserving user-provided keys
                (e.g. ``pip``, ``working_dir``).
        """
        from ray.serve.llm import LLMConfig

        merged_env = self._merge_runtime_envs(self.runtime_env, quiet_runtime_env)

        return LLMConfig(
            model_loading_config={
                "model_id": self.model_name or self.model_identifier,
                "model_source": self.model_identifier,
            },
            deployment_config=self.deployment_config,
            engine_kwargs=self.engine_kwargs,
            runtime_env=merged_env or None,
        )

    @staticmethod
    def _merge_runtime_envs(
        base: dict[str, Any],
        override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge two runtime_env dicts, with special handling for ``env_vars``.

        Top-level keys from *override* win, except ``env_vars`` which is
        merged key-by-key (override env vars take precedence over base).
        """
        if not base and not override:
            return {}
        if not override:
            return {**base}
        if not base:
            return {**override}

        merged = {**base, **override}
        base_env_vars = base.get("env_vars", {})
        override_env_vars = override.get("env_vars", {})
        if base_env_vars or override_env_vars:
            merged["env_vars"] = {**base_env_vars, **override_env_vars}
        return merged


@dataclass
class InferenceServer:
    """Serve one or more models via Ray Serve with an OpenAI-compatible endpoint.

    Requires a running Ray cluster (e.g. via RayClient or RAY_ADDRESS env var).

    Cleanup semantics:
        ``stop()`` tears down the active backend. For Ray Serve this calls
        ``serve.shutdown()``, releasing deployments, the Serve controller, and
        the HTTP proxy.

    Args:
        models: List of InferenceModelConfig instances to deploy.
        name: Ray Serve application name (default ``"default"``).
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for models to become healthy.
        verbose: If True, keep Ray Serve and vLLM logging at default levels.
            If False (default), suppress per-request logs from both vLLM and
            Ray Serve access logs.
    """

    models: list[InferenceModelConfig]
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False

    _started: bool = field(init=False, default=False, repr=False)
    _backend_impl: InferenceBackend | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.verbose:
            # Suppress driver-side Ray Serve INFO logs ("Deployment Options",
            # "Ingress Options", "Started Serve", etc.).
            logging.getLogger("ray.serve").setLevel(logging.WARNING)

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy.

        Raises:
            RuntimeError: If another InferenceServer is already active in this
                process. Only one InferenceServer can run at a time because Ray
                Serve uses a single HTTP proxy per cluster, and all models are
                deployed as a single application sharing the same ``/v1`` routes.
        """
        if _active_servers:
            running = ", ".join(sorted(_active_servers))
            msg = (
                f"Cannot start InferenceServer '{self.name}': another InferenceServer is "
                f"already active (running: {running}). Stop the existing server first."
            )
            raise RuntimeError(msg)

        atexit.register(self.stop)
        self._backend_impl = self._create_backend()
        try:
            self._backend_impl.start()
        except Exception:
            self._backend_impl = None
            with suppress(ValueError):
                atexit.unregister(self.stop)
            raise

        _active_servers.add(self.name)
        self._started = True
        logger.info(f"Ray Serve is ready at {self.endpoint}")

    def _create_backend(self) -> InferenceBackend:
        from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

        return RayServeBackend(self)

    def stop(self) -> None:
        """Shut down the active backend and release resources."""
        if not self._started:
            return

        try:
            if self._backend_impl is not None:
                self._backend_impl.stop()
        finally:
            self._backend_impl = None
            _active_servers.discard(self.name)
            self._started = False
            with suppress(ValueError):
                atexit.unregister(self.stop)

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models."""
        return f"http://localhost:{self.port}/v1"

    def _wait_for_healthy(self) -> None:
        """Poll the /v1/models endpoint until all models are ready.

        Uses wall-clock time to enforce the timeout accurately, regardless
        of how long individual HTTP requests take.
        """
        models_url = f"{self.endpoint}/models"
        deadline = time.monotonic() + self.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    logger.info(f"Model server ready after {attempt} health check(s)")
                    return
            except Exception:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(1)
        msg = f"Model server did not become ready within {self.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
