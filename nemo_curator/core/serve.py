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
import time
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.serve.llm import LLMConfig

from loguru import logger
from ray import serve
from ray.serve.llm import build_openai_app
from ray.serve.schema import LoggingConfig

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT
from nemo_curator.core.utils import get_free_port

# Track which application names are currently managed by a ModelServer in
# this process.  ``is_ray_serve_active()`` checks this set so that other
# parts of the codebase (e.g. Pipeline.run()) can detect potential GPU
# resource contention.
_active_servers: set[str] = set()


def is_ray_serve_active() -> bool:
    """Check whether any ModelServer is currently running in this process."""
    return bool(_active_servers)


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

    def to_llm_config(self) -> "LLMConfig":
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

    Cleanup semantics:
        ``stop()`` calls ``serve.shutdown()``, tearing down all applications,
        the Serve controller, and HTTP proxy.  This is safe because a
        singleton guard ensures only one ModelServer is active at a time.
        The overhead of recreating the controller on the next ``start()``
        is ~2-5 s — negligible compared to model loading time.

    Args:
        models: List of ModelConfig instances to deploy.
        name: Ray Serve application name (default ``"default"``).
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for models to become healthy.
        verbose: If True, keep Ray Serve logging at default levels (INFO).
            If False (default), set Ray Serve replica log level to WARNING
            and disable per-request HTTP access logs via
            ``LoggingConfig``.  Note: vLLM model-loading logs are emitted
            by Ray actor subprocesses and are not affected by this flag.

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
            # Use with NeMo Curator's OpenAIClient or AsyncOpenAIClient
    """

    models: list[ModelConfig]
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False

    _started: bool = field(init=False, default=False, repr=False)

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy.

        Raises:
            RuntimeError: If another ModelServer is already active in this
                process.  Only one ModelServer can run at a time because all
                instances share the same ``/v1`` routes on the Ray Serve HTTP
                proxy.  Stop the existing server before starting a new one.
        """
        if _active_servers:
            running = ", ".join(sorted(_active_servers))
            msg = (
                f"Cannot start ModelServer '{self.name}': another ModelServer is "
                f"already active (running: {running}). Stop the existing server first."
            )
            raise RuntimeError(msg)

        # Register atexit handler so that abnormal exits
        atexit.register(self.stop)

        # Delete our app if a previous session crashed or was non-cleanly stopped.
        self._delete_stale_app()

        # Resolve the effective HTTP port.  If a Serve controller is already running,
        # we reuse its port; otherwise we find a free one.
        self._resolve_port()

        llm_configs = [m.to_llm_config() for m in self.models]
        model_names = [m.model_name or m.model_identifier for m in self.models]
        logger.info(f"Starting Ray Serve with models: {model_names} on port {self.port}")

        # LoggingConfig is applied inside the Ray Serve replica actors via serve.run().
        logging_config = None
        if not self.verbose:
            logging_config = LoggingConfig(
                log_level="WARNING",
                enable_access_log=False,
            )

        # Start the Serve controller and HTTP proxy (idempotent — reuses
        # existing controller if one is already running).
        # We do this before serve.run() because serve.run() does not accept
        # http_options and would default to port 8000.
        serve.start(http_options={"port": self.port})

        app = build_openai_app({"llm_configs": llm_configs})
        try:
            serve.run(app, name=self.name, blocking=False, logging_config=logging_config)
            self._wait_for_healthy()
        except Exception:
            # Clean up the partially-deployed app so GPUs / resources are
            # released rather than left dangling.
            self._cleanup_failed_deploy()
            raise

        _active_servers.add(self.name)
        self._started = True
        logger.info(f"Ray Serve is ready at {self.endpoint}")

    def stop(self) -> None:
        """Shut down Ray Serve (all applications, controller, and HTTP proxy)."""
        if not self._started:
            return
        logger.info("Shutting down Ray Serve")
        try:
            serve.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("serve.shutdown() failed (cluster may already be gone)")

        _active_servers.discard(self.name)
        self._started = False

        logger.info("Ray Serve stopped")

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models."""
        return f"http://localhost:{self.port}/v1"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_port(self) -> None:
        """Determine the effective HTTP port.

        If a Serve controller is already running (e.g. from another
        ModelServer or a previous session), reuse its port — Ray Serve
        binds the HTTP proxy once and silently ignores subsequent
        ``serve.start()`` calls with a different port.

        If no controller is running, find a free port starting from
        ``self.port``.
        """
        controller_port = self._get_controller_port()
        if controller_port is not None:
            if controller_port != self.port:
                logger.info(
                    f"Serve controller already running on port {controller_port}, "
                    f"using that instead of requested port {self.port}"
                )
            self.port = controller_port
        else:
            self.port = get_free_port(self.port)

    def _get_controller_port(self) -> int | None:
        """Read the HTTP port from the running Serve controller, if any.

        Uses Ray Serve's internal ``_get_global_client`` to query the
        controller's HTTP config.  Returns ``None`` if no controller is
        running.
        """
        try:
            from ray.serve.context import _get_global_client

            client = _get_global_client(_health_check_controller=True)
        except Exception:  # noqa: BLE001
            return None
        else:
            return client.http_config.port

    def _delete_stale_app(self) -> None:
        """Delete our app if it already exists from a previous session.

        Only the application matching ``self.name`` is deleted — other
        applications on the same cluster are left untouched.
        """
        from ray import serve

        try:
            status = serve.status()
        except Exception:  # noqa: BLE001
            return

        if self.name not in status.applications:
            return

        logger.info(f"Found existing Serve application '{self.name}', deleting before redeploying")

        try:
            serve.delete(self.name, _blocking=True)
        except Exception:  # noqa: BLE001
            logger.warning(f"Failed to delete existing Serve application '{self.name}'")

    def _cleanup_failed_deploy(self) -> None:
        """Best-effort cleanup after a failed deploy (e.g. health check timeout).

        Shuts down Ray Serve so that GPU memory and other resources held by
        partially-deployed replicas are released.
        """
        from ray import serve

        try:
            serve.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("Cleanup: serve.shutdown() failed after failed deploy")

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
