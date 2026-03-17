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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.serve.llm import LLMConfig

from loguru import logger

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT
from nemo_curator.core.utils import get_free_port

# Track which application names are currently managed by an InferenceServer in
# this process.  ``is_ray_serve_active()`` checks this set so that other
# parts of the codebase (e.g. Pipeline.run()) can detect potential GPU
# resource contention.
_active_servers: set[str] = set()


def is_ray_serve_active() -> bool:
    """Check whether any InferenceServer is currently running in this process."""
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
                overrides.  Merged on top of ``self.runtime_env`` so that
                quiet env vars take precedence while preserving user-provided
                keys (e.g. ``pip``, ``working_dir``).
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
        # Merge env_vars from both dicts rather than clobbering
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
        ``stop()`` calls ``serve.shutdown()``, tearing down all applications,
        the Serve controller, and HTTP proxy.  This is safe because a
        singleton guard ensures only one InferenceServer is active at a time.
        The overhead of recreating the controller on the next ``start()``
        is ~2-5 s — negligible compared to model loading time.

    Args:
        models: List of InferenceModelConfig instances to deploy.
        name: Ray Serve application name (default ``"default"``).
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for models to become healthy.
        verbose: If True, keep Ray Serve and vLLM logging at default levels.
            If False (default), suppress per-request logs from both vLLM
            (``VLLM_LOGGING_LEVEL=WARNING``) and Ray Serve access logs
            (``RAY_SERVE_LOG_TO_STDERR=0``).  Serve logs still go to
            files under the Ray session log directory.

    Example::

        from nemo_curator.core.serve import InferenceModelConfig, InferenceServer

        config = InferenceModelConfig(
            model_identifier="google/gemma-3-27b-it",
            engine_kwargs={"tensor_parallel_size": 4},
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,
                },
            },
        )

        with InferenceServer(models=[config]) as server:
            print(server.endpoint)  # http://localhost:8000/v1
            # Use with NeMo Curator's OpenAIClient or AsyncOpenAIClient
    """

    models: list[InferenceModelConfig]
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False

    _started: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.verbose:
            # Suppress driver-side Ray Serve INFO logs ("Deployment Options",
            # "Ingress Options", "Started Serve", etc.).
            logging.getLogger("ray.serve").setLevel(logging.WARNING)

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy.

        The driver connects to the Ray cluster only for the duration of
        deployment.  Once models are healthy the driver disconnects, so that
        the next ``ray.init()`` (e.g. from a pipeline executor) becomes the
        first driver-level init and its ``runtime_env`` takes effect on
        workers.  Serve actors are detached and survive the disconnect.

        Raises:
            RuntimeError: If another InferenceServer is already active in this
                process.  Only one InferenceServer can run at a time because
                Ray Serve uses a single HTTP proxy per cluster, and all
                models are deployed as a single application sharing the
                same ``/v1`` routes.  You can deploy multiple models in one
                InferenceServer (via the ``models`` list) — clients select a
                model by passing ``model="<model_name>"`` in the API
                request body.  Stop the existing server before starting a
                new one.
        """
        if _active_servers:
            running = ", ".join(sorted(_active_servers))
            msg = (
                f"Cannot start InferenceServer '{self.name}': another InferenceServer is "
                f"already active (running: {running}). Stop the existing server first."
            )
            raise RuntimeError(msg)

        # Register atexit handler so that abnormal exits
        atexit.register(self.stop)

        # Connect to the Ray cluster for deployment only.  The context manager
        # disconnects the driver on exit so that subsequent ray.init() calls
        # (from pipeline executors) are the "first" driver-level init and their
        # runtime_env actually takes effect on workers.
        # Serve actors are detached and survive the driver disconnect.
        #
        import ray

        self._reset_serve_client_cache()
        with ray.init(ignore_reinit_error=True):
            self._deploy()
        self._reset_serve_client_cache()

        _active_servers.add(self.name)
        self._started = True
        logger.info(f"Ray Serve is ready at {self.endpoint}")

    def _deploy(self) -> None:
        """Deploy models onto the connected Ray cluster (internal).

        Must be called while a Ray connection is active.
        """
        self.port = get_free_port(self.port)

        model_names = [m.model_name or m.model_identifier for m in self.models]
        logger.info(f"Starting Ray Serve with models: {model_names} on port {self.port}")

        quiet_env = self._quiet_runtime_env() if not self.verbose else None

        llm_configs = [m.to_llm_config(quiet_runtime_env=quiet_env) for m in self.models]

        build_args: dict[str, Any] = {"llm_configs": llm_configs}
        if quiet_env:
            # Suppress access logs on the OpenAI ingress deployment too.
            build_args["ingress_deployment_config"] = {
                "ray_actor_options": {"runtime_env": quiet_env},
            }

        from ray.serve.schema import LoggingConfig

        logging_config = None
        if not self.verbose:
            logging_config = LoggingConfig(
                log_level="WARNING",
                enable_access_log=False,
            )

        from ray import serve
        from ray.serve.llm import build_openai_app

        app = build_openai_app(build_args)
        # Start the Serve controller and HTTP proxy.
        # We call serve.start() before serve.run() because serve.run() does not
        # accept http_options and would default to port 8000.
        serve.start(http_options={"port": self.port}, logging_config=logging_config)

        try:
            serve.run(app, name=self.name, blocking=False, logging_config=logging_config)
            self._wait_for_healthy()
        except Exception:
            self._cleanup_failed_deploy()
            raise

    def stop(self) -> None:
        """Shut down Ray Serve (all applications, controller, and HTTP proxy).

        Reconnects to the Ray cluster to tear down Serve actors and release
        GPU memory, then disconnects.  If the cluster is already gone (e.g.
        ``RayClient`` was stopped first), the shutdown is skipped silently.
        """
        if not self._started:
            return
        logger.info("Shutting down Ray Serve")
        try:
            import ray
            from ray import serve

            self._reset_serve_client_cache()
            with ray.init(ignore_reinit_error=True):
                serve.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("serve.shutdown() failed (cluster may already be gone)")
        finally:
            self._reset_serve_client_cache()

        _active_servers.discard(self.name)
        self._started = False

        logger.info("Ray Serve stopped")

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models.

        When multiple models are deployed, clients select a model by passing
        ``model="<model_name>"`` in the request body (standard OpenAI API
        convention).  The ``/v1/models`` endpoint lists all available models.
        """
        return f"http://localhost:{self.port}/v1"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_serve_client_cache() -> None:
        """Reset Ray Serve's cached controller client.

        Ray Serve caches the controller actor handle in a module-level
        ``_global_client``.  This handle becomes stale when the driver
        disconnects and reconnects (e.g. via ``with ray.init()``).  The
        built-in staleness check only catches ``RayActorError``, not the
        "different cluster" exception that occurs across driver sessions.

        Resetting forces the next Serve API call to look up the controller
        by its well-known actor name, producing a fresh handle.

        TODO: Remove this method once https://github.com/ray-project/ray/issues/61608 is fixed.
        """
        try:
            from ray.serve.context import _set_global_client

            _set_global_client(None)
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def _quiet_runtime_env() -> dict[str, Any]:
        """Return a ``runtime_env`` dict that suppresses per-request logs.

        Works around two upstream bugs in Ray Serve (as of Ray 2.44+):

        1. **vLLM request logs** (``Added request chatcmpl-...``):
           ``_start_async_llm_engine`` creates ``AsyncLLM()`` without passing
           ``log_requests``, so it defaults to ``True``.
           Workaround: ``VLLM_LOGGING_LEVEL=WARNING``.
           TODO: Once we upgrade past Ray 2.54 (see ray-project/ray#60824),
           pass ``"enable_log_requests": False`` in ``engine_kwargs`` instead
           and remove the ``VLLM_LOGGING_LEVEL`` env var workaround.

        2. **Ray Serve access logs** (``POST /v1/... 200 Xms``):
           ``configure_component_logger()`` only adds the access-log filter
           to the *file* handler, not the stderr stream handler, so
           ``LoggingConfig(enable_access_log=False)`` has no effect on
           console output.  Workaround: ``RAY_SERVE_LOG_TO_STDERR=0``
           (logs still go to files under the Ray session log directory).
           TODO: Ray might fix this in the future.
        """
        return {
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "WARNING",
                "RAY_SERVE_LOG_TO_STDERR": "0",
            },
        }

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
