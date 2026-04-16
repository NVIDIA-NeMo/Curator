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
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.core.serve.base import BaseModelConfig, BaseServerConfig, InferenceBackend
from nemo_curator.core.serve.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT
from nemo_curator.core.serve.dynamo.config import DynamoServerConfig
from nemo_curator.core.serve.ray_serve.config import RayServeServerConfig

# Track which application names are currently managed by an InferenceServer in
# this process so other components can detect possible GPU contention.
_active_servers: set[str] = set()


def is_inference_server_active() -> bool:
    """Check whether any inference server is currently running in this process."""
    return bool(_active_servers)


@dataclass
class InferenceServer:
    """Serve one or more models behind a typed backend config."""

    models: list[BaseModelConfig]
    backend: BaseServerConfig = field(default_factory=RayServeServerConfig)
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False

    _started: bool = field(init=False, default=False, repr=False)
    _backend_impl: InferenceBackend | None = field(init=False, default=None, repr=False)
    _host: str = field(init=False, default="localhost", repr=False)

    def __post_init__(self) -> None:
        self._validate_model_configs()
        if not self.verbose:
            logging.getLogger("ray.serve").setLevel(logging.WARNING)

    def _validate_model_configs(self) -> None:
        """Check every model is accepted by the backend and that all models share one concrete type."""
        accepted = self.backend.model_configs
        wrong = sorted({type(m).__name__ for m in self.models if not isinstance(m, accepted)})
        if wrong:
            accepted_names = ", ".join(c.__name__ for c in accepted) or "<none>"
            msg = f"{type(self.backend).__name__} accepts {accepted_names}, but got {', '.join(wrong)}."
            raise TypeError(msg)

        model_types = {type(m) for m in self.models}
        if len(model_types) > 1:
            names = sorted(t.__name__ for t in model_types)
            msg = f"All models in one InferenceServer must be the same config type; got {', '.join(names)}."
            raise TypeError(msg)

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy."""
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
            atexit.unregister(self.stop)
            raise

        _active_servers.add(self.name)
        self._started = True
        logger.info(f"Inference server is ready at {self.endpoint}")

    def _create_backend(self) -> InferenceBackend:
        if isinstance(self.backend, RayServeServerConfig):
            from nemo_curator.core.serve.ray_serve.backend import RayServeBackend

            return RayServeBackend(self)
        if isinstance(self.backend, DynamoServerConfig):
            from nemo_curator.core.serve.dynamo.backend import DynamoBackend

            return DynamoBackend(self)
        msg = f"Unknown backend config type: {type(self.backend)!r}"
        raise TypeError(msg)

    def stop(self) -> None:
        """Shut down the active inference backend and release resources."""
        if not self._started:
            return

        try:
            if self._backend_impl is not None:
                self._backend_impl.stop()
        finally:
            self._backend_impl = None
            _active_servers.discard(self.name)
            self._started = False
            atexit.unregister(self.stop)

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models."""
        return f"http://{self._host}:{self.port}/v1"

    def _wait_for_healthy(self) -> None:
        """Poll ``/v1/models`` until all expected models appear in the response."""
        expected = {model.resolved_model_name for model in self.models}
        models_url = f"{self.endpoint}/models"
        deadline = time.monotonic() + self.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    body = json.loads(resp.read())
                    model_ids = {model["id"] for model in body.get("data", [])}
                    if expected.issubset(model_ids):
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
