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

"""Subprocess-based vLLM server for cross-environment isolation.

Unlike :class:`~nemo_curator.core.serve.InferenceServer` (which deploys
models via Ray Serve and requires vLLM in the *same* Python environment),
:class:`VLLMSubprocessServer` launches ``vllm serve`` as a **separate OS
process**.  This is useful when vLLM and Curator live in different Python
environments — for example, a base Docker image with vLLM/Qwen deps and a
virtualenv with Curator.
"""

import atexit
import contextlib
import http
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT


@dataclass
class VLLMSubprocessServer:
    """Launch vLLM as a subprocess and wait for it to become healthy.

    The server exposes an OpenAI-compatible ``/v1`` endpoint that Curator's
    :class:`~nemo_curator.models.client.openai_client.OpenAIClient` (or any
    OpenAI-compatible client) can query.

    Args:
        model: HuggingFace model ID or local path passed to ``vllm serve``.
        port: Port for the OpenAI-compatible HTTP server.
        host: Bind address for vLLM (default ``"0.0.0.0"``).
        python_executable: Python binary used to launch vLLM.  Defaults to
            ``"python"`` (resolved via ``PATH``).  Set to an explicit path
            (e.g. ``"/usr/bin/python3"``) to use a specific interpreter —
            useful when Curator runs in a virtualenv but vLLM is installed
            in the system Python.
        extra_args: Additional CLI arguments appended to ``vllm serve``
            (e.g. ``["--dtype", "bfloat16", "--max-model-len", "65536"]``).
        health_check_timeout_s: Seconds to wait for ``/v1/models`` to
            return HTTP 200 before raising :class:`TimeoutError`.
        log_file: Optional path to redirect vLLM stdout/stderr.  If
            ``None``, vLLM output goes to the parent's stdout/stderr.

    Example::

        from nemo_curator.core.vllm_server import VLLMSubprocessServer

        server = VLLMSubprocessServer(
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            port=8200,
            python_executable="/usr/bin/python3",
            extra_args=[
                "--dtype", "bfloat16",
                "--max-model-len", "65536",
                "--allowed-local-media-path", "/",
                "-tp", "2",
            ],
        )

        with server:
            print(server.endpoint)  # http://localhost:8200/v1
            # Use with Curator's OpenAIClient or AsyncOpenAIClient
    """

    model: str
    port: int = DEFAULT_SERVE_PORT
    host: str = "0.0.0.0"  # noqa: S104
    python_executable: str = "python"
    extra_args: list[str] = field(default_factory=list)
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    log_file: str | None = None

    _process: subprocess.Popen | None = field(init=False, default=None, repr=False)
    _started: bool = field(init=False, default=False, repr=False)
    _log_fh: Any = field(init=False, default=None, repr=False)

    def start(self) -> None:
        """Start the vLLM server and block until it is healthy.

        Raises:
            TimeoutError: If the server does not respond within
                *health_check_timeout_s* seconds.
            RuntimeError: If the vLLM process exits before becoming healthy.
        """
        atexit.register(self.stop)

        cmd = [
            self.python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--port",
            str(self.port),
            "--host",
            self.host,
            *self.extra_args,
        ]

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        # Redirect output to log file if requested, otherwise inherit
        # parent's stdout/stderr so the user can see vLLM's output.
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_path, "w")  # noqa: SIM115
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                shell=False,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                shell=False,
                start_new_session=True,
            )

        try:
            self._wait_for_healthy()
        except Exception:
            self.stop()
            raise

        self._started = True
        logger.info(f"vLLM server is ready at {self.endpoint}")

    def stop(self) -> None:
        """Stop the vLLM subprocess.

        Sends ``SIGTERM`` to the process group and waits up to 10 seconds
        for a graceful shutdown.  Falls back to ``SIGKILL`` if needed.
        Safe to call multiple times or before ``start()``.
        """
        if self._process is not None:
            logger.info("Shutting down vLLM server")
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    self._process.wait()
            except (ProcessLookupError, OSError):
                logger.debug("vLLM process already terminated")
            self._process = None
            logger.info("vLLM server stopped")

        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.close()
            self._log_fh = None

        self._started = False

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL (e.g. ``http://localhost:8200/v1``)."""
        return f"http://localhost:{self.port}/v1"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_healthy(self) -> None:
        """Poll ``/v1/models`` until the server responds with HTTP 200.

        Also monitors whether the vLLM process exits prematurely (e.g.
        OOM, missing model) and raises immediately instead of waiting
        for the full timeout.
        """
        models_url = f"{self.endpoint}/models"
        deadline = time.monotonic() + self.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1

            # Check if the process died before we got a healthy response
            if self._process is not None and self._process.poll() is not None:
                rc = self._process.returncode
                msg = f"vLLM process exited prematurely with return code {rc}"
                raise RuntimeError(msg)

            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    logger.info(f"vLLM server ready after {attempt} health check(s)")
                    return
            except Exception:  # noqa: BLE001
                logger.debug(f"Health check attempt {attempt} failed, retrying...")

            time.sleep(1)

        msg = f"vLLM server did not become ready within {self.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
