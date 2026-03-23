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

import http
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.core.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SGLANG_PORT
from nemo_curator.core.utils import get_free_port

# Track which ports are currently used by active SGLang servers.
# ``is_sglang_active()`` checks this set so that Pipeline.run() can detect
# potential GPU resource contention.
_active_sglang_servers: set[int] = set()

# Keep a reference to the active server instance for pipeline.py to inspect
# (e.g. to check ``allows_colocated_gpu_stages`` and ``model.nnodes``).
_active_server: "SGLangInferenceServer | None" = None


def is_sglang_active() -> bool:
    """Check whether any SGLangInferenceServer is currently running in this process."""
    return bool(_active_sglang_servers)


def get_active_sglang_server() -> "SGLangInferenceServer | None":
    """Return the active SGLangInferenceServer instance, or None if not running."""
    return _active_server


# ---------------------------------------------------------------------------
# Ray Actor for multi-node deployments (nnodes > 1)
# ---------------------------------------------------------------------------


def _make_sglang_actor_class() -> type:
    """Lazily define the Ray remote class to avoid importing Ray at module load."""
    import ray

    @ray.remote
    class _SGLangServerActor:
        """Ray Actor that launches a single SGLang server process on a remote node.

        Used only for multi-node deployments (``nnodes > 1``).  Ray places the actor
        on the correct node via ``NodeAffinitySchedulingStrategy`` and uses
        ``num_gpus`` to track GPU usage, making it visible to Ray's resource scheduler.
        """

        def __init__(self, command: list[str], env: dict[str, str], verbose: bool = False) -> None:
            stdout = None if verbose else subprocess.DEVNULL
            stderr = None if verbose else subprocess.DEVNULL
            self.proc = subprocess.Popen(  # noqa: S603
                command,
                env=env,
                start_new_session=True,
                stdout=stdout,
                stderr=stderr,
            )

        def is_alive(self) -> bool:
            """Return True if the SGLang subprocess is still running."""
            return self.proc.poll() is None

        def stop(self) -> None:
            """Gracefully terminate, then force-kill if needed."""
            if self.proc.poll() is not None:
                return
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    self.proc.wait()
                except (ProcessLookupError, OSError):
                    pass
            except (ProcessLookupError, OSError):
                pass

    return _SGLangServerActor


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SGLangModelConfig:
    """Configuration for a single model to be served via SGLang.

    Args:
        model_path: HuggingFace model ID or local path.
        model_name: API-facing model name clients use in requests.
            Defaults to ``model_path``.
        dp_size: Data parallelism — number of replicas, load-balanced internally
            by SGLang.  Equivalent to Ray Serve ``min_replicas`` / ``max_replicas``
            in the ``InferenceServer`` paradigm.
        tp_size: Tensor parallelism — number of GPUs per replica.
        nnodes: Total number of nodes.  Use ``1`` for a single-node subprocess
            (no Ray required).  Use ``> 1`` for a multi-node deployment where
            SGLang processes are launched via Ray Actors (NCCL handles inter-node
            GPU communication).
        dist_port: NCCL rendezvous port (``--dist-init-addr``) for multi-node
            deployments.  Must be free on the head node.
        mem_fraction_static: Fraction of GPU memory reserved for the KV cache
            (``--mem-fraction-static``).  ``None`` uses SGLang's default.
        gpu_ids: GPU indices to assign to the subprocess via ``CUDA_VISIBLE_DEVICES``.
            Only effective for single-node (``nnodes=1``) deployments.
            For multi-node, Ray controls GPU assignment through Actor ``num_gpus``.
        server_kwargs: Arbitrary extra CLI flags passed to ``sglang.launch_server``.
            Keys map to ``--key value`` pairs (underscores converted to hyphens).
        python_executable: Python interpreter used to launch the SGLang server.
            Defaults to ``sys.executable`` (the current interpreter).  Override
            this when sglang is installed in a different virtual environment from
            the one running Curator (e.g. when vllm and sglang cannot share an
            environment due to conflicting ``flashinfer-python`` pins)::

                config = SGLangModelConfig(
                    model_path="...",
                    python_executable="/opt/sglang-venv/bin/python",
                )
    """

    model_path: str
    model_name: str | None = None
    dp_size: int = 1
    tp_size: int = 1
    nnodes: int = 1
    dist_port: int = 50000
    mem_fraction_static: float | None = None
    gpu_ids: list[int] | None = None
    server_kwargs: dict[str, Any] = field(default_factory=dict)
    python_executable: str = field(
        default_factory=lambda: os.environ.get("SGLANG_PYTHON_EXECUTABLE", sys.executable)
    )

    def _build_command(self, port: int, node_rank: int = 0, dist_init_addr: str | None = None) -> list[str]:
        """Build the ``sglang.launch_server`` CLI command for this node.

        Args:
            port: HTTP port (only meaningful on rank-0 / head node).
            node_rank: This node's rank (0 = head, 1+ = workers).
            dist_init_addr: ``host:port`` for NCCL rendezvous (multi-node only).
        """
        cmd = [
            self.python_executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--tp-size",
            str(self.tp_size),
            "--dp-size",
            str(self.dp_size),
        ]

        # Only the head node (rank 0) needs the HTTP port
        if node_rank == 0:
            cmd += ["--port", str(port)]

        if self.model_name:
            cmd += ["--served-model-name", self.model_name]

        if self.mem_fraction_static is not None:
            cmd += ["--mem-fraction-static", str(self.mem_fraction_static)]

        if self.nnodes > 1:
            cmd += [
                "--nnodes",
                str(self.nnodes),
                "--node-rank",
                str(node_rank),
            ]
            if dist_init_addr:
                cmd += ["--dist-init-addr", dist_init_addr]

        # Append arbitrary extra kwargs
        for key, value in self.server_kwargs.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd += [flag, str(value)]

        return cmd


@dataclass
class SGLangInferenceServer:
    """Serve a single model via SGLang with an OpenAI-compatible endpoint.

    For single-node deployments (``model.nnodes == 1``), SGLang is launched as a
    direct subprocess — no Ray is required.  For multi-node deployments
    (``model.nnodes > 1``), Ray Actors spawn SGLang processes on the correct nodes;
    Ray then tracks GPU usage automatically, enabling safe coexistence with
    ``RayDataExecutor``.

    Multi-instance (data parallelism) equivalence:
        - Ray Serve: ``deployment_config={"autoscaling_config": {"min_replicas": N}}``
        - SGLang: ``SGLangModelConfig(dp_size=N)``

    GPU coexistence:
        - Single-node subprocess: invisible to Ray's scheduler.  Pipeline stages
          requiring GPUs are blocked by default.  Set ``gpu_ids`` on the model config
          and ``allows_colocated_gpu_stages=True`` to opt in (user manages partitioning).
        - Multi-node Ray Actors: GPUs are tracked via ``num_gpus`` per actor, enabling
          ``RayDataExecutor`` to schedule on remaining GPUs automatically.

    Example::

        from nemo_curator.core.sglang_serve import SGLangInferenceServer, SGLangModelConfig

        config = SGLangModelConfig(
            model_path="google/gemma-3-27b-it",
            tp_size=4,
            dp_size=2,
        )

        with SGLangInferenceServer(model=config) as server:
            print(server.endpoint)  # http://localhost:30000/v1

    Args:
        model: Model configuration.
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for the server to become healthy.
        verbose: If True, keep SGLang logging at default levels.  If False
            (default), suppress per-request logs via ``SGLANG_LOGGING_LEVEL=WARNING``.
        allows_colocated_gpu_stages: Single-node only opt-in flag.  When True,
            Pipeline.run() will not block GPU stages from running alongside this
            server.  Use together with ``model.gpu_ids`` to partition GPUs manually.
    """

    model: SGLangModelConfig
    port: int = DEFAULT_SGLANG_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False
    allows_colocated_gpu_stages: bool = False

    _process: subprocess.Popen | None = field(init=False, default=None, repr=False)
    _stderr_tempfile: Any | None = field(init=False, default=None, repr=False)
    _head_actor: Any | None = field(init=False, default=None, repr=False)
    _worker_actors: list = field(init=False, default_factory=list, repr=False)
    _started: bool = field(init=False, default=False, repr=False)

    def start(self) -> None:
        """Start the SGLang server and wait for it to become healthy.

        Raises:
            RuntimeError: If another SGLangInferenceServer is already using the
                same port, or if the server process dies before becoming healthy.
            TimeoutError: If the server does not respond within ``health_check_timeout_s``.
        """
        import atexit

        self.port = get_free_port(self.port)

        if self.port in _active_sglang_servers:
            msg = (
                f"Cannot start SGLangInferenceServer on port {self.port}: "
                "another SGLangInferenceServer is already active on this port."
            )
            raise RuntimeError(msg)

        atexit.register(self.stop)

        if self.model.nnodes > 1:
            self._start_multinode()
        else:
            self._start_singlenode()

        global _active_server  # noqa: PLW0603
        _active_sglang_servers.add(self.port)
        _active_server = self
        self._started = True
        logger.info(f"SGLang server is ready at {self.endpoint}")

    def _start_singlenode(self) -> None:
        """Launch SGLang as a local subprocess (no Ray required)."""
        cmd = self.model._build_command(port=self.port)
        env = self._build_env()

        logger.info(
            f"Starting SGLang server (single-node, dp_size={self.model.dp_size}, "
            f"tp_size={self.model.tp_size}) on port {self.port}"
        )

        stdout = None if self.verbose else subprocess.DEVNULL
        # Use a NamedTemporaryFile for stderr so that:
        # 1. Startup error output is preserved for diagnostics.
        # 2. The subprocess never blocks on a full PIPE buffer (SGLang is very
        #    verbose during initialisation, especially with large dp_size values;
        #    a PIPE deadlocks once the ~64 KB kernel buffer fills).
        self._stderr_tempfile = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+b", suffix=".log", prefix="sglang_stderr_", delete=False
        )
        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            start_new_session=True,
            stdout=stdout,
            stderr=self._stderr_tempfile,
        )

        try:
            self._wait_for_healthy()
        except Exception:
            self._cleanup_failed_start_singlenode()
            raise

    def _start_multinode(self) -> None:
        """Launch SGLang via Ray Actors across multiple nodes (nnodes > 1)."""
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from nemo_curator.backends.experimental.utils import get_head_node_id

        num_gpus_per_actor = self.model.tp_size

        logger.info(
            f"Starting SGLang server (multi-node, nnodes={self.model.nnodes}, "
            f"dp_size={self.model.dp_size}, tp_size={self.model.tp_size}) on port {self.port}"
        )

        # Find the head node and its IP
        head_node_id = get_head_node_id()
        if head_node_id is None:
            msg = "Cannot start multi-node SGLang: no Ray head node found. Ensure Ray is initialised."
            raise RuntimeError(msg)

        head_ip = None
        for node in ray.nodes():
            if node.get("NodeID") == head_node_id and node.get("Alive", False):
                head_ip = node.get("NodeManagerAddress")
                break

        if head_ip is None:
            msg = f"Could not determine IP address for Ray head node {head_node_id}."
            raise RuntimeError(msg)

        dist_init_addr = f"{head_ip}:{self.model.dist_port}"

        actor_cls = _make_sglang_actor_class()
        env = self._build_env()

        # Spawn head actor (rank 0) pinned to the head node
        head_cmd = self.model._build_command(
            port=self.port,
            node_rank=0,
            dist_init_addr=dist_init_addr,
        )
        self._head_actor = actor_cls.options(  # type: ignore[attr-defined]
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=NodeAffinitySchedulingStrategy(head_node_id, soft=False),
        ).remote(head_cmd, env, self.verbose)

        # Select worker nodes (non-head, alive)
        worker_nodes = [n for n in ray.nodes() if n.get("Alive", False) and n.get("NodeID") != head_node_id]

        needed_workers = self.model.nnodes - 1
        if len(worker_nodes) < needed_workers:
            msg = (
                f"Multi-node SGLang requires {needed_workers} worker node(s), "
                f"but only {len(worker_nodes)} non-head alive node(s) found in the Ray cluster."
            )
            # Clean up head actor before raising
            import contextlib

            with contextlib.suppress(Exception):
                ray.get(self._head_actor.stop.remote())
                ray.kill(self._head_actor)
            self._head_actor = None
            raise RuntimeError(msg)

        for rank, node in enumerate(worker_nodes[:needed_workers], start=1):
            worker_cmd = self.model._build_command(
                port=self.port,
                node_rank=rank,
                dist_init_addr=dist_init_addr,
            )
            actor = actor_cls.options(  # type: ignore[attr-defined]
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
            ).remote(worker_cmd, env, self.verbose)
            self._worker_actors.append(actor)

        try:
            self._wait_for_healthy()
        except Exception:
            self._cleanup_failed_start_multinode()
            raise

    def stop(self) -> None:
        """Shut down the SGLang server and release resources.

        For single-node: sends SIGTERM then SIGKILL to the subprocess.
        For multi-node: gracefully stops and kills each Ray Actor.
        Safe to call multiple times (idempotent).
        """
        if not self._started:
            return

        logger.info("Shutting down SGLang server")

        if self.model.nnodes > 1:
            self._stop_multinode()
        else:
            self._stop_singlenode()

        global _active_server  # noqa: PLW0603
        _active_sglang_servers.discard(self.port)
        _active_server = None
        self._started = False
        logger.info("SGLang server stopped")

    def _stop_singlenode(self) -> None:
        if self._process is None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
            except (ProcessLookupError, OSError):
                pass
        except (ProcessLookupError, OSError):
            pass
        self._process = None
        self._cleanup_stderr_tempfile()

    def _read_stderr_tempfile(self) -> str:
        """Read all content written to the stderr temp file so far."""
        if self._stderr_tempfile is None:
            return ""
        import contextlib

        content = ""
        with contextlib.suppress(Exception):
            self._stderr_tempfile.flush()
            self._stderr_tempfile.seek(0)
            content = self._stderr_tempfile.read().decode(errors="replace").strip()
        return content

    def _cleanup_stderr_tempfile(self) -> None:
        """Close and delete the stderr temp file."""
        if self._stderr_tempfile is None:
            return
        import contextlib

        path = getattr(self._stderr_tempfile, "name", None)
        with contextlib.suppress(Exception):
            self._stderr_tempfile.close()
        if path:
            with contextlib.suppress(Exception):
                os.unlink(path)
        self._stderr_tempfile = None

    def _stop_multinode(self) -> None:
        import contextlib

        import ray

        actors = []
        if self._head_actor is not None:
            actors.append(self._head_actor)
        actors.extend(self._worker_actors)

        for actor in actors:
            with contextlib.suppress(Exception):
                ray.get(actor.stop.remote(), timeout=10)
            with contextlib.suppress(Exception):
                ray.kill(actor)

        self._head_actor = None
        self._worker_actors = []

    def _cleanup_failed_start_singlenode(self) -> None:
        """Best-effort cleanup after a failed single-node start."""
        if self._process is not None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    self._process.wait()
                except (ProcessLookupError, OSError):
                    pass
            self._process = None
        self._cleanup_stderr_tempfile()

    def _cleanup_failed_start_multinode(self) -> None:
        """Best-effort cleanup after a failed multi-node start."""
        self._stop_multinode()

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served model."""
        return f"http://localhost:{self.port}/v1"

    def complementary_gpu_ids(self, total_gpus: int) -> list[int]:
        """Return the GPU IDs *not* used by this server.

        Useful for setting ``CUDA_VISIBLE_DEVICES`` before initialising Ray so that
        the pipeline executor only sees GPUs not already claimed by the SGLang
        subprocess.

        Args:
            total_gpus: Total number of GPUs on the node.

        Returns:
            List of GPU indices not in ``model.gpu_ids``.  If ``model.gpu_ids`` is
            ``None`` (server uses all GPUs), returns an empty list.
        """
        if self.model.gpu_ids is None:
            return list(range(total_gpus))
        return [i for i in range(total_gpus) if i not in self.model.gpu_ids]

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment dict."""
        env = {**os.environ}
        if not self.verbose:
            env["SGLANG_LOGGING_LEVEL"] = "WARNING"
        if self.model.gpu_ids is not None and self.model.nnodes == 1:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.model.gpu_ids)
        return env

    def _check_head_actor_alive(self) -> None:
        """Raise RuntimeError if the Ray head actor subprocess has already exited."""
        import contextlib

        import ray

        alive: bool | None = None
        with contextlib.suppress(Exception):
            alive = ray.get(self._head_actor.is_alive.remote(), timeout=2)
        if alive is False:
            msg = (
                "SGLang head actor process exited before the server became healthy. "
                "Enable verbose=True for server logs."
            )
            raise RuntimeError(msg)

    def _wait_for_healthy(self) -> None:
        """Poll the /v1/models endpoint until the server is ready.

        Uses wall-clock time to enforce the timeout accurately.  On single-node
        deployments, also checks ``_process.poll()`` for a fast-fail if the
        subprocess exits unexpectedly.
        """
        models_url = f"{self.endpoint}/models"
        deadline = time.monotonic() + self.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1

            # Fast-fail: check if single-node subprocess has already exited
            if self._process is not None and self._process.poll() is not None:
                msg = (
                    f"SGLang subprocess exited with code {self._process.returncode} "
                    "before the server became healthy."
                )
                stderr_output = self._read_stderr_tempfile()
                if stderr_output:
                    msg += f"\nServer stderr:\n{stderr_output}"
                raise RuntimeError(msg)

            # Fast-fail: check if head actor has exited (multi-node)
            if self._head_actor is not None:
                self._check_head_actor_alive()

            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    logger.info(f"SGLang server ready after {attempt} health check(s)")
                    return
            except Exception:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(1)

        msg = f"SGLang server did not become ready within {self.health_check_timeout_s}s"
        stderr_output = self._read_stderr_tempfile()
        if stderr_output:
            msg += f"\nServer stderr (last output):\n{stderr_output[-4000:]}"
        raise TimeoutError(msg)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
