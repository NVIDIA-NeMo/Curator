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
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

import yaml
from loguru import logger

from nemo_curator.core.constants import (
    DEFAULT_RAY_CLIENT_SERVER_PORT,
    DEFAULT_RAY_DASHBOARD_HOST,
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_METRICS_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_RAY_TEMP_DIR,
)
from nemo_curator.core.utils import (
    check_ray_responsive,
    get_free_port,
    init_cluster,
)
from nemo_curator.metrics.utils import (
    add_ray_prometheus_metrics_service_discovery,
    is_grafana_running,
    is_prometheus_running,
    remove_ray_prometheus_metrics_service_discovery,
)


@dataclass
class RayClient:
    """
    This class is used to setup the Ray cluster and configure metrics integration.

    If the specified ports are already in use, it will find the next available port and use that.


    Args:
        ray_port: The port number of the Ray GCS.
        ray_dashboard_port: The port number of the Ray dashboard.
        ray_temp_dir: The temporary directory to use for Ray.
        include_dashboard: Whether to include dashboard integration. If true, adds Ray metrics service discovery.
        ray_metrics_port: The port number of the Ray metrics.
        ray_dashboard_host: The host of the Ray dashboard.
        num_gpus: The number of GPUs to use.
        num_cpus: The number of CPUs to use.
        object_store_memory: The amount of memory to use for the object store.
        enable_object_spilling: Whether to enable object spilling.
        ray_stdouterr_capture_file: The file to capture stdout/stderr to.
        metrics_dir: The directory for Prometheus/Grafana metrics data. If None, uses the per-user default.

    Note:
        To start monitoring services (Prometheus and Grafana), use the standalone
        start_prometheus_grafana.py script separately.
    """

    ray_port: int = DEFAULT_RAY_PORT
    ray_dashboard_port: int = DEFAULT_RAY_DASHBOARD_PORT
    ray_client_server_port: int = DEFAULT_RAY_CLIENT_SERVER_PORT
    ray_temp_dir: str = DEFAULT_RAY_TEMP_DIR
    include_dashboard: bool = True
    ray_metrics_port: int = DEFAULT_RAY_METRICS_PORT
    ray_dashboard_host: str = DEFAULT_RAY_DASHBOARD_HOST
    num_gpus: int | None = None
    num_cpus: int | None = None
    object_store_memory: int | None = None
    enable_object_spilling: bool = False
    ray_stdouterr_capture_file: str | None = None
    metrics_dir: str | None = None

    ray_process: subprocess.Popen | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.ray_stdouterr_capture_file and os.path.exists(self.ray_stdouterr_capture_file):
            msg = f"Capture file {self.ray_stdouterr_capture_file} already exists."
            raise FileExistsError(msg)

    def start(self) -> None:
        """Start the Ray cluster if not already started, optionally capturing stdout/stderr to a file."""

        # register atexit handler to stop the Ray cluster when the program exits
        atexit.register(self.stop)

        if self.include_dashboard:
            # Add Ray metrics service discovery to existing Prometheus configuration
            if is_prometheus_running(self.metrics_dir) and is_grafana_running(self.metrics_dir):
                try:
                    add_ray_prometheus_metrics_service_discovery(self.ray_temp_dir, self.metrics_dir)
                except Exception as e:  # noqa: BLE001
                    msg = f"Failed to add Ray metrics service discovery: {e}"
                    logger.warning(msg)
            else:
                metrics_dir_hint = f" with --metrics_dir={self.metrics_dir}" if self.metrics_dir else ""
                msg = (
                    "No monitoring services are running. "
                    "Please run the `start_prometheus_grafana.py` "
                    f"script from nemo_curator/metrics folder{metrics_dir_hint} to setup monitoring services separately."
                )
                logger.warning(msg)

        # Use the RAY_ADDRESS environment variable to determine if Ray is already running.
        # If a Ray cluster is not running:
        #   RAY_ADDRESS will be set below when the Ray cluster is started and self.ray_process
        #   will be assigned the cluster process
        # If a Ray cluster is already running:
        #   RAY_ADDRESS will have been set prior to calling start(), presumably by a user starting
        #   it externally, which means a cluster was already running and self.ray_process will be None.
        #
        # Note that the stop() method will stop the cluster only if it was started here and
        # self.ray_process was assigned, otherwise it leaves it running with the assumption it
        # was started externally and should not be stopped.
        if os.environ.get("RAY_ADDRESS"):
            logger.info("Ray is already running. Skipping the setup.")
        else:
            # If the port is not provided, it will get the next free port. If the user provided the port, it will check if the port is free.
            self.ray_dashboard_port = get_free_port(
                self.ray_dashboard_port, get_next_free_port=(self.ray_dashboard_port == DEFAULT_RAY_DASHBOARD_PORT)
            )
            self.ray_metrics_port = get_free_port(
                self.ray_metrics_port, get_next_free_port=(self.ray_metrics_port == DEFAULT_RAY_METRICS_PORT)
            )
            self.ray_port = get_free_port(self.ray_port, get_next_free_port=(self.ray_port == DEFAULT_RAY_PORT))
            self.ray_client_server_port = get_free_port(
                self.ray_client_server_port,
                get_next_free_port=(self.ray_client_server_port == DEFAULT_RAY_CLIENT_SERVER_PORT),
            )
            ip_address = socket.gethostbyname(socket.gethostname())

            self.ray_process = init_cluster(
                ray_port=self.ray_port,
                ray_temp_dir=self.ray_temp_dir,
                ray_dashboard_port=self.ray_dashboard_port,
                ray_metrics_port=self.ray_metrics_port,
                ray_client_server_port=self.ray_client_server_port,
                ray_dashboard_host=self.ray_dashboard_host,
                num_gpus=self.num_gpus,
                num_cpus=self.num_cpus,
                object_store_memory=self.object_store_memory,
                enable_object_spilling=self.enable_object_spilling,
                block=True,
                ip_address=ip_address,
                stdouterr_capture_file=self.ray_stdouterr_capture_file,
            )
            # Set environment variable for RAY_ADDRESS
            os.environ["RAY_ADDRESS"] = f"{ip_address}:{self.ray_port}"
            # Verify that Ray cluster actually started successfully
            if not check_ray_responsive():
                self.stop()  # Clean up the process we just started
                msg = "Ray cluster did not become responsive in time. Please check the logs for more information."
                raise RuntimeError(msg)

    def stop(self) -> None:
        # Remove Ray metrics service discovery entry from prometheus config
        if self.include_dashboard:
            try:
                remove_ray_prometheus_metrics_service_discovery(self.ray_temp_dir, self.metrics_dir)
            except (OSError, KeyError, yaml.YAMLError):
                logger.debug("Could not remove Ray metrics service discovery during shutdown.")

        if self.ray_process:
            # Kill the entire process group to ensure child processes are terminated
            try:
                os.killpg(os.getpgid(self.ray_process.pid), signal.SIGTERM)
                self.ray_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination doesn't work
                try:
                    os.killpg(os.getpgid(self.ray_process.pid), signal.SIGKILL)
                    self.ray_process.wait()
                except (ProcessLookupError, OSError):
                    # Process group not found or process group already terminated
                    pass
            except (ProcessLookupError, OSError):
                # Process group not found or process group already terminated
                pass
            # Reset the environment variable for RAY_ADDRESS
            os.environ.pop("RAY_ADDRESS", None)
            # Currently there is no good way of stopping a particular Ray cluster. https://github.com/ray-project/ray/issues/54989
            # We kill the Ray GCS process to stop the cluster, but still we have some Ray processes running.
            msg = "NeMo Curator has stopped the Ray cluster it started by killing the Ray GCS process. "
            msg += "It is advised to wait for a few seconds before running any Ray commands to ensure Ray can cleanup other processes."
            msg += f"If you are seeing any Ray commands like `ray status` failing, please ensure {self.ray_temp_dir}/ray_current_cluster has correct information."
            logger.info(msg)
            # Clear the process to prevent double execution (atexit handler)
            self.ray_process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# --------------------------------------------------------------------------- #
# SLURM helpers
# --------------------------------------------------------------------------- #


def _find_ray_binary() -> str:
    """Locate the ``ray`` CLI in the active Python environment."""
    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which("ray")
    if found:
        return found
    msg = "Could not find the `ray` binary. Make sure Ray is installed in the active Python environment."
    raise FileNotFoundError(msg)


def _expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand a SLURM node-list expression into individual hostnames."""
    scontrol = shutil.which("scontrol")
    if scontrol:
        try:
            result = subprocess.run(  # noqa: S603
                [scontrol, "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                check=True,
            )
            nodes = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
            if nodes:
                return nodes
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return [nodelist]


# --------------------------------------------------------------------------- #
# SlurmRayClient
# --------------------------------------------------------------------------- #


@dataclass
class SlurmRayClient(RayClient):
    """RayClient extended for multi-node SLURM jobs.

    On single-node SLURM jobs (or when not running under SLURM at all),
    behaves identically to :class:`RayClient`.

    On multi-node jobs, additionally starts Ray workers on remote nodes
    via ``srun`` and waits for every node to register with the head
    before returning from :meth:`start`.

    Usage::

        from nemo_curator.core.client import SlurmRayClient

        with SlurmRayClient() as client:
            pipeline.run(executor=XennaExecutor())

    Parameters
    ----------
    worker_connect_timeout_s:
        Maximum seconds to wait for all worker nodes to join after the
        head is up.  Raises ``TimeoutError`` if exceeded.
    env_setup_cmd:
        Shell snippet executed *before* ``ray start`` on every remote
        node (e.g. ``"source /path/to/venv/bin/activate"``).
    cleanup_on_start:
        If *True*, run ``ray stop --force`` on every allocated node
        before starting the cluster.
    """

    worker_connect_timeout_s: int = 300
    env_setup_cmd: str | None = None
    cleanup_on_start: bool = True

    ray_dashboard_host: str = "0.0.0.0"  # noqa: S104

    _worker_procs: list[subprocess.Popen] = field(init=False, default_factory=list, repr=False)
    _slurm_nodes: list[str] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._detect_slurm_resources()

    def _detect_slurm_resources(self) -> None:
        """Auto-detect per-node CPU/GPU counts from SLURM env vars when not set explicitly."""
        if self.num_cpus is None:
            slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
            if slurm_cpus:
                self.num_cpus = int(slurm_cpus)

        if self.num_gpus is None:
            slurm_gpus = os.environ.get("SLURM_GPUS_ON_NODE")
            if slurm_gpus:
                self.num_gpus = int(slurm_gpus)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the Ray cluster, adding worker nodes when running under SLURM."""
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if not slurm_job_id:
            logger.warning("SLURM_JOB_ID not set — falling back to single-node RayClient behaviour")
            super().start()
            return

        nodelist = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())
        self._slurm_nodes = _expand_slurm_nodelist(nodelist)

        logger.info(
            f"SlurmRayClient: job {slurm_job_id}, {len(self._slurm_nodes)} node(s), "
            f"head={self._slurm_nodes[0]}, cpus/node={self.num_cpus}, gpus/node={self.num_gpus}"
        )

        if self.cleanup_on_start:
            self._cleanup_stale_ray()

        # Start Ray head on the current node (reuses all RayClient logic:
        # port selection, metrics, dashboard, RAY_ADDRESS, responsiveness check)
        super().start()

        # Multi-node: launch workers on every additional node
        if len(self._slurm_nodes) > 1:
            srun_bin = shutil.which("srun")
            if srun_bin is None:
                msg = (
                    f"Multi-node SLURM job ({len(self._slurm_nodes)} nodes) but "
                    "`srun` is not on PATH. Cannot launch Ray workers on remote nodes."
                )
                raise OSError(msg)
            self._start_workers(srun_bin)
            self._wait_for_workers()

    def stop(self) -> None:
        """Tear down workers, then stop the head via the parent.

        Safe to call multiple times — subsequent calls are no-ops for
        already-cleaned resources.
        """
        import contextlib

        # 1. Kill all worker srun subprocesses we spawned
        for proc in self._worker_procs:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001, PERF203
                with contextlib.suppress(Exception):
                    proc.kill()
        self._worker_procs.clear()

        # 2. Run `ray stop --force` on every worker node
        if self._slurm_nodes:
            self._stop_remote_ray()

        # 3. Stop head (kills ray_process, clears RAY_ADDRESS)
        super().stop()

    # ------------------------------------------------------------------ #
    # Worker management
    # ------------------------------------------------------------------ #

    def _start_workers(self, srun_bin: str) -> None:
        head_addr = os.environ.get("RAY_ADDRESS", f"{socket.gethostbyname(socket.gethostname())}:{self.ray_port}")
        ray_bin = _find_ray_binary()

        worker_nodes = self._slurm_nodes[1:]
        logger.info(f"Starting Ray workers on {len(worker_nodes)} node(s): {worker_nodes}")

        for node in worker_nodes:
            ray_cmd = [
                ray_bin,
                "start",
                "--address",
                head_addr,
                "--temp-dir",
                self.ray_temp_dir,
                "--block",
                "--disable-usage-stats",
            ]
            if self.num_gpus is not None:
                ray_cmd.extend(["--num-gpus", str(self.num_gpus)])
            if self.num_cpus is not None:
                ray_cmd.extend(["--num-cpus", str(self.num_cpus)])

            if self.env_setup_cmd:
                raw = " ".join(ray_cmd)
                wrapped: list[str] = ["bash", "-c", f"{self.env_setup_cmd} && {raw}"]
            else:
                wrapped = ray_cmd

            full = [srun_bin, "--nodes=1", "--ntasks=1", "-w", node, "--overlap", *wrapped]
            logger.debug(f"Worker cmd ({node}): {' '.join(full)}")
            proc = subprocess.Popen(  # noqa: S603
                full,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._worker_procs.append(proc)
            time.sleep(1)

    def _wait_for_workers(self) -> None:
        """Block until every allocated node is alive in the Ray cluster.

        Raises ``TimeoutError`` (after tearing everything down) if not all
        nodes join within ``worker_connect_timeout_s``.  Also fails early
        if any worker subprocess exits unexpectedly.
        """
        import ray as _ray

        expected = len(self._slurm_nodes)
        deadline = time.time() + self.worker_connect_timeout_s

        _ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
        try:
            while True:
                # Fail fast if any worker srun process already died
                dead = [p for p in self._worker_procs if p.poll() is not None]
                if dead:
                    codes = [p.returncode for p in dead]
                    logger.error(f"{len(dead)} worker process(es) exited early with codes {codes}")
                    self.stop()
                    msg = (
                        f"{len(dead)} worker process(es) died before joining the cluster "
                        f"(exit codes: {codes}). Check srun/Ray logs for details."
                    )
                    raise RuntimeError(msg)

                alive = [n for n in _ray.nodes() if n.get("Alive")]
                if len(alive) >= expected:
                    total_cpus = sum(n.get("Resources", {}).get("CPU", 0) for n in alive)
                    total_gpus = sum(n.get("Resources", {}).get("GPU", 0) for n in alive)
                    logger.info(
                        f"All {expected} node(s) connected — "
                        f"total CPUs: {total_cpus:.0f}, total GPUs: {total_gpus:.0f}"
                    )
                    return

                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.error(
                        f"Timeout: only {len(alive)}/{expected} node(s) connected "
                        f"after {self.worker_connect_timeout_s}s. Killing everything."
                    )
                    self.stop()
                    msg = (
                        f"Timed out after {self.worker_connect_timeout_s}s: "
                        f"only {len(alive)}/{expected} node(s) connected. Cluster torn down."
                    )
                    raise TimeoutError(msg)

                logger.info(f"Waiting for workers: {len(alive)}/{expected} ({remaining:.0f}s left)")
                time.sleep(min(5, remaining))
        finally:
            _ray.shutdown()

    # ------------------------------------------------------------------ #
    # Cleanup helpers
    # ------------------------------------------------------------------ #

    def _run_on_node(self, node: str, cmd: list[str], timeout: int = 30) -> None:
        """Execute *cmd* on *node*: directly for head, via srun otherwise."""
        if node == self._slurm_nodes[0]:
            full = self._wrap_with_env(cmd)
        else:
            srun_bin = shutil.which("srun")
            if not srun_bin:
                return
            full = [srun_bin, "--nodes=1", "--ntasks=1", "-w", node, "--overlap", *self._wrap_with_env(cmd)]

        import contextlib

        with contextlib.suppress(Exception):
            subprocess.run(full, capture_output=True, timeout=timeout, check=False)  # noqa: S603

    def _wrap_with_env(self, cmd: list[str]) -> list[str]:
        if self.env_setup_cmd:
            return ["bash", "-c", f"{self.env_setup_cmd} && {' '.join(cmd)}"]
        return cmd

    def _cleanup_stale_ray(self) -> None:
        logger.info("Cleaning up stale Ray processes on all allocated nodes …")
        ray_bin = _find_ray_binary()
        for node in self._slurm_nodes:
            self._run_on_node(node, [ray_bin, "stop", "--force"])

    def _stop_remote_ray(self) -> None:
        ray_bin = _find_ray_binary()
        for node in self._slurm_nodes[1:]:
            self._run_on_node(node, [ray_bin, "stop", "--force"])
