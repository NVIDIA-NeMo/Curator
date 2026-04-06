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
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
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
    """Expand a SLURM node-list expression into individual hostnames.

    Tries ``scontrol show hostnames`` first, then falls back to a
    pure-Python parser that handles common compact formats like
    ``prefix-[01,03-05]`` and ``node1,node2``.
    """
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
    return _parse_slurm_nodelist(nodelist)


def _parse_slurm_nodelist(nodelist: str) -> list[str]:
    """Pure-Python parser for SLURM compact nodelist notation.

    Handles formats like:
    - ``node1,node2,node3``
    - ``prefix-[01,03,05]``
    - ``prefix-[01-05]``
    - ``prefix-[01-03,07,10-12]``
    """
    import re

    nodes: list[str] = []
    for token in re.split(r",(?![^\[]*\])", nodelist):
        m = re.match(r"^(.+?)\[(.+)\]$", token)
        if not m:
            nodes.append(token)
            continue
        prefix, ranges = m.group(1), m.group(2)
        for part in ranges.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                width = len(lo)
                for n in range(int(lo), int(hi) + 1):
                    nodes.append(f"{prefix}{str(n).zfill(width)}")
            else:
                nodes.append(f"{prefix}{part}")
    return nodes if nodes else [nodelist]


# --------------------------------------------------------------------------- #
# SlurmRayClient
# --------------------------------------------------------------------------- #


@dataclass
class SlurmRayClient(RayClient):
    """RayClient extended for multi-node SLURM jobs.

    On single-node SLURM jobs (or when not running under SLURM at all),
    behaves identically to :class:`RayClient`.

    On multi-node jobs, the script must be launched on **every** node
    (e.g. via ``srun --ntasks-per-node=1``).  Each process calls
    ``SlurmRayClient``, which inspects ``SLURM_NODEID`` to determine
    its role:

    - **Head (SLURM_NODEID=0)**: starts the Ray head, waits for all
      workers to connect, then returns from :meth:`start` so the
      pipeline can run.
    - **Workers (SLURM_NODEID>0)**: start a Ray worker that connects
      to the head and **block until the cluster is torn down**.  When
      the head stops Ray (after the pipeline finishes), the worker
      process exits cleanly with ``sys.exit(0)``.

    This is analogous to how ``torchrun`` works: the same script is
    launched on every node and each process discovers its role from the
    environment.

    Example ``sbatch`` script::

        #!/bin/bash
        #SBATCH --nodes=4
        #SBATCH --ntasks-per-node=1
        #SBATCH --gpus-per-node=8

        srun --ntasks-per-node=1 \\
            --container-image=nvcr.io/nvidia/nemo-curator:26.02 \\
            --container-mounts="/lustre:/lustre" \\
            bash -c "source .venv/bin/activate && python my_pipeline.py"

    For bare-metal (no container) setups, the same pattern works::

        #!/bin/bash
        #SBATCH --nodes=4
        #SBATCH --ntasks-per-node=1
        #SBATCH --gpus-per-node=8

        srun python my_pipeline.py

    If ``RAY_ADDRESS`` is set before :meth:`start` is called,
    ``SlurmRayClient`` connects to the existing cluster without
    starting or stopping anything.

    Parameters
    ----------
    worker_connect_timeout_s:
        Maximum seconds to wait for all worker nodes to join after the
        head is up.  Raises ``TimeoutError`` if exceeded.
    cleanup_on_start:
        If *True*, run ``ray stop --force`` on the local node before
        starting Ray.  Helps clear stale processes from previous runs.
    """

    worker_connect_timeout_s: int = 300
    cleanup_on_start: bool = True

    ray_dashboard_host: str = "0.0.0.0"  # noqa: S104

    _slurm_nodes: list[str] = field(init=False, default_factory=list, repr=False)
    _manages_cluster: bool = field(init=False, default=False, repr=False)

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
        """Start the Ray cluster, with role detection on multi-node SLURM jobs.

        If ``RAY_ADDRESS`` is already set, connects to the existing
        cluster without starting a new head or launching workers.

        On multi-node jobs, worker processes (``SLURM_NODEID > 0``)
        block here until the cluster is torn down, then exit with
        ``sys.exit(0)``.  Only the head (``SLURM_NODEID = 0``) returns
        from this method.
        """
        if os.environ.get("RAY_ADDRESS"):
            logger.info(
                f"RAY_ADDRESS already set ({os.environ['RAY_ADDRESS']}). "
                "Connecting to existing Ray cluster — skipping head/worker startup."
            )
            super().start()
            return

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if not slurm_job_id:
            logger.warning("SLURM_JOB_ID not set — falling back to single-node RayClient behaviour")
            super().start()
            return

        nodelist = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())
        self._slurm_nodes = _expand_slurm_nodelist(nodelist)
        self._manages_cluster = True
        node_id = int(os.environ.get("SLURM_NODEID", "0"))

        logger.info(
            f"SlurmRayClient: job {slurm_job_id}, {len(self._slurm_nodes)} node(s), "
            f"SLURM_NODEID={node_id}, head={self._slurm_nodes[0]}, "
            f"cpus/node={self.num_cpus}, gpus/node={self.num_gpus}"
        )

        if self.cleanup_on_start:
            self._cleanup_local_ray()

        if len(self._slurm_nodes) <= 1 or node_id == 0:
            # Head node — start Ray head (super().start() selects the actual port via get_free_port)
            super().start()
            # Broadcast the actual port the head chose so workers don't have to guess.
            # Workers may be on different physical nodes and cannot call get_free_port on
            # the head, so we write the port to a shared Lustre file keyed on job ID.
            if len(self._slurm_nodes) > 1:
                self._write_head_port(slurm_job_id)
                self._wait_for_workers()
        else:
            # Worker node — read the port the head actually chose, then connect.
            head_ip = socket.gethostbyname(self._slurm_nodes[0])
            actual_port = self._read_head_port(slurm_job_id)
            self.ray_port = actual_port
            logger.info(f"SlurmRayClient worker {node_id}: connecting to head at {head_ip}:{self.ray_port}")
            sys.exit(self._run_as_worker(head_ip))

    def stop(self) -> None:
        """Stop the Ray head.  Workers detect the head's death and exit on their own.

        Safe to call multiple times.  Does not stop an externally
        managed cluster (one discovered via ``RAY_ADDRESS``).
        """
        if self._manages_cluster:
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id:
                port_file = self._head_port_file(slurm_job_id)
                with contextlib.suppress(FileNotFoundError):
                    os.remove(port_file)
                    logger.info(f"SlurmRayClient: removed port file {port_file}")
        super().stop()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _head_port_file(self, slurm_job_id: str) -> str:
        """Return path to the shared port-broadcast file for this job.

        Must be on a filesystem visible to ALL nodes (Lustre, not /tmp).
        Uses env var ``RAY_PORT_BROADCAST_DIR`` if set, otherwise falls back to
        ``/tmp`` (works on single-node or when /tmp is shared, e.g. via NFS).
        """
        broadcast_dir = os.environ.get("RAY_PORT_BROADCAST_DIR", "/tmp")  # noqa: S108
        os.makedirs(broadcast_dir, exist_ok=True)
        return os.path.join(broadcast_dir, f"ray_head_port_{slurm_job_id}")

    def _write_head_port(self, slurm_job_id: str) -> None:
        """Write the actual Ray GCS port to a shared file so workers can read it.

        Uses an atomic write-then-rename so workers never observe an empty or
        partially-written file (important on Lustre / NFS where open() truncates
        before write() completes).
        """
        port_file = self._head_port_file(slurm_job_id)
        broadcast_dir = os.path.dirname(port_file)
        with tempfile.NamedTemporaryFile(mode="w", dir=broadcast_dir, delete=False) as f:
            tmp_path = f.name
            f.write(str(self.ray_port))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, port_file)  # atomic on POSIX
        logger.info(f"SlurmRayClient head: wrote port {self.ray_port} to {port_file}")

    def _read_head_port(self, slurm_job_id: str, timeout_s: int = 600) -> int:
        """Wait for the head to write its port file and return the port number."""
        port_file = self._head_port_file(slurm_job_id)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if os.path.exists(port_file):
                try:
                    with open(port_file) as f:
                        port = int(f.read().strip())
                except (ValueError, OSError):
                    pass  # file may be partially written; retry
                else:
                    logger.info(f"SlurmRayClient worker: read head port {port} from {port_file}")
                    return port
            time.sleep(2)
        msg = f"Timed out waiting for head port file {port_file} after {timeout_s}s"
        raise TimeoutError(msg)

    def _run_as_worker(self, head_ip: str) -> int:
        """Start a Ray worker that connects to *head_ip* and block until the cluster is torn down.

        Returns the exit code of ``ray start --block`` so the caller can pass it to ``sys.exit``.
        Exit code 0 means the cluster was torn down cleanly; non-zero indicates an error.
        """
        ray_bin = _find_ray_binary()
        cmd = [
            ray_bin,
            "start",
            "--address",
            f"{head_ip}:{self.ray_port}",
            "--temp-dir",
            self.ray_temp_dir,
            "--block",
            "--disable-usage-stats",
        ]
        if self.num_gpus is not None:
            cmd.extend(["--num-gpus", str(self.num_gpus)])
        if self.num_cpus is not None:
            cmd.extend(["--num-cpus", str(self.num_cpus)])

        logger.info(f"Ray worker starting: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)  # noqa: S603
        logger.info(f"Ray worker exited with code {result.returncode}")
        return result.returncode

    def _cleanup_local_ray(self) -> None:
        """Stop any stale Ray processes on the local node."""
        with contextlib.suppress(Exception):
            ray_bin = _find_ray_binary()
            subprocess.run([ray_bin, "stop", "--force"], capture_output=True, timeout=30, check=False)  # noqa: S603

    @staticmethod
    def _ray_init_with_timeout(address: str, timeout_s: int = 120) -> None:
        """Call ``ray.init(address=...)`` with a SIGALRM-based timeout.

        ``ray.init`` can hang indefinitely if the GCS is slow or unstable
        after a multi-job start.  We use SIGALRM (Linux/macOS only) to raise
        a ``TimeoutError`` if the call blocks longer than *timeout_s* seconds.

        Falls back to an unguarded ``ray.init`` when called from a non-main
        thread, where SIGALRM is unavailable.
        """
        import ray as _ray

        if threading.current_thread() is not threading.main_thread():
            logger.warning("SIGALRM unavailable outside main thread — calling ray.init without timeout")
            _ray.init(address=address, ignore_reinit_error=True)
            return

        def _handler(_signum: int, _frame: object) -> None:
            msg = (
                f"ray.init(address={address!r}) timed out after {timeout_s}s — "
                "GCS may be unresponsive; the job will exit and can be resubmitted."
            )
            raise TimeoutError(msg)

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)
        try:
            _ray.init(address=address, ignore_reinit_error=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _wait_for_workers(self) -> None:
        """Block until every allocated node is alive in the Ray cluster.

        Raises ``TimeoutError`` (after tearing everything down) if not
        all nodes join within ``worker_connect_timeout_s``.
        """
        import ray as _ray

        expected = len(self._slurm_nodes)
        deadline = time.time() + self.worker_connect_timeout_s

        self._ray_init_with_timeout(os.environ["RAY_ADDRESS"], timeout_s=120)
        try:
            while True:
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
                        f"after {self.worker_connect_timeout_s}s."
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
