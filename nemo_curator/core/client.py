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
import signal
import socket
import subprocess
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
