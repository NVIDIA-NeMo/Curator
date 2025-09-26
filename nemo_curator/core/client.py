# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import socket
import subprocess
from dataclasses import dataclass

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
    get_free_port,
    init_cluster,
)
from nemo_curator.metrics.utils import (
    add_ray_prometheus_metrics_service_discovery,
    is_grafana_running,
    is_prometheus_running,
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
        enable_object_spilling: Whether to enable object spilling.

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
    enable_object_spilling: bool = False

    ray_process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.include_dashboard:
            # Add Ray metrics service discovery to existing Prometheus configuration
            if is_prometheus_running() and is_grafana_running():
                try:
                    add_ray_prometheus_metrics_service_discovery(self.ray_temp_dir)
                except Exception as e:
                    msg = f"Failed to add Ray metrics service discovery: {e}"
                    logger.warning(msg)
                    raise
            else:
                msg = (
                    "No monitoring services are running. "
                    "Please run the `start_prometheus_grafana.py` "
                    "script from nemo_curator/metrics folder to setup monitoring services separately."
                )
                logger.warning(msg)

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
                self.ray_port,
                self.ray_temp_dir,
                self.ray_dashboard_port,
                self.ray_metrics_port,
                self.ray_client_server_port,
                self.ray_dashboard_host,
                self.num_gpus,
                self.num_cpus,
                self.enable_object_spilling,
                block=True,
                ip_address=ip_address,
            )
            # Set environment variable for RAY_ADDRESS

            os.environ["RAY_ADDRESS"] = f"{ip_address}:{self.ray_port}"
            # Register atexit handler only when we have a ray process
            atexit.register(self.stop)

    def stop(self) -> None:
        if self.ray_process:
            self.ray_process.kill()
            self.ray_process.wait()
            # Reset the environment variable for RAY_ADDRESS
            os.environ.pop("RAY_ADDRESS", None)
            # Currently there is no good way of stopping a particular Ray cluster. https://github.com/ray-project/ray/issues/54989
            # We kill the Ray GCS process to stop the cluster, but still we have some Ray processes running.
            msg = "NeMo Curator has stopped the Ray cluster it started by killing the Ray GCS process. "
            msg += "It is advised to wait for a few seconds before running any Ray commands to ensure Ray can cleanup other processes."
            msg += "If you are seeing any Ray commands like `ray status` failing, please ensure /tmp/ray/ray_current_cluster has correct information."
            logger.info(msg)
            # Clear the process to prevent double execution (atexit handler)
            self.ray_process = None
