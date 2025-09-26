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

import os
import socket
import subprocess
from typing import TYPE_CHECKING

import ray
from loguru import logger

from nemo_curator.core.constants import (
    DEFAULT_RAY_AUTOSCALER_METRIC_PORT,
    DEFAULT_RAY_DASHBOARD_METRIC_PORT,
    DEFAULT_RAY_MAX_WORKER_PORT,
    DEFAULT_RAY_MIN_WORKER_PORT,
)

if TYPE_CHECKING:
    import loguru


def get_free_port(start_port: int, get_next_free_port: bool = True) -> int:
    """Checks if start_port is free.
    If not, it will get the next free port starting from start_port if get_next_free_port is True.
    Else, it will raise an error if the free port is not equal to start_port.
    """
    for port in range(start_port, 65535):
        if port >= DEFAULT_RAY_MIN_WORKER_PORT and port <= DEFAULT_RAY_MAX_WORKER_PORT:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # SO_REUSEADDR to avoid TIME_WAIT issues on some OSes
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
                # If bind succeeds, port is free
                return port  # noqa: TRY300
            except OSError:
                if not get_next_free_port and port == start_port:
                    msg = f"Port {start_port} is already in use. Please provide a different port."
                    raise RuntimeError(msg)  # noqa: B904
                continue
    msg = f"No free port found between {start_port} and 65535"
    raise RuntimeError(msg)


def _logger_custom_serializer(
    _: "loguru.Logger",
) -> None:
    return None


def _logger_custom_deserializer(
    _: None,
) -> "loguru.Logger":
    # Initialize a default logger
    return logger


def init_cluster(  # noqa: PLR0913
    ray_port: int,
    ray_temp_dir: str,
    ray_dashboard_port: int,
    ray_metrics_port: int,
    ray_client_server_port: int,
    ray_dashboard_host: str,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
    enable_object_spilling: bool = False,
    block: bool = True,
    ip_address: str | None = None,
) -> subprocess.Popen:
    """Initialize a new local Ray cluster or connects to an existing one."""
    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )

    ip_address = ip_address or socket.gethostbyname(socket.gethostname())
    ray_command = ["ray", "start", "--head"]
    ray_command.extend(["--node-ip-address", ip_address])
    ray_command.extend(["--port", str(ray_port)])
    ray_command.extend(["--metrics-export-port", str(ray_metrics_port)])
    ray_command.extend(["--dashboard-host", ray_dashboard_host])
    ray_command.extend(["--dashboard-port", str(ray_dashboard_port)])
    ray_command.extend(["--ray-client-server-port", str(ray_client_server_port)])
    ray_command.extend(["--temp-dir", ray_temp_dir])
    ray_command.extend(["--disable-usage-stats"])
    if enable_object_spilling:
        ray_command.extend(
            [
                "--system-config",
                '{"local_fs_capacity_threshold": 0.95, "object_spilling_config": "{ "type": "filesystem", "params": {"directory_path": "/tmp/ray_spill", "buffer_size": 1000000 } }"}',
            ]
        )
    if num_gpus:
        ray_command.extend(["--num-gpus", str(num_gpus)])
    if num_cpus:
        ray_command.extend(["--num-cpus", str(num_cpus)])
    if block:
        ray_command.extend(["--block"])

    # We need to set these env vars to ensure that metrics of ray dashboard and autoscaler are available for various different clusters.
    os.environ["DASHBOARD_METRIC_PORT"] = str(get_free_port(DEFAULT_RAY_DASHBOARD_METRIC_PORT))
    os.environ["AUTOSCALER_METRIC_PORT"] = str(get_free_port(DEFAULT_RAY_AUTOSCALER_METRIC_PORT))

    # We set some env vars for Xenna here. This is only used for Xenna clusters.
    os.environ["XENNA_RAY_METRICS_PORT"] = str(ray_metrics_port)
    os.environ["XENNA_RESPECT_CUDA_VISIBLE_DEVICES"] = "1"

    proc = subprocess.Popen(ray_command, shell=False)  # noqa: S603
    logger.info(f"Ray start command: {' '.join(ray_command)}")
    return proc
