import os
import re
import socket
import subprocess
import sys
from typing import TYPE_CHECKING

import psutil
import ray
from loguru import logger

from ray_curator.core.constants import DEFAULT_RAY_AUTOSCALER_METRIC_PORT, DEFAULT_RAY_DASHBOARD_METRIC_PORT

if TYPE_CHECKING:
    import loguru


def is_prometheus_running() -> bool:
    return any(proc.info["name"].lower() == "prometheus" for proc in psutil.process_iter(["name"]))


def is_grafana_running() -> bool:
    return any(proc.info["name"].lower() == "grafana" for proc in psutil.process_iter(["name"]))


def get_next_free_port(start_port: int) -> int:
    for port in range(start_port, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # SO_REUSEADDR to avoid TIME_WAIT issues on some OSes
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
                # If bind succeeds, port is free
                return port  # noqa: TRY300
            except OSError:
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


def init_or_connect_to_cluster(  # noqa: PLR0913
    executor: str,
    ray_port: int,
    ray_temp_dir: str,
    ray_dashboard_port: int,
    ray_metrics_port: int,
    ray_dashboard_host: str,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
    enable_object_spilling: bool = False,
) -> None:
    """Initialize a new local Ray cluster or connects to an existing one."""
    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )

    if executor == "Xenna":
        from cosmos_xenna.ray_utils.cluster import API_LIMIT

        # We need to set this env var to avoid ray from setting CUDA_VISIBLE_DEVICES.
        # We set these manually in Xenna because we allocate the gpus manually instead of relying on ray's mechanisms.
        # This will *only* get picked up from here if the cluster is started from this script. In the case of previously
        # existing clusters, this needs to be set in the processes that set up the cluster.
        os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "0"
        # These need to be set to allow listing debug info about more than 10k actors.
        os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = str(API_LIMIT)
        os.environ["RAY_MAX_LIMIT_FROM_DATA_SOURCE"] = str(API_LIMIT)
        os.environ["XENNA_RAY_METRICS_PORT"] = str(ray_metrics_port)

    ip_address = socket.gethostbyname(socket.gethostname())
    ray_command = ["ray", "start", "--head"]
    ray_command.extend(["--node-ip-address", ip_address])
    ray_command.extend(["--port", str(ray_port)])
    ray_command.extend(["--metrics-export-port", str(ray_metrics_port)])
    ray_command.extend(["--dashboard-host", ray_dashboard_host])
    ray_command.extend(["--dashboard-port", str(ray_dashboard_port)])
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

    os.environ["DASHBOARD_METRIC_PORT"] = str(get_next_free_port(DEFAULT_RAY_DASHBOARD_METRIC_PORT))
    os.environ["AUTOSCALER_METRIC_PORT"] = str(get_next_free_port(DEFAULT_RAY_AUTOSCALER_METRIC_PORT))
    proc = subprocess.Popen(ray_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    out, err = proc.communicate()
    logger.info(f"Ray start command: {' '.join(ray_command)}")
    logger.info(f"Ray start output: {out.decode('utf-8')}")
    logger.info(f"Ray start error: {err.decode('utf-8')}")
    if proc.returncode != 0:
        logger.error(f"Ray failed to start. Error: {err.decode('utf-8')}")
        sys.exit(1)
    else:
        os.environ["RAY_ADDRESS"] = f"{ip_address}:{ray_port}"
        logger.info("Ray started successfully.")


def get_prometheus_port() -> int:
    result = subprocess.run(["ps", "-eo", "args"], check=False, capture_output=True, text=True)  # noqa: S603, S607

    port = None
    for line in result.stdout.splitlines():
        if "prometheus" in line:
            print(line)
            match = re.search(r"--web\.listen-address=:(\d+)", line)
            if match:
                port = match.group(1)
                break

    return port or 9090
