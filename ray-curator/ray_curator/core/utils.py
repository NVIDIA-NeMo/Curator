import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
from typing import TYPE_CHECKING

import psutil
import ray
import requests
import yaml
from loguru import logger

from ray_curator.core.constants import (
    DEFAULT_NEMO_CURATOR_METRICS_PATH,
    DEFAULT_RAY_AUTOSCALER_METRIC_PORT,
    DEFAULT_RAY_DASHBOARD_METRIC_PORT,
    GRAFANA_DASHBOARD_YAML_TEMPLATE,
    GRAFANA_DATASOURCE_YAML_TEMPLATE,
    GRAFANA_INI_TEMPLATE,
    GRAFANA_VERSION,
    PROMETHEUS_YAML_TEMPLATE,
)

if TYPE_CHECKING:
    import loguru


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    """Safely extract a *trusted* tarball.

    The implementation first tries to leverage the `filter="data"` argument
    that became available in Python 3.12, which tells the runtime to only
    extract regular files and directories (i.e. no symlinks or unusual
    members).  When running on an older Python version, we manually guard
    against path-traversal attacks by ensuring that every member would be
    placed inside *path* before delegating to ``extractall``.
    """

    try:
        # Python 3.12+: built-in protection via the ``filter`` argument.
        tar.extractall(path=path, filter="data")
        return  # noqa: TRY300
    except TypeError:
        # Older Python - fall back to manual checks.
        pass

    def _is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            msg = "Attempted Path Traversal in Tar File"
            raise RuntimeError(msg)

    tar.extractall(path=path)  # noqa: S202


def move_and_extract_prometheus(file_name: str) -> None:
    """
    Move and extract the prometheus file to the default nemo curator metrics path.
    """
    # Get absolute path to the file and move it to NEMO_CURATOR_METRICS_PATH, filename is a tar.gz file
    file_path = os.path.abspath(file_name)
    os.makedirs(DEFAULT_NEMO_CURATOR_METRICS_PATH, exist_ok=True)
    # Move the file to NEMO_CURATOR_METRICS_PATH
    shutil.move(file_path, os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, file_name))
    # Extract the tar.gz file in the NEMO_CURATOR_METRICS_PATH

    file_path = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, file_name)
    with tarfile.open(file_path) as tar:
        _safe_extract(tar, DEFAULT_NEMO_CURATOR_METRICS_PATH)

    return file_path


def run_prometheus(ray_temp_dir: str, prometheus_dir: str, prometheus_web_port: int) -> None:
    """
    Run the prometheus server.
    """
    # Write the prometheus.yml file to the NEMO_CURATOR_METRICS_PATH directory
    prometheus_config_path = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, "prometheus.yml")
    with open(prometheus_config_path, "w") as f:
        f.write(
            PROMETHEUS_YAML_TEMPLATE.format(
                service_discovery_path=os.path.join(ray_temp_dir, "prom_metrics_service_discovery.json")
            )
        )

    prometheus_cmd = [
        f"{prometheus_dir}/prometheus",
        "--config.file",
        str(prometheus_config_path),
        "--web.enable-lifecycle",
        f"--web.listen-address=:{prometheus_web_port}",
    ]

    try:
        # Start prometheus in the background with log file
        prometheus_log_file = os.path.join(
            DEFAULT_NEMO_CURATOR_METRICS_PATH,
            "prometheus.log",
        )
        prometheus_err_file = os.path.join(
            DEFAULT_NEMO_CURATOR_METRICS_PATH,
            "prometheus.err",
        )
        with (
            open(prometheus_log_file, "a") as log_f,
            open(
                prometheus_err_file,
                "a",
            ) as err_f,
        ):
            subprocess.Popen(  # noqa: S603
                prometheus_cmd,
                stdout=log_f,
                stderr=err_f,
            )
        logger.info("Prometheus has started.")
    except Exception as error:
        error_msg = f"Failed to start Prometheus: {error}"
        logger.error(error_msg)
        raise


def download_grafana() -> str:
    """
    Download the grafana tarball and extract it to the default nemo curator metrics path.
    """
    # Determine download URL based on architecture
    arch = platform.machine()
    if arch not in ("x86_64", "amd64"):
        logger.warning(
            "Automatic Grafana installation is only tested on x86_64/amd64 architectures. "
            "Please install Grafana manually if the following steps fail."
        )

    grafana_version = GRAFANA_VERSION
    grafana_tar_name = f"grafana-enterprise-{grafana_version}.linux-amd64.tar.gz"
    grafana_url = f"https://dl.grafana.com/enterprise/release/{grafana_tar_name}"

    # Paths
    metrics_dir = DEFAULT_NEMO_CURATOR_METRICS_PATH
    os.makedirs(metrics_dir, exist_ok=True)

    grafana_extract_dir = os.path.join(metrics_dir, f"grafana-v{grafana_version}")
    grafana_tar_path = os.path.join(metrics_dir, grafana_tar_name)

    if not os.path.isdir(grafana_extract_dir):
        # Download if tar not present
        if not os.path.isfile(grafana_tar_path):
            logger.info(f"Downloading Grafana from {grafana_url} ...")
            urllib.request.urlretrieve(grafana_url, grafana_tar_path)  # noqa: S310

        # Extract
        logger.info("Extracting Grafana archive ...")
        with tarfile.open(grafana_tar_path, "r:gz") as tar:
            _safe_extract(tar, metrics_dir)

    return grafana_extract_dir


def launch_grafana(grafana_dir: str, grafana_ini_path: str) -> None:
    """
    Launch the grafana server.
    """
    # -------------------
    # Launch Grafana
    # -------------------
    grafana_cmd = [
        os.path.join(grafana_dir, "bin", "grafana-server"),
        "--config",
        grafana_ini_path,
        f"--homepath={grafana_dir}",
        "web",
    ]

    grafana_log_file = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, "grafana.log")
    grafana_err_file = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, "grafana.err")

    with (
        open(grafana_log_file, "a") as log_f,
        open(
            grafana_err_file,
            "a",
        ) as err_f,
    ):
        subprocess.Popen(  # noqa: S603
            grafana_cmd,
            stdout=log_f,
            stderr=err_f,
        )
    logger.info("Grafana has started.")


def write_grafana_configs(grafana_web_port: int, prometheus_web_port: int) -> str:
    """
    Write the grafana configs to the grafana directory.
    """
    # -------------------
    # Provisioning setup
    # -------------------
    grafana_config_root = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, "grafana")
    provisioning_path = os.path.join(grafana_config_root, "provisioning")
    dashboards_path = os.path.join(grafana_config_root, "dashboards")
    datasources_path = os.path.join(provisioning_path, "datasources")
    dashboards_prov_path = os.path.join(provisioning_path, "dashboards")

    for p in [grafana_config_root, provisioning_path, datasources_path, dashboards_path, dashboards_prov_path]:
        os.makedirs(p, exist_ok=True)

    # Write grafana.ini
    grafana_ini_path = os.path.join(grafana_config_root, "grafana.ini")
    with open(grafana_ini_path, "w") as f:
        f.write(GRAFANA_INI_TEMPLATE.format(provisioning_path=provisioning_path, grafana_web_port=grafana_web_port))

    # Write provisioning dashboard yaml
    dashboards_yaml_path = os.path.join(dashboards_prov_path, "default.yml")
    with open(dashboards_yaml_path, "w") as f:
        f.write(GRAFANA_DASHBOARD_YAML_TEMPLATE.format(dashboards_path=dashboards_path))

    # Write datasource yaml (points to Prometheus instance we just launched)
    datasources_yaml_path = os.path.join(datasources_path, "default.yml")
    prometheus_url = f"http://localhost:{prometheus_web_port}"
    with open(datasources_yaml_path, "w") as f:
        f.write(GRAFANA_DATASOURCE_YAML_TEMPLATE.format(prometheus_url=prometheus_url))

    # Copy Xenna dashboard json if not already present
    xenna_dashboard_src = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "xenna_grafana_dashboard.json"
    )
    xenna_dashboard_src = os.path.abspath(xenna_dashboard_src)
    xenna_dashboard_dst = os.path.join(dashboards_path, "xenna_grafana_dashboard.json")
    if os.path.isfile(xenna_dashboard_src) and not os.path.isfile(xenna_dashboard_dst):
        shutil.copy(xenna_dashboard_src, xenna_dashboard_dst)
    return grafana_ini_path


def add_ray_prometheus_metrics_service_discovery(ray_temp_dir: str) -> None:
    """
    Add the ray prometheus metrics service discovery to the prometheus config.
    """
    # Check if ray_temp_dir exists in DEFAULT_NEMO_CURATOR_METRICS_PATH/prometheus.yml, if not add it
    prometheus_config_path = os.path.join(
        DEFAULT_NEMO_CURATOR_METRICS_PATH,
        "prometheus.yml",
    )
    with open(prometheus_config_path) as prometheus_config_file:
        prometheus_config = yaml.safe_load(prometheus_config_file)
    ray_prom_metrics_service_discovery_path = os.path.join(ray_temp_dir, "prom_metrics_service_discovery.json")
    if (
        ray_prom_metrics_service_discovery_path
        not in prometheus_config["scrape_configs"][0]["file_sd_configs"][0]["files"]
    ):
        prometheus_config["scrape_configs"][0]["file_sd_configs"][0]["files"].append(
            ray_prom_metrics_service_discovery_path
        )
        with open(prometheus_config_path, "w") as f:
            yaml.dump(prometheus_config, f)
    # Get prometheus port
    prometheus_port = get_prometheus_port()
    # Send a curl to prometheus for reloading
    requests.post(f"http://localhost:{prometheus_port}/-/reload", timeout=5)


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

    # We need to set these env vars to ensure that metrics of ray dashboard and autoscaler are available for various different clusters.
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
    result = subprocess.run(["ps", "-eo", "args"], check=False, capture_output=True, text=True)  # noqa: S603,S607

    port = None
    for line in result.stdout.splitlines():
        if "prometheus" in line:
            print(line)
            match = re.search(r"--web\.listen-address=:(\d+)", line)
            if match:
                port = match.group(1)
                break

    return port or 9090
