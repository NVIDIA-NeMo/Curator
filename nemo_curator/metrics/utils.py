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

import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request

import psutil
import requests
import yaml
from loguru import logger
from ray.dashboard.modules.metrics.install_and_start_prometheus import (
    download_file,
    get_prometheus_download_url,
    get_prometheus_filename,
)

from nemo_curator.metrics.constants import (
    DEFAULT_NEMO_CURATOR_METRICS_PATH,
    GRAFANA_DASHBOARD_YAML_TEMPLATE,
    GRAFANA_DATASOURCE_YAML_TEMPLATE,
    GRAFANA_INI_TEMPLATE,
    GRAFANA_PID_FILE,
    GRAFANA_PORT_FILE,
    GRAFANA_VERSION,
    PROMETHEUS_PID_FILE,
    PROMETHEUS_PORT_FILE,
    PROMETHEUS_YAML_TEMPLATE,
)
from nemo_curator.utils.file_utils import tar_safe_extract


def _resolve_metrics_dir(metrics_dir: str | None) -> str:
    """Resolve the metrics directory, defaulting to DEFAULT_NEMO_CURATOR_METRICS_PATH."""
    return metrics_dir if metrics_dir is not None else DEFAULT_NEMO_CURATOR_METRICS_PATH


def _is_process_running_from_pidfile(pid_file_path: str) -> bool:
    """Check if a process is running by reading its PID from a file and verifying it's alive."""
    if not os.path.isfile(pid_file_path):
        return False
    try:
        with open(pid_file_path) as f:
            pid = int(f.read().strip())
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def download_and_extract_prometheus(
    metrics_dir: str | None = None,
    os_type=None,  # noqa: ANN001
    architecture=None,  # noqa: ANN001
    prometheus_version=None,  # noqa: ANN001
) -> str:
    """Download the prometheus tarball and extract it to the metrics directory."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    file_name, _ = get_prometheus_filename(os_type, architecture, prometheus_version)
    download_url = get_prometheus_download_url(os_type, architecture, prometheus_version)
    file_path = os.path.join(metrics_dir, file_name)
    directory_path = file_path.rstrip(".tar.gz")  # noqa: B005
    if not os.path.isdir(directory_path):
        # Download the prometheus tarball
        if not os.path.isfile(file_path):
            downloaded = download_file(download_url, file_path)
            if not downloaded:
                error_msg = "Failed to download Prometheus."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        # Extract the tar.gz file in the metrics directory
        with tarfile.open(file_path) as tar:
            tar_safe_extract(tar, metrics_dir)
    return directory_path


def is_prometheus_running(metrics_dir: str | None = None) -> bool:
    """Check if Prometheus is currently running for this metrics instance."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    pid_file_path = os.path.join(metrics_dir, PROMETHEUS_PID_FILE)
    return _is_process_running_from_pidfile(pid_file_path)


def is_grafana_running(metrics_dir: str | None = None) -> bool:
    """Check if Grafana is currently running for this metrics instance."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    pid_file_path = os.path.join(metrics_dir, GRAFANA_PID_FILE)
    return _is_process_running_from_pidfile(pid_file_path)


def get_prometheus_port(metrics_dir: str | None = None) -> int:
    """Get the port number that Prometheus is running on by reading the port file."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    port_file_path = os.path.join(metrics_dir, PROMETHEUS_PORT_FILE)
    try:
        with open(port_file_path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 9090  # Default port


def run_prometheus(prometheus_dir: str, prometheus_web_port: int, metrics_dir: str | None = None) -> None:
    """Run the prometheus server."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)

    # Write the prometheus.yml file to the metrics directory
    prometheus_config_path = os.path.join(metrics_dir, "prometheus.yml")
    with open(prometheus_config_path, "w") as f:
        f.write(PROMETHEUS_YAML_TEMPLATE)

    prometheus_cmd = [
        f"{prometheus_dir}/prometheus",
        "--config.file",
        str(prometheus_config_path),
        "--web.enable-lifecycle",
        f"--web.listen-address=:{prometheus_web_port}",
    ]

    try:
        # Start prometheus in the background with log file
        prometheus_log_file = os.path.join(metrics_dir, "prometheus.log")
        prometheus_err_file = os.path.join(metrics_dir, "prometheus.err")
        with (
            open(prometheus_log_file, "a") as log_f,
            open(prometheus_err_file, "a") as err_f,
        ):
            process = subprocess.Popen(  # noqa: S603
                prometheus_cmd,
                stdout=log_f,
                stderr=err_f,
            )

        # Write PID and port files for instance tracking
        with open(os.path.join(metrics_dir, PROMETHEUS_PID_FILE), "w") as f:
            f.write(str(process.pid))
        with open(os.path.join(metrics_dir, PROMETHEUS_PORT_FILE), "w") as f:
            f.write(str(prometheus_web_port))

        logger.info("Prometheus has started.")
    except Exception as error:
        error_msg = f"Failed to start Prometheus: {error}"
        logger.error(error_msg)
        raise


def download_grafana(metrics_dir: str | None = None) -> str:
    """Download the grafana tarball and extract it to the metrics directory."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)

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
            tar_safe_extract(tar, metrics_dir)

    return grafana_extract_dir


def launch_grafana(
    grafana_dir: str, grafana_ini_path: str, grafana_web_port: int, metrics_dir: str | None = None
) -> None:
    """Launch the grafana server."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)

    grafana_cmd = [
        os.path.join(grafana_dir, "bin", "grafana-server"),
        "--config",
        grafana_ini_path,
        f"--homepath={grafana_dir}",
        "web",
    ]

    grafana_log_file = os.path.join(metrics_dir, "grafana.log")
    grafana_err_file = os.path.join(metrics_dir, "grafana.err")

    with (
        open(grafana_log_file, "a") as log_f,
        open(grafana_err_file, "a") as err_f,
    ):
        process = subprocess.Popen(  # noqa: S603
            grafana_cmd,
            stdout=log_f,
            stderr=err_f,
        )

    # Write PID and port files for instance tracking
    with open(os.path.join(metrics_dir, GRAFANA_PID_FILE), "w") as f:
        f.write(str(process.pid))
    with open(os.path.join(metrics_dir, GRAFANA_PORT_FILE), "w") as f:
        f.write(str(grafana_web_port))

    logger.info("Grafana has started.")


def write_grafana_configs(grafana_web_port: int, prometheus_web_port: int, metrics_dir: str | None = None) -> str:
    """Write the grafana configs to the grafana directory."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)

    grafana_config_root = os.path.join(metrics_dir, "grafana")
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
    xenna_dashboard_src = os.path.join(os.path.dirname(__file__), "xenna_grafana_dashboard.json")
    xenna_dashboard_src = os.path.abspath(xenna_dashboard_src)
    xenna_dashboard_dst = os.path.join(dashboards_path, "xenna_grafana_dashboard.json")
    if os.path.isfile(xenna_dashboard_src) and not os.path.isfile(xenna_dashboard_dst):
        shutil.copy(xenna_dashboard_src, xenna_dashboard_dst)

    # Generate Ray's default Grafana dashboards
    _write_ray_default_dashboards(dashboards_path)

    return grafana_ini_path


def _write_ray_default_dashboards(dashboards_path: str) -> None:
    """Generate and write Ray's default Grafana dashboards to the dashboards directory."""
    try:
        from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
            generate_data_grafana_dashboard,
            generate_default_grafana_dashboard,
            generate_serve_deployment_grafana_dashboard,
            generate_serve_grafana_dashboard,
        )
    except ImportError:
        logger.debug("Could not import Ray dashboard factory. Skipping Ray default dashboards.")
        return

    dashboard_generators = {
        "default": generate_default_grafana_dashboard,
        "serve": generate_serve_grafana_dashboard,
        "serve_deployment": generate_serve_deployment_grafana_dashboard,
        "data": generate_data_grafana_dashboard,
    }

    for _name, _generator in dashboard_generators.items():
        dashboard_file = os.path.join(dashboards_path, f"ray_{_name}_dashboard.json")
        if not os.path.isfile(dashboard_file):
            try:
                content, _uid = _generator()
                with open(dashboard_file, "w") as f:
                    f.write(content)
                logger.debug(f"Wrote Ray {_name} dashboard to {dashboard_file} ")
            except Exception:  # noqa: BLE001
                logger.debug(f"Failed to generate Ray {_name} dashboard, skipping.")


def _get_all_discovery_paths(prometheus_config: dict) -> list[str]:
    """Extract all file paths from all file_sd_configs entries in the prometheus config."""
    paths = []
    for entry in prometheus_config["scrape_configs"][0].get("file_sd_configs", []):
        files = entry.get("files", [])
        if files:
            paths.extend(files)
    return paths


def add_ray_prometheus_metrics_service_discovery(ray_temp_dir: str, metrics_dir: str | None = None) -> None:
    """Add the ray prometheus metrics service discovery to the prometheus config."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    prometheus_config_path = os.path.join(metrics_dir, "prometheus.yml")
    with open(prometheus_config_path) as prometheus_config_file:
        prometheus_config = yaml.safe_load(prometheus_config_file)

    ray_prom_metrics_service_discovery_path = os.path.join(ray_temp_dir, "prom_metrics_service_discovery.json")
    existing_paths = _get_all_discovery_paths(prometheus_config)

    if ray_prom_metrics_service_discovery_path not in existing_paths:
        # Add a new file_sd_configs entry with this path
        file_sd_configs = prometheus_config["scrape_configs"][0].get("file_sd_configs", [])
        # If empty or None, initialize as list
        if not file_sd_configs:
            file_sd_configs = []
            prometheus_config["scrape_configs"][0]["file_sd_configs"] = file_sd_configs
        file_sd_configs.append({"files": [ray_prom_metrics_service_discovery_path]})
        with open(prometheus_config_path, "w") as f:
            yaml.dump(prometheus_config, f)

        # Reload prometheus to pick up the new config
        prometheus_port = get_prometheus_port(metrics_dir)
        requests.post(f"http://localhost:{prometheus_port}/-/reload", timeout=5)


def remove_ray_prometheus_metrics_service_discovery(ray_temp_dir: str, metrics_dir: str | None = None) -> None:
    """Remove the ray prometheus metrics service discovery from the prometheus config."""
    metrics_dir = _resolve_metrics_dir(metrics_dir)
    prometheus_config_path = os.path.join(metrics_dir, "prometheus.yml")

    if not os.path.isfile(prometheus_config_path):
        return

    try:
        with open(prometheus_config_path) as prometheus_config_file:
            prometheus_config = yaml.safe_load(prometheus_config_file)
    except (yaml.YAMLError, OSError):
        return

    ray_prom_metrics_service_discovery_path = os.path.join(ray_temp_dir, "prom_metrics_service_discovery.json")
    file_sd_configs = prometheus_config["scrape_configs"][0].get("file_sd_configs", [])
    if not file_sd_configs:
        return

    # Remove entries that contain only this path, or remove path from multi-path entries
    updated = []
    changed = False
    for entry in file_sd_configs:
        files = entry.get("files", [])
        if ray_prom_metrics_service_discovery_path in files:
            changed = True
            files = [f for f in files if f != ray_prom_metrics_service_discovery_path]
            if files:
                updated.append({"files": files})
            # else: drop the entire entry
        else:
            updated.append(entry)

    if not changed:
        return

    prometheus_config["scrape_configs"][0]["file_sd_configs"] = updated
    with open(prometheus_config_path, "w") as f:
        yaml.dump(prometheus_config, f)

    # Reload prometheus if it's running
    if is_prometheus_running(metrics_dir):
        try:
            prometheus_port = get_prometheus_port(metrics_dir)
            requests.post(f"http://localhost:{prometheus_port}/-/reload", timeout=5)
        except requests.RequestException:
            logger.debug("Could not reload Prometheus after removing service discovery entry.")
