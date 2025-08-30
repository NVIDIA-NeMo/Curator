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
import platform
import re
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
    GRAFANA_VERSION,
    PROMETHEUS_YAML_TEMPLATE,
)
from nemo_curator.utils.file_utils import tar_safe_extract


def download_and_extract_prometheus(os_type=None, architecture=None, prometheus_version=None) -> str:  # noqa: ANN001
    """Download the prometheus tarball and extract it to the default nemo curator metrics path."""
    file_name, _ = get_prometheus_filename(os_type, architecture, prometheus_version)
    download_url = get_prometheus_download_url(os_type, architecture, prometheus_version)
    file_path = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, file_name)
    directory_path = file_path.rstrip(".tar.gz")  # noqa: B005
    if not os.path.isdir(directory_path):
        # Download the prometheus tarball
        if not os.path.isfile(file_path):
            downloaded = download_file(download_url, file_path)
            if not downloaded:
                error_msg = "Failed to download Prometheus."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        # Extract the tar.gz file in the NEMO_CURATOR_METRICS_PATH
        with tarfile.open(file_path) as tar:
            tar_safe_extract(tar, DEFAULT_NEMO_CURATOR_METRICS_PATH)
    return directory_path


def is_prometheus_running() -> bool:
    """Check if Prometheus is currently running."""
    return any(proc.info["name"].lower() == "prometheus" for proc in psutil.process_iter(["name"]))


def is_grafana_running() -> bool:
    """Check if Grafana is currently running."""
    return any(proc.info["name"].lower() == "grafana" for proc in psutil.process_iter(["name"]))


def get_prometheus_port() -> int:
    """Get the port number that Prometheus is running on."""
    result = subprocess.run(["ps", "-ef", "|", "grep", "prometheus"], check=False, capture_output=True, text=True)  # noqa: S603,S607

    port = None
    for i in result.stdout.splitlines():
        if "prometheus" in i:
            match = re.search(r"--web\.listen-address=:(\d+)", i)
            if match:
                port = match.group(1)
                break
    return port or 9090  # Default port


def run_prometheus(prometheus_dir: str, prometheus_web_port: int) -> None:
    """Run the prometheus server."""
    # Write the prometheus.yml file to the NEMO_CURATOR_METRICS_PATH directory
    prometheus_config_path = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, "prometheus.yml")
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
    """Download the grafana tarball and extract it to the default nemo curator metrics path."""
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
            tar_safe_extract(tar, metrics_dir)

    return grafana_extract_dir


def launch_grafana(grafana_dir: str, grafana_ini_path: str) -> None:
    """Launch the grafana server."""
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
    """Write the grafana configs to the grafana directory."""
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
    xenna_dashboard_src = os.path.join(os.path.dirname(__file__), "xenna_grafana_dashboard.json")
    xenna_dashboard_src = os.path.abspath(xenna_dashboard_src)
    xenna_dashboard_dst = os.path.join(dashboards_path, "xenna_grafana_dashboard.json")
    if os.path.isfile(xenna_dashboard_src) and not os.path.isfile(xenna_dashboard_dst):
        shutil.copy(xenna_dashboard_src, xenna_dashboard_dst)
    return grafana_ini_path


def add_ray_prometheus_metrics_service_discovery(ray_temp_dir: str) -> None:
    """Add the ray prometheus metrics service discovery to the prometheus config."""
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
