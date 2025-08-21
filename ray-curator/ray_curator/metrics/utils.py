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

"""Utilities for Prometheus and Grafana monitoring services."""

import os
import re
import subprocess

import psutil
import requests
import yaml

from ray_curator.metrics.constants import DEFAULT_NEMO_CURATOR_METRICS_PATH


def is_prometheus_running() -> bool:
    """Check if Prometheus is currently running."""
    return any(proc.info["name"].lower() == "prometheus" for proc in psutil.process_iter(["name"]))


def is_grafana_running() -> bool:
    """Check if Grafana is currently running."""
    return any(proc.info["name"].lower() == "grafana" for proc in psutil.process_iter(["name"]))


def get_prometheus_port() -> int:
    """Get the port number that Prometheus is running on."""
    result = subprocess.run(["ps", "-ef", "|", "grep", "prometheus"], check=False, capture_output=True, text=True)  # noqa: S607, S603

    port = None
    for i in result.stdout.splitlines():
        if "prometheus" in i:
            match = re.search(r"--web\.listen-address=:(\d+)", i)
            if match:
                port = match.group(1)
                break
    return port or 9090  # Default port


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
