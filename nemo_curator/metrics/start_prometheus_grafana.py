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

import argparse
import os
import sys
import time

from loguru import logger

from nemo_curator.core.utils import get_free_port
from nemo_curator.metrics.constants import (
    DEFAULT_GRAFANA_WEB_PORT,
    DEFAULT_NEMO_CURATOR_METRICS_PATH,
    DEFAULT_PROMETHEUS_WEB_PORT,
)
from nemo_curator.metrics.utils import (
    download_and_extract_prometheus,
    download_grafana,
    is_grafana_running,
    is_prometheus_running,
    launch_grafana,
    run_prometheus,
    write_grafana_configs,
)

# ----------------------------
# Helpers
# ----------------------------


def start_prometheus_grafana(
    prometheus_web_port: int = DEFAULT_PROMETHEUS_WEB_PORT,
    grafana_web_port: int = DEFAULT_GRAFANA_WEB_PORT,
    metrics_dir: str | None = None,
) -> None:
    # Resolve metrics_dir to the default if not provided
    if metrics_dir is None:
        metrics_dir = DEFAULT_NEMO_CURATOR_METRICS_PATH

    # Check if the prometheus or grafana is running. If yes we assume that they were setup using this script and skip the setup.
    if is_prometheus_running(metrics_dir) or is_grafana_running(metrics_dir):
        logger.info("Prometheus or Grafana is already running. Skipping the setup.")
        return

    # Touch our directory for nemo curator in case it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)

    prometheus_web_port = get_free_port(prometheus_web_port)
    grafana_web_port = get_free_port(grafana_web_port)

    prometheus_dir = download_and_extract_prometheus(metrics_dir)

    # Run prometheus
    try:
        run_prometheus(prometheus_dir, prometheus_web_port, metrics_dir)
    except Exception as error:
        error_msg = f"Failed to start Prometheus: {error}"
        logger.error(error_msg)
        raise

    # -----------------------------
    # Grafana setup and launch
    # -----------------------------
    try:
        grafana_dir = download_grafana(metrics_dir)
        grafana_ini_path = write_grafana_configs(grafana_web_port, prometheus_web_port, metrics_dir)
        launch_grafana(grafana_dir, grafana_ini_path, grafana_web_port, metrics_dir)

        # Wait a bit to ensure Grafana starts
        time.sleep(2)
    except Exception as error:
        error_msg = f"Failed to setup or start Grafana: {error}"
        logger.error(error_msg)
        raise

    logger.info("If you are running using Xenna, please remember to export XENNA_RAY_METRICS_PORT=8080")
    logger.info(
        f"You can access the grafana dashboard at http://localhost:{grafana_web_port}, username: admin, password: admin"
    )
    logger.info(f"You can access the prometheus dashboard at http://localhost:{prometheus_web_port}")
    logger.info("Ray dashboards have been auto-generated. You can add more dashboards by adding json files")
    logger.info(f"in {metrics_dir}/grafana/dashboards")
    logger.info(
        f"To stop prometheus and grafana, run: "
        f"kill $(cat {metrics_dir}/prometheus.pid) ; kill $(cat {metrics_dir}/grafana.pid)"
    )
    logger.info("Prometheus stores persistent data inside its data/ directory.")
    return


if __name__ == "__main__":
    # Add an argument parser to get the prometheus and grafana ports
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prometheus_web_port", type=int, default=DEFAULT_PROMETHEUS_WEB_PORT, help="The port to run prometheus on"
    )
    parser.add_argument(
        "--grafana_web_port", type=int, default=DEFAULT_GRAFANA_WEB_PORT, help="The port to run grafana on"
    )
    parser.add_argument(
        "--metrics_dir", type=str, default=None, help="Directory for metrics data (default: per-user temp directory)"
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if not args.yes:
        display_metrics_dir = args.metrics_dir or DEFAULT_NEMO_CURATOR_METRICS_PATH
        print("This will download and start prometheus and grafana on the following ports:")
        print(f"Prometheus: {args.prometheus_web_port}")
        print(f"Grafana: {args.grafana_web_port}")
        print(f"Metrics directory: {display_metrics_dir}")
        print("Are you sure you want to continue? (y/n)")
        if input() != "y":
            print("Exiting...")
            sys.exit(0)

    start_prometheus_grafana(args.prometheus_web_port, args.grafana_web_port, args.metrics_dir)
