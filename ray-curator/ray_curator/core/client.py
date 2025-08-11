import os
import time

from loguru import logger
from ray.dashboard.modules.metrics.install_and_start_prometheus import download_prometheus

from ray_curator.core.constants import (
    DEFAULT_GRAFANA_WEB_PORT,
    DEFAULT_PROMETHEUS_WEB_PORT,
    DEFAULT_RAY_DASHBOARD_HOST,
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_METRICS_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_RAY_TEMP_DIR,
)
from ray_curator.core.utils import (
    add_ray_prometheus_metrics_service_discovery,
    download_grafana,
    get_next_free_port,
    init_or_connect_to_cluster,
    is_grafana_running,
    is_prometheus_running,
    launch_grafana,
    move_and_extract_prometheus,
    run_prometheus,
    write_grafana_configs,
)

# ----------------------------
# Helpers
# ----------------------------


def get_ray_client(  # noqa: PLR0913
    ray_port: int = DEFAULT_RAY_PORT,
    ray_dashboard_port: int = DEFAULT_RAY_DASHBOARD_PORT,
    ray_temp_dir: str = DEFAULT_RAY_TEMP_DIR,
    include_dashboard: bool = True,
    prometheus_web_port: int = DEFAULT_PROMETHEUS_WEB_PORT,
    grafana_web_port: int = DEFAULT_GRAFANA_WEB_PORT,
    ray_metrics_port: int = DEFAULT_RAY_METRICS_PORT,
    ray_dashboard_host: str = DEFAULT_RAY_DASHBOARD_HOST,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
    enable_object_spilling: bool = False,
) -> dict:
    """
    This function is used to setup the ray cluster and the metrics dashboard.
    It does this by first checking if the prometheus and grafana are running.
    If they are not running, it downloads the prometheus and grafana and starts them.
    If the specified ports are already in use, it will find the next available port and use that.
    It returns a dictionary with the following keys:
    - prometheus_web_port: The port number of the prometheus web UI.
    - grafana_web_port: The port number of the grafana web UI.
    - ray_port: The port number of the ray dashboard.
    - ray_dashboard_port: The port number of the ray dashboard.
    - ray_metrics_port: The port number of the ray metrics.

    Args:
        ray_port: The port number of the ray GCS.
        ray_dashboard_port: The port number of the ray dashboard.
        ray_temp_dir: The temporary directory to use for ray.
        include_dashboard: Whether to include the dashboard. If true, Grafana and Prometheus will be setup and launched.
        prometheus_web_port: The port number of the prometheus web UI.
        grafana_web_port: The port number of the grafana web UI.
        ray_metrics_port: The port number of the ray metrics.
        ray_dashboard_host: The host of the ray dashboard.
        num_gpus: The number of GPUs to use.
        num_cpus: The number of CPUs to use.
        enable_object_spilling: Whether to enable object spilling.
    """
    # This function returns a dictionary with the following keys:
    return_dict = {
        "prometheus_web_port": None,
        "grafana_web_port": None,
        "ray_port": None,
        "ray_dashboard_port": None,
        "ray_metrics_port": None,
    }
    if include_dashboard:
        # Check if the prometheus or grafana is running. If yes we assume that they were setup using this script and skip the setup.
        if is_prometheus_running() or is_grafana_running():
            logger.info("Prometheus or Grafana is already running. Skipping the setup.")

            add_ray_prometheus_metrics_service_discovery(ray_temp_dir)

        else:
            # Setup prometheus and grafana.
            # Get port numbers for prometheus and grafana.

            prometheus_web_port = get_next_free_port(prometheus_web_port)
            grafana_web_port = get_next_free_port(grafana_web_port)

            return_dict["prometheus_web_port"] = prometheus_web_port
            return_dict["grafana_web_port"] = grafana_web_port

            downloaded, file_name = download_prometheus()
            if not downloaded:
                error_msg = "Failed to download Prometheus."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            file_path = move_and_extract_prometheus(file_name)

            prometheus_dir = file_path.rstrip(".tar.gz")  # noqa: B005

            # Run prometheus
            try:
                run_prometheus(ray_temp_dir, prometheus_dir, prometheus_web_port)
            except Exception as error:
                error_msg = f"Failed to start Prometheus: {error}"
                logger.error(error_msg)
                raise

            # -----------------------------
            # Grafana setup and launch
            # -----------------------------
            try:
                grafana_dir = download_grafana()
                grafana_ini_path = write_grafana_configs(grafana_web_port, prometheus_web_port)
                launch_grafana(grafana_dir, grafana_ini_path)

                # Wait a bit to ensure Grafana starts
                time.sleep(2)
            except Exception as error:
                error_msg = f"Failed to setup or start Grafana: {error}"
                logger.error(error_msg)
                raise

    if os.environ.get("RAY_ADDRESS"):
        logger.info("Ray is already running. Skipping the setup.")
    else:
        ray_dashboard_port = get_next_free_port(ray_dashboard_port)
        ray_metrics_port = get_next_free_port(ray_metrics_port)
        ray_port = get_next_free_port(ray_port)

        init_or_connect_to_cluster(
            ray_port,
            ray_temp_dir,
            ray_dashboard_port,
            ray_metrics_port,
            ray_dashboard_host,
            num_gpus,
            num_cpus,
            enable_object_spilling,
        )

        return_dict["ray_port"] = ray_port
        return_dict["ray_dashboard_port"] = ray_dashboard_port
        return_dict["ray_metrics_port"] = ray_metrics_port

    return return_dict


if __name__ == "__main__":
    get_ray_client()
