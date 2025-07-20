import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

import requests
import yaml
from loguru import logger
from ray.dashboard.modules.metrics.install_and_start_prometheus import download_prometheus

from ray_curator.core.constants import (
    DEFAULT_EXECUTOR,
    DEFAULT_GRAFANA_WEB_PORT,
    DEFAULT_NEMO_CURATOR_METRICS_PATH,
    DEFAULT_PROMETHEUS_WEB_PORT,
    DEFAULT_RAY_DASHBOARD_HOST,
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_METRICS_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_RAY_TEMP_DIR,
    GRAPHANA_DASHBOARD_YAML_TEMPLATE,
    GRAPHANA_DATASOURCE_YAML_TEMPLATE,
    GRAPHANA_INI_TEMPLATE,
    PROMETHEUS_YAML_TEMPLATE,
)
from ray_curator.core.utils import (
    get_next_free_port,
    get_prometheus_port,
    init_or_connect_to_cluster,
    is_grafana_running,
    is_prometheus_running,
)

# ----------------------------
# Helpers
# ----------------------------


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
            raise Exception(msg)  # noqa: TRY002

    tar.extractall(path=path)  # noqa: S202


def get_ray_client(  # noqa: C901, PLR0912, PLR0913, PLR0915
    executor: str = DEFAULT_EXECUTOR,
    ray_port: int = DEFAULT_RAY_PORT,
    ray_dashboard_port: int = DEFAULT_RAY_DASHBOARD_PORT,
    ray_temp_dir: str = DEFAULT_RAY_TEMP_DIR,
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
    """
    # This function returns a dictionary with the following keys:
    return_dict = {
        "prometheus_web_port": None,
        "grafana_web_port": None,
        "ray_port": None,
        "ray_dashboard_port": None,
        "ray_metrics_port": None,
    }
    # Check if the prometheus or grafana is running. If yes we assume that they were setup using this script and skip the setup.
    if is_prometheus_running() or is_grafana_running():
        logger.info("Prometheus or Grafana is already running. Skipping the setup.")
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
    else:
        # Setup prometheus and grafana.
        # Get port numbers for prometheus and grafana.

        prometheus_web_port = get_next_free_port(prometheus_web_port)
        grafana_web_port = get_next_free_port(grafana_web_port)

        return_dict["prometheus_web_port"] = prometheus_web_port
        return_dict["grafana_web_port"] = grafana_web_port

        downloaded, file_name = download_prometheus()
        if not downloaded:
            logger.error("Failed to download Prometheus.")
            sys.exit(1)
        # Get absolute path to the file and move it to NEMO_CURATOR_METRICS_PATH, filename is a tar.gz file
        file_path = os.path.abspath(file_name)
        os.makedirs(DEFAULT_NEMO_CURATOR_METRICS_PATH, exist_ok=True)
        # Move the file to NEMO_CURATOR_METRICS_PATH
        shutil.move(file_path, os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, file_name))
        # Extract the tar.gz file in the NEMO_CURATOR_METRICS_PATH

        file_path = os.path.join(DEFAULT_NEMO_CURATOR_METRICS_PATH, file_name)
        with tarfile.open(file_path) as tar:
            _safe_extract(tar, DEFAULT_NEMO_CURATOR_METRICS_PATH)

        prometheus_dir = file_path.rstrip(".tar.gz")  # noqa: B005

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
        except Exception as error:  # noqa: BLE001
            logger.error(f"Failed to start Prometheus: {error}")

        # -----------------------------
        # Grafana setup and launch
        # -----------------------------
        try:
            # Determine download URL based on architecture
            arch = platform.machine()
            if arch not in ("x86_64", "amd64"):
                logger.warning(
                    "Automatic Grafana installation is only tested on x86_64/amd64 architectures. "
                    "Please install Grafana manually if the following steps fail."
                )

            grafana_version = "12.0.2"
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

            # -------------------
            # Provisioning setup
            # -------------------
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
                f.write(
                    GRAPHANA_INI_TEMPLATE.format(
                        provisioning_path=provisioning_path, grafana_web_port=grafana_web_port
                    )
                )

            # Write provisioning dashboard yaml
            dashboards_yaml_path = os.path.join(dashboards_prov_path, "default.yml")
            with open(dashboards_yaml_path, "w") as f:
                f.write(GRAPHANA_DASHBOARD_YAML_TEMPLATE.format(dashboards_path=dashboards_path))

            # Write datasource yaml (points to Prometheus instance we just launched)
            datasources_yaml_path = os.path.join(datasources_path, "default.yml")
            prometheus_url = f"http://localhost:{prometheus_web_port}"
            with open(datasources_yaml_path, "w") as f:
                f.write(GRAPHANA_DATASOURCE_YAML_TEMPLATE.format(prometheus_url=prometheus_url))

            # Copy Xenna dashboard json if not already present
            xenna_dashboard_src = os.path.join(
                os.path.dirname(__file__), "..", "..", "scripts", "xenna_graphana_dashboard.json"
            )
            xenna_dashboard_src = os.path.abspath(xenna_dashboard_src)
            xenna_dashboard_dst = os.path.join(dashboards_path, "xenna_graphana_dashboard.json")
            if os.path.isfile(xenna_dashboard_src) and not os.path.isfile(xenna_dashboard_dst):
                shutil.copy(xenna_dashboard_src, xenna_dashboard_dst)

            # -------------------
            # Launch Grafana
            # -------------------
            grafana_cmd = [
                os.path.join(grafana_extract_dir, "bin", "grafana-server"),
                "--config",
                grafana_ini_path,
                f"--homepath={grafana_extract_dir}",
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

            # Wait a bit to ensure Grafana starts
            time.sleep(2)
        except Exception as error:  # noqa: BLE001
            logger.error(f"Failed to setup or start Grafana: {error}")

        # Depending upon the executor, we need to start the ray dashboard.

    if num_gpus is not None:
        if executor != "Xenna":
            msg = "num_gpus is only supported for Xenna. For other executors, use CUDA_VISIBLE_DEVICES"
            raise ValueError(
                msg,
            )
        logger.info(
            "Num GPUs: %s. Xenna will use the first %s GPUs. Currently we do not support custom GPU numbers.",
            num_gpus,
            num_gpus,
        )

    if os.environ.get("RAY_ADDRESS"):
        logger.info("Ray is already running. Skipping the setup.")
    else:
        ray_dashboard_port = get_next_free_port(ray_dashboard_port)
        ray_metrics_port = get_next_free_port(ray_metrics_port)
        ray_port = get_next_free_port(ray_port)

        init_or_connect_to_cluster(
            executor,
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
