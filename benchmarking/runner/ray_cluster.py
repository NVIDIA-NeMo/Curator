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
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import ray
from loguru import logger
from runner.utils import run_shm_size_check

from nemo_curator.core.client import RayClient
from nemo_curator.core.utils import check_ray_responsive

ray_client_start_timeout_s = 30
ray_client_start_poll_interval_s = 0.5


def setup_ray_cluster_and_env(  # noqa: PLR0913
    num_cpus: int,
    num_gpus: int,
    enable_object_spilling: bool,
    ray_log_path: Path,
    object_store_size: int | None = None,
    include_dashboard: bool = True,
) -> tuple[RayClient, Path]:
    """Setup a Ray cluster and set the RAY_ADDRESS environment variable and return the Ray client and temp dir."""
    # Create a short temp dir to avoid Unix socket path length limits
    short_temp_path = Path(f"/tmp/ray_{uuid.uuid4().hex[:8]}")  # noqa: S108
    short_temp_path.mkdir(parents=True, exist_ok=True)

    # Capture stdout/stderr to a file if provided, otherwise suppress it
    ray_stdouterr_capture_file = str(ray_log_path) if ray_log_path else os.devnull

    # Check environment variables that might interfere
    ray_address_env = os.environ.get("RAY_ADDRESS")
    if ray_address_env:
        logger.warning(f"RAY_ADDRESS already set in environment: {ray_address_env}")

    responsive = False
    retries = 0
    max_retries = 5
    while not responsive and retries < max_retries:
        logger.info(f"Starting Ray cluster (attempt {retries + 1} of {max_retries})...")

        # Capture the ray cluster output for each attempt to start it using a unique file name
        if ray_log_path and retries > 0:
            ray_stdouterr_capture_file = f"{ray_log_path!s}-{retries + 1}"

        # Create and start the Ray client
        client = RayClient(
            ray_temp_dir=str(short_temp_path),
            include_dashboard=include_dashboard,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            enable_object_spilling=enable_object_spilling,
            ray_dashboard_host="0.0.0.0",  # noqa: S104
            ray_stdouterr_capture_file=ray_stdouterr_capture_file,
            object_store_memory=object_store_size,
        )
        client.start()

        _ensure_ray_client_process_started(client, ray_client_start_timeout_s, ray_client_start_poll_interval_s)
        responsive = check_ray_responsive()
        run_shm_size_check(human_readable=True)
        if not responsive:
            logger.info("Ray cluster did not become responsive in time, stopping client and retrying...")
            client.stop()
            retries += 1

    if not responsive:
        msg = f"Failed to start Ray cluster after {max_retries} attempts"
        raise RuntimeError(msg)

    logger.info(f"RayClient started successfully: pid={client.ray_process.pid}, port={client.ray_port}")
    return client, short_temp_path


def teardown_ray_cluster_and_env(
    ray_client: RayClient,
    ray_temp_path: Path,
    ray_cluster_path: Path,
) -> None:
    """Teardown Ray cluster and environment variables."""
    if ray_client is not None:
        # This also removes the RAY_ADDRESS environment variable if the client also started the Ray cluster
        try:
            # Stop the Ray client
            # This also removes the RAY_ADDRESS environment variable if the client also started the Ray cluster
            ray_client.stop()
        except Exception:
            logger.exception("Failed to stop Ray client")
        # Copy debugging artifacts and clean up temp directory
        try:
            _copy_ray_debug_artifacts(ray_temp_path, ray_cluster_path)
            shutil.rmtree(ray_temp_path, ignore_errors=True)
        except Exception:
            logger.exception("Failed to copy/remove Ray temp dir")


def get_ray_cluster_data() -> dict[str, Any]:
    """Get resource data from the Ray cluster."""
    ray.init(ignore_reinit_error=True)
    time.sleep(0.2)  # ray.available_resources() returns might have a lag
    ray_data = ray.cluster_resources()
    ray.shutdown()
    return ray_data


def _ensure_ray_client_process_started(client: RayClient, timeout_s: int, poll_interval_s: float) -> None:
    """Ensure the Ray client process has been started, no longer than timeout."""
    elapsed_s = 0
    while client.ray_process is None and elapsed_s < timeout_s:
        time.sleep(poll_interval_s)
        elapsed_s += poll_interval_s
    if client.ray_process is None:
        msg = f"Ray client process failed to start in {timeout_s} seconds"
        raise RuntimeError(msg)


def _copy_item_safely(src_path: Path, dst_path: Path) -> None:
    """Copy a single file or directory, logging warnings on failure."""
    try:
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
    except Exception as e:
        logger.warning(f"Failed to copy {src_path.name}: {e}")


def _copy_session_contents(session_src: Path, session_dst: Path) -> None:
    """Copy session directory contents, excluding sockets."""
    session_dst.mkdir(parents=True, exist_ok=True)

    for item in session_src.iterdir():
        if item.name == "sockets":  # Skip sockets directory
            logger.debug("Skipping sockets directory")
            continue

        dst_item = session_dst / item.name
        _copy_item_safely(item, dst_item)


def _copy_ray_debug_artifacts(short_temp_path: Path, ray_destination_path: Path) -> None:
    """Copy Ray debugging artifacts to the specified ray destination directory."""

    if not short_temp_path.exists():
        return

    # Use the provided ray destination directory directly
    ray_destination_path.mkdir(parents=True, exist_ok=True)

    # Copy log files from Ray temp dir
    logs_src = short_temp_path / "logs"
    if logs_src.exists():
        logs_dst = ray_destination_path / "logs"
        shutil.copytree(logs_src, logs_dst, dirs_exist_ok=True, ignore_errors=True)

    # Copy session info but skip sockets directory
    session_src = short_temp_path / "session_latest"
    if session_src.exists():
        session_dst = ray_destination_path / "session_latest"
        _copy_session_contents(session_src, session_dst)
