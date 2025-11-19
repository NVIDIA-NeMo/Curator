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
import subprocess
import time
import uuid
from pathlib import Path

from loguru import logger

from nemo_curator.core.client import RayClient

ray_client_start_timeout_s = 30
ray_client_start_poll_interval_s = 0.5


def setup_ray_cluster_and_env(
    num_cpus: int,
    num_gpus: int,
    enable_object_spilling: bool,
    ray_log_path: Path,
) -> tuple[RayClient, Path, dict[str, str]]:
    """Setup Ray cluster and environment variables."""
    ray_client, ray_temp_path = start_ray_head(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        enable_object_spilling=enable_object_spilling,
        ray_log_path=ray_log_path,
    )
    verify_ray_responsive(ray_client)

    env = os.environ.copy()
    ray_address = f"localhost:{ray_client.ray_port}"
    env["RAY_ADDRESS"] = ray_address
    # Also set globally so Ray Data executor can find it
    os.environ["RAY_ADDRESS"] = ray_address
    logger.debug(f"Set RAY_ADDRESS={ray_address}")

    return ray_client, ray_temp_path, env


def teardown_ray_cluster_and_env(
    ray_client: RayClient,
    ray_temp_path: Path,
    ray_cluster_path: Path,
) -> None:
    """Teardown Ray cluster and environment variables."""
    if ray_client is not None:
        stop_ray_head(ray_client, ray_temp_path, ray_cluster_path)

        # Clean up RAY_ADDRESS environment variable immediately after stopping cluster
        if "RAY_ADDRESS" in os.environ:
            del os.environ["RAY_ADDRESS"]
            logger.debug("Cleaned up RAY_ADDRESS environment variable")


def start_ray_head(
    num_cpus: int,
    num_gpus: int,
    include_dashboard: bool = True,
    enable_object_spilling: bool = False,
    ray_log_path: Path | None = None,
) -> tuple[RayClient, Path]:
    # Create a short temp dir to avoid Unix socket path length limits
    short_temp_path = Path(f"/tmp/ray_{uuid.uuid4().hex[:8]}")  # noqa: S108
    short_temp_path.mkdir(parents=True, exist_ok=True)

    # Check environment variables that might interfere
    ray_address_env = os.environ.get("RAY_ADDRESS")
    if ray_address_env:
        logger.warning(f"RAY_ADDRESS already set in environment: {ray_address_env}")
    client = RayClient(
        ray_temp_dir=str(short_temp_path),
        include_dashboard=include_dashboard,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        enable_object_spilling=enable_object_spilling,
        ray_dashboard_host="0.0.0.0",  # noqa: S104
    )
    ray_stdouterr_capture_file = str(ray_log_path) if ray_log_path else os.devnull
    client.start(stdouterr_capture_file=ray_stdouterr_capture_file)
    # Wait for Ray client to start, no longer than timeout
    wait_for_ray_client_start(client, ray_client_start_timeout_s, ray_client_start_poll_interval_s)
    logger.debug(f"RayClient started successfully: pid={client.ray_process.pid}, port={client.ray_port}")

    return client, short_temp_path


def wait_for_ray_client_start(client: RayClient, timeout_s: int, poll_interval_s: float) -> None:
    """Wait for Ray client to start, no longer than timeout."""
    elapsed_s = 0
    while client.ray_process is None and elapsed_s < timeout_s:
        if client.ray_process is not None:
            break
        time.sleep(poll_interval_s)
        elapsed_s += poll_interval_s
    if client.ray_process is None:
        msg = f"Ray client failed to start in {timeout_s} seconds"
        raise RuntimeError(msg)


def verify_ray_responsive(client: RayClient, timeout_s: int = 15) -> None:
    env = os.environ.copy()
    env["RAY_ADDRESS"] = f"localhost:{client.ray_port}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            subprocess.run(  # noqa: S603
                ["ray", "status"],  # noqa: S607
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001, PERF203
            time.sleep(1)
        else:
            return

    msg = "Ray cluster did not become responsive in time"
    raise TimeoutError(msg)


def _stop_ray_client(client: RayClient) -> None:
    """Stop the Ray client and clean up environment variables."""
    try:
        client.stop()
        # Clean up RAY_ADDRESS environment variable to prevent interference
        if "RAY_ADDRESS" in os.environ:
            del os.environ["RAY_ADDRESS"]
            logger.debug("Cleaned up RAY_ADDRESS environment variable")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stop Ray client")


def _copy_item_safely(src_path: Path, dst_path: Path) -> None:
    """Copy a single file or directory, logging warnings on failure."""
    try:
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
    except Exception as e:  # noqa: BLE001
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


def stop_ray_head(client: RayClient, ray_temp_path: Path, ray_destination_path: Path) -> None:
    """Stop Ray head node and clean up artifacts."""
    # Stop the Ray client
    _stop_ray_client(client)

    # Copy debugging artifacts and clean up temp directory
    try:
        _copy_ray_debug_artifacts(ray_temp_path, ray_destination_path)
        shutil.rmtree(ray_temp_path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to copy/remove Ray temp dir")
