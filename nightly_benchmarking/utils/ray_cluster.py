import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import ray  # noqa: F401
from loguru import logger

from nemo_curator.core.client import RayClient


def start_ray_head(
    num_cpus: int,
    num_gpus: int,
    include_dashboard: bool = False,
    enable_object_spilling: bool = False,
    ray_log_file: str | None = None,
) -> tuple[RayClient, str, str]:
    # Create a short temp dir to avoid Unix socket path length limits
    short_temp_dir = f"/tmp/ray_{uuid.uuid4().hex[:8]}"  # noqa: S108
    os.makedirs(short_temp_dir, exist_ok=True)

    # Check environment variables that might interfere
    ray_address_env = os.environ.get("RAY_ADDRESS")
    if ray_address_env:
        logger.warning(f"RAY_ADDRESS already set in environment: {ray_address_env}")
    client = RayClient(
        ray_temp_dir=short_temp_dir,
        include_dashboard=include_dashboard,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        enable_object_spilling=enable_object_spilling,
        ray_dashboard_host="0.0.0.0",  # noqa: S104
    )
    # Redirect Ray startup output to log file if provided, otherwise suppress it
    import sys

    if ray_log_file:
        with open(ray_log_file, "w") as f:
            # Save original stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                # Redirect to log file
                sys.stdout = f
                sys.stderr = f
                client.start()
            finally:
                # Restore original stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        # Suppress Ray startup output by redirecting to devnull
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                client.start()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    time.sleep(2)
    if client.ray_process is None:
        msg = f"RayClient failed to start (ray_process is None). Short temp dir: {short_temp_dir}"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.debug(f"RayClient started successfully: pid={client.ray_process.pid}, port={client.ray_port}")
    return client, short_temp_dir


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


def _copy_ray_debug_artifacts(short_temp_dir: str, ray_destination_dir: str) -> None:
    """Copy Ray debugging artifacts to the specified ray destination directory."""
    temp_path = Path(short_temp_dir)
    if not temp_path.exists():
        return

    # Use the provided ray destination directory directly
    ray_debug_dir = Path(ray_destination_dir)
    ray_debug_dir.mkdir(parents=True, exist_ok=True)

    # Copy log files from Ray temp dir
    logs_src = temp_path / "logs"
    if logs_src.exists():
        logs_dst = ray_debug_dir / "logs"
        shutil.copytree(logs_src, logs_dst, dirs_exist_ok=True, ignore_errors=True)

    # Copy session info but skip sockets directory
    session_src = temp_path / "session_latest"
    if session_src.exists():
        session_dst = ray_debug_dir / "session_latest"
        _copy_session_contents(session_src, session_dst)


def stop_ray_head(client: RayClient, ray_temp_dir: str, ray_destination_dir: str) -> None:
    """Stop Ray head node and clean up artifacts."""
    # Stop the Ray client
    _stop_ray_client(client)

    # Copy debugging artifacts and clean up temp directory
    try:
        _copy_ray_debug_artifacts(ray_temp_dir, ray_destination_dir)
        shutil.rmtree(ray_temp_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to copy/remove Ray temp dir")
