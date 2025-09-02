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

"""Unified test configuration for Ray Curator tests.

This module provides smart Ray cluster setup that automatically configures
GPU resources based on the test session's requirements.
"""

import os
import socket
import subprocess
from typing import Any

import pytest
import ray
from loguru import logger


def find_free_port() -> int:
    """Find an available port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def gpu_available() -> bool:
    """Check if GPU is available on the system using multiple detection methods."""
    # Method 1: Try pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count > 0:
            logger.info(f"Detected {gpu_count} GPU(s) via pynvml")
            return True
    except Exception:  # noqa: BLE001,S110
        pass

    # Method 2: Try nvidia-smi with short timeout
    try:
        result = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            gpu_count = int(result.stdout.strip())
            logger.info(f"Detected {gpu_count} GPU(s) via nvidia-smi")
            return gpu_count > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    logger.warning("No GPU detected")
    return False


def session_needs_gpu(config: pytest.Config, collected_items: list[pytest.Item]) -> bool:
    """Determine if the current test session needs GPU resources.

    This checks:
    1. If GPU tests are explicitly being run (via -m gpu)
    2. If we're in a GPU test environment (CUDA_VISIBLE_DEVICES set)
    3. If any collected test has the gpu marker
    """
    # Check if running with -m gpu marker
    gpu_marker = config.getoption("-m", default="")
    if gpu_marker:
        if "not gpu" in gpu_marker:
            logger.info("'not gpu' marker detected, disabling GPU cluster")
            return False
        elif "gpu" in gpu_marker:
            logger.info("GPU marker detected in test selection, enabling GPU cluster")
            return True

    # Check if any collected test has gpu marker
    return any(item.get_closest_marker("gpu") for item in collected_items)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Hook to store collected items in config for later use."""
    config._collected_items = items  # Store in config instead of global


def _build_ray_command(temp_dir: str, num_cpus: int, num_gpus: int, object_store_memory: int) -> tuple[list[str], int]:
    """Build the Ray start command with the given configuration."""
    ray_port = find_free_port()
    dashboard_port = find_free_port()
    ray_client_server_port = find_free_port()

    return [
        "ray",
        "start",
        "--head",
        "--disable-usage-stats",
        "--port",
        str(ray_port),
        "--dashboard-port",
        str(dashboard_port),
        "--ray-client-server-port",
        str(ray_client_server_port),
        "--dashboard-host",
        "0.0.0.0",  # noqa: S104
        "--temp-dir",
        str(temp_dir),
        "--num-cpus",
        str(num_cpus),
        "--num-gpus",
        str(num_gpus),
        "--object-store-memory",
        str(object_store_memory),
        "--block",
    ], ray_port


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster(tmp_path_factory: pytest.TempPathFactory, pytestconfig: pytest.Config) -> str:
    """Set up a shared Ray cluster with dynamic GPU configuration.

    This fixture automatically determines whether GPU resources are needed
    based on the test session and configures Ray accordingly.
    """
    # If RAY_ADDRESS is already set (e.g., in CI), we unset it and still start a new Ray cluster
    if "RAY_ADDRESS" in os.environ:
        del os.environ["RAY_ADDRESS"]

    # Get collected items from config (set by pytest_collection_modifyitems)
    collected_items = getattr(pytestconfig, "_collected_items", [])

    # Determine if we need GPU resources
    needs_gpu = session_needs_gpu(pytestconfig, collected_items)
    gpu_available_on_system = gpu_available()

    # Configure GPU resources with strict checking
    if needs_gpu and not gpu_available_on_system:
        error_msg = (
            "GPU tests detected but no GPU available on system. "
            "Either install GPU drivers/hardware or run CPU-only tests with '-m \"not gpu\"'"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Set up Ray configuration values
    num_cpus = 11
    num_gpus = 2 if needs_gpu else 0
    object_store_memory = 2 * (1024**3)  # 2 GB

    logger.info(f"Configuring Ray cluster with {'GPU' if needs_gpu else 'CPU-only'} support")

    # Create a temporary directory for Ray to avoid conflicts with other instances
    temp_dir = tmp_path_factory.mktemp("ray")

    # Build and execute Ray command
    cmd_to_run, ray_port = _build_ray_command(str(temp_dir), num_cpus, num_gpus, object_store_memory)

    logger.info(f"Starting Ray cluster with {num_gpus} GPUs")
    logger.info(f"Running Ray command: {' '.join(cmd_to_run)}")

    # Use explicit path to ray command for security
    ray_process = subprocess.Popen(cmd_to_run, shell=False)  # noqa: S603
    logger.info(f"Started Ray process: {ray_process.pid}")

    ray_address = f"localhost:{ray_port}"
    os.environ["RAY_ADDRESS"] = ray_address
    logger.info(f"Set RAY_ADDRESS for tests to: {ray_address}")

    try:
        yield ray_address
    finally:
        # Ensure cleanup happens even if tests fail
        logger.info("Shutting down Ray cluster")
        ray_process.kill()
        ray_process.wait()  # Wait for process to actually terminate


@pytest.fixture
def shared_ray_client(shared_ray_cluster: str) -> None:
    """Initialize Ray client for tests that need Ray API access."""
    ray.init(
        address=shared_ray_cluster,
        ignore_reinit_error=True,
        log_to_driver=True,
        local_mode=False,
    )

    try:
        yield
    finally:
        logger.info("Shutting down Ray client")
        ray.shutdown()


@pytest.fixture
def ray_gpu_resources() -> dict[str, Any]:
    """Provide information about available GPU resources in the Ray cluster."""
    try:
        resources = ray.available_resources()
        return {
            "gpu_count": resources.get("GPU", 0),
            "has_gpu": resources.get("GPU", 0) > 0,
            "total_resources": resources,
        }
    except (RuntimeError, ConnectionError) as e:
        logger.warning(f"Could not get Ray resources: {e}")
        return {"gpu_count": 0, "has_gpu": False, "total_resources": {}}


@pytest.fixture
def ray_client_with_id_generator(shared_ray_client: None) -> None:  # noqa: ARG001
    """Create and manage ID generator actor for each test."""
    from nemo_curator.stages.deduplication.id_generator import (
        create_id_generator_actor,
        kill_id_generator_actor,
    )

    # Create the ID generator actor
    create_id_generator_actor()

    try:
        yield
    finally:
        # Cleanup after test completes
        kill_id_generator_actor()
