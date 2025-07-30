"""Shared test configuration for Ray Curator tests.

This module provides shared Ray cluster setup and teardown for all test modules.
Using a single Ray instance across tests improves performance while maintaining
proper isolation through Ray's actor/task lifecycle management.
"""

import os
import subprocess

import pytest
import ray
from loguru import logger


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster(tmp_path_factory: pytest.TempPathFactory):
    """Set up a shared Ray cluster for all tests in the session.

    This fixture automatically sets up Ray at the beginning of the test session
    and tears it down at the end. It configures Ray with fixed resources for
    consistent testing behavior.
    """

    ONE_GB = 1024**3  # noqa: N806

    # This ensures we are not reusing an existing cluster but starting a new one
    if "RAY_ADDRESS" in os.environ:
        del os.environ["RAY_ADDRESS"]

    # Create a temporary directory for Ray to avoid conflicts with other instances
    temp_dir = tmp_path_factory.mktemp("ray")

    # TODO: Create a cluster with 11 cpus instead of one head node with 11 cpus
    # If we have 6-7 stages all needing 1 cpu, then we atleast need 10 cpus for Xenna / Ray Data to work
    # The 11th CPU is for StageCallCounter

    ray_port = find_free_port()
    dashboard_port = find_free_port()
    ray_client_server_port = find_free_port()

    # TODO: See if we can use get_client in the future
    cmd_to_run = [
        "ray",
        "start",
        "--head",
        "--port",
        str(ray_port),
        "--dashboard-port",
        str(dashboard_port),
        "--ray-client-server-port",
        str(ray_client_server_port),
        "--temp-dir",
        str(temp_dir),
        "--num-cpus",
        "11",
        "--num-gpus",
        "0",
        "--object-store-memory",
        str(2 * ONE_GB),
        "--block",
    ]

    for k, v in os.environ.items():
        if k.startswith("RAY_"):
            logger.info(f"{k}: {v}")

    # Start Ray cluster without --block so it doesn't hang
    logger.info(f"Running Ray command: {' '.join(cmd_to_run)}")
    ray_process = subprocess.Popen(  # noqa: S603
        cmd_to_run,
    )
    logger.info(f"Ran Ray process: {ray_process.pid}")
    ray_address = f"localhost:{ray_port}"

    # Set RAY_ADDRESS so Xenna will connect to our cluster
    os.environ["RAY_ADDRESS"] = ray_address
    logger.info(f"Set RAY_ADDRESS for tests to: {ray_address}")

    yield ray_address

    # Shutdown Ray after all tests complete
    logger.info("Shutting down Ray cluster")
    ray_process.kill()


@pytest.fixture
def shared_ray_client(shared_ray_cluster: str) -> None:
    """Initialize Ray client for tests that need Ray API access.

    This fixture should be used by tests that need to call Ray functions
    like ray.nodes(), ray.available_resources(), etc. Tests that don't
    need direct Ray API access (like integration tests) should not use
    this fixture.

    Args:
        shared_ray_cluster: The Ray cluster address from shared_ray_cluster fixture
    """
    ray.init(
        address=shared_ray_cluster,
        ignore_reinit_error=True,
        log_to_driver=True,
        local_mode=False,  # Use cluster mode for better testing of distributed features
    )

    yield

    # Shutdown Ray client after test completes
    logger.info("Shutting down Ray client")
    ray.shutdown()
