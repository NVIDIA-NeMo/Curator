"""Shared test configuration for Ray Curator tests.

This module provides shared Ray cluster setup and teardown for all test modules.
Using a single Ray instance across tests improves performance while maintaining
proper isolation through Ray's actor/task lifecycle management.
"""

import os
import subprocess
from pathlib import Path

import pytest
import ray
from loguru import logger


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster():
    """Set up a shared Ray cluster for all tests in the session.

    This fixture automatically sets up Ray at the beginning of the test session
    and tears it down at the end. It configures Ray with fixed resources for
    consistent testing behavior.
    """

    ONE_GB = 1024**3  # noqa: N806

    # This ensures we are not reusing an existing cluster but starting a new one
    if "RAY_ADDRESS" in os.environ:
        del os.environ["RAY_ADDRESS"]

    # Get the ray-curator directory to add to the working_dir to enable serialization of test modules
    ray_curator_path = Path(__file__).parent.parent.parent.resolve()

    # TODO: Create a cluster with 11 cpus instead of one head node with 11 cpus
    # If we have 6-7 stages all needing 1 cpu, then we atleast need 10 cpus for Xenna / Ray Data to work
    # The 11th CPU is for StageCallCounter

    ray_port = find_free_port()

    # TODO: See if we can use get_client in the future
    ray_process = subprocess.Popen(  # noqa: S603
        [  # noqa: S607
            "ray",
            "start",
            "--head",
            "--port",
            str(ray_port),
            "--num-cpus",
            "11",
            "--num-gpus",
            "0",
            "--object-store-memory",
            str(2 * ONE_GB),
            "--block",
        ],
        env={**os.environ, "RAY_MAX_LIMIT_FROM_API_SERVER": "40000", "RAY_MAX_LIMIT_FROM_DATA_SOURCE": "40000"},
    )

    ray.init(
        address=f"localhost:{ray_port}",
        ignore_reinit_error=True,
        log_to_driver=True,
        local_mode=False,  # Use cluster mode for better testing of distributed features
        runtime_env={
            "working_dir": str(ray_curator_path),
            "excludes": [
                ".ruff_cache/",
                "__pycache__/",
                "ray_curator/examples/*",
            ],
        },
    )
    # Get the actual Ray address more reliably
    ray_address = f"localhost:{ray_port}"
    # Set RAY_ADDRESS so Xenna will connect to our cluster
    os.environ["RAY_ADDRESS"] = ray_address
    logger.info(f"Set RAY_ADDRESS for tests to: {ray_address}")

    yield ray_address

    # Shutdown Ray after all tests complete
    logger.info("Shutting down Ray cluster")
    ray_process.kill()
