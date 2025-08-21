import os
import tempfile
import time

import pytest

from ray_curator.core.client import RayClient

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


# This test should be allowed to fail since it's not deterministic.
@pytest.mark.xfail(strict=False, reason="Non-deterministic test due to Ray startup timing")
def test_get_ray_client_single_start():
    with tempfile.TemporaryDirectory(prefix="ray_test_single_") as ray_tmp:
        # Clear the environment variable RAY_ADDRESS
        os.environ.pop("RAY_ADDRESS", None)
        client = RayClient(ray_temp_dir=ray_tmp)
        client.start()
        time.sleep(10)  # Wait for ray to start.
        with open(os.path.join(ray_tmp, "ray_current_cluster")) as f:
            assert f.read() == f"127.0.0.1:{client.ray_port}"
        client.stop()


@pytest.mark.xfail(strict=False, reason="Non-deterministic test due to Ray startup timing")
def test_get_ray_client_multiple_start():
    with (
        tempfile.TemporaryDirectory(prefix="ray_test_first_") as ray_tmp1,
        tempfile.TemporaryDirectory(prefix="ray_test_second_") as ray_tmp2,
    ):
        # Clear the environment variable RAY_ADDRESS
        os.environ.pop("RAY_ADDRESS", None)
        client1 = RayClient(ray_temp_dir=ray_tmp1)
        client1.start()
        time.sleep(10)  # Wait for ray to start.
        with open(os.path.join(ray_tmp1, "ray_current_cluster")) as f:
            assert f.read() == f"127.0.0.1:{client1.ray_port}"
        # Clear the environment variable RAY_ADDRESS
        os.environ.pop("RAY_ADDRESS", None)
        client2 = RayClient(ray_temp_dir=ray_tmp2)
        client2.start()
        time.sleep(10)  # Wait for ray to start.
        with open(os.path.join(ray_tmp2, "ray_current_cluster")) as f:
            assert f.read() == f"127.0.0.1:{client2.ray_port}"
        client1.stop()
        client2.stop()
