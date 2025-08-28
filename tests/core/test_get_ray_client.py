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
import tempfile
import time

import pytest

from nemo_curator.core.client import RayClient

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


# This test should be allowed to fail since it's not deterministic.
@pytest.mark.xfail(strict=False, reason="Non-deterministic test due to Ray startup timing")
def test_get_ray_client_single_start():
    initial_address = os.environ.pop("RAY_ADDRESS", None)
    client = None
    try:
        with tempfile.TemporaryDirectory(prefix="ray_test_single_") as ray_tmp:
            client = RayClient(ray_temp_dir=ray_tmp)
            client.start()
            time.sleep(10)  # Wait for ray to start.

            with open(os.path.join(ray_tmp, "ray_current_cluster")) as f:
                content = f.read()
                assert content.split(":")[1].strip() == str(client.ray_port)
    finally:
        if client:
            client.stop()
        if initial_address:
            os.environ["RAY_ADDRESS"] = initial_address
        else:
            os.environ.pop("RAY_ADDRESS", None)


@pytest.mark.xfail(strict=False, reason="Non-deterministic test due to Ray startup timing")
def test_get_ray_client_multiple_start():
    initial_address = os.environ.pop("RAY_ADDRESS", None)
    client1 = None
    client2 = None
    try:
        with (
            tempfile.TemporaryDirectory(prefix="ray_test_first_") as ray_tmp1,
            tempfile.TemporaryDirectory(prefix="ray_test_second_") as ray_tmp2,
        ):
            client1 = RayClient(ray_temp_dir=ray_tmp1)
            client1.start()
            time.sleep(10)  # Wait for ray to start.
            with open(os.path.join(ray_tmp1, "ray_current_cluster")) as f:
                assert f.read().split(":")[1].strip() == str(client1.ray_port)
            # Clear the environment variable RAY_ADDRESS
            os.environ.pop("RAY_ADDRESS", None)
            client2 = RayClient(ray_temp_dir=ray_tmp2)
            client2.start()
            time.sleep(10)  # Wait for ray to start.
            with open(os.path.join(ray_tmp2, "ray_current_cluster")) as f:
                assert f.read().split(":")[1].strip() == str(client2.ray_port)
    finally:
        if client1:
            client1.stop()
        if client2:
            client2.stop()
        if initial_address:
            os.environ["RAY_ADDRESS"] = initial_address
        else:
            os.environ.pop("RAY_ADDRESS", None)
