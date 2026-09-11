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

# ruff: noqa: ARG001

import glob
import os
import tempfile
import time

import pytest

from nemo_curator.core.client import RayClient


def _assert_ray_cluster_started(client: RayClient, timeout: int = 30) -> None:
    fn = os.path.join(client.ray_temp_dir, "ray_current_cluster")
    t_start = time.perf_counter()
    while True:
        if os.path.exists(fn):
            # Cluster is up and running
            break
        elif time.perf_counter() - t_start > timeout:
            msg = f"Ray cluster didn't start after {timeout} seconds"
            raise AssertionError(msg)
        else:
            time.sleep(1)

    with open(fn) as f:
        content = f.read()
        assert content.split(":")[1].strip() == str(client.ray_port)


def _assert_ray_stdouterr_output(stdouterr_capture_file: str) -> None:
    """Assert that the expected output is in capture file."""
    # stdout/stderr output may not always appear immediately, hence the loop.
    timeout = 30
    elapsed = 0
    while elapsed < timeout:
        if os.path.exists(stdouterr_capture_file):
            with open(stdouterr_capture_file) as f:
                if "Ray runtime started." in f.read():
                    break
        time.sleep(1)
        elapsed += 1
    if elapsed >= timeout:
        msg = f"Expected output not found in {stdouterr_capture_file} after {timeout} seconds"
        raise AssertionError(msg)


@pytest.fixture(scope="module")
def clean_env():
    initial_address = os.environ.pop("RAY_ADDRESS", None)
    yield
    if initial_address:
        os.environ["RAY_ADDRESS"] = initial_address
    else:
        os.environ.pop("RAY_ADDRESS", None)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_get_ray_client_single_start(clean_env: pytest.fixture):
    client = None
    try:
        with tempfile.TemporaryDirectory(prefix="ray_test_single_") as ray_tmp:
            client = RayClient(ray_temp_dir=ray_tmp)
            client.start()
            _assert_ray_cluster_started(client)

    finally:
        if client:
            client.stop()


def test_get_ray_client_multiple_start(clean_env: pytest.fixture):
    client1 = None
    client2 = None
    try:
        with (
            tempfile.TemporaryDirectory(prefix="ray_test_first_") as ray_tmp1,
            tempfile.TemporaryDirectory(prefix="ray_test_second_") as ray_tmp2,
        ):
            client1 = RayClient(ray_temp_dir=ray_tmp1)
            client1.start()
            _assert_ray_cluster_started(client1)
            # Clear the environment variable RAY_ADDRESS
            os.environ.pop("RAY_ADDRESS", None)
            client2 = RayClient(ray_temp_dir=ray_tmp2)
            client2.start()
            _assert_ray_cluster_started(client2)
    finally:
        if client1:
            client1.stop()
        if client2:
            client2.stop()


def test_ray_client_context_manager(clean_env: pytest.fixture):
    with tempfile.TemporaryDirectory(prefix="ray_test_ctx_manager_") as ray_tmp:
        with RayClient(ray_temp_dir=ray_tmp) as client:
            _assert_ray_cluster_started(client)

        assert client.ray_process is None


def test_get_ray_client_single_start_with_stdouterr_capture(clean_env: pytest.fixture):
    client = None
    try:
        with tempfile.TemporaryDirectory(prefix="ray_test_single_") as ray_tmp:
            # Check that stdout/stderr is captured after calling start()
            stdouterr_capture_file = os.path.join(ray_tmp, "ray_stdouterr.log")
            client = RayClient(ray_temp_dir=ray_tmp, ray_stdouterr_capture_file=stdouterr_capture_file)
            client.start()
            _assert_ray_cluster_started(client)
            _assert_ray_stdouterr_output(stdouterr_capture_file)
            client.stop()
            os.environ.pop("RAY_ADDRESS", None)

            # Check that an error is raised if the capture file already exists
            with pytest.raises(FileExistsError):
                RayClient(ray_temp_dir=ray_tmp, ray_stdouterr_capture_file=stdouterr_capture_file)

        with tempfile.TemporaryDirectory(prefix="ray_test_single_") as ray_tmp:
            # Check that stdout/stderr is captured if the client is used with a context manager
            stdouterr_capture_file = os.path.join(ray_tmp, "ray_stdouterr.log")
            with RayClient(ray_temp_dir=ray_tmp, ray_stdouterr_capture_file=stdouterr_capture_file) as client:
                _assert_ray_cluster_started(client)
                _assert_ray_stdouterr_output(stdouterr_capture_file)

    finally:
        if client:
            client.stop()


def test_ray_client_stop_removes_only_its_own_session_dir(clean_env: pytest.fixture):
    kept = None
    client = None
    try:
        with tempfile.TemporaryDirectory(prefix="ray_test_session_") as ray_tmp:

            def sessions() -> list[str]:
                return sorted(glob.glob(os.path.join(ray_tmp, "session_2*")))

            # By default the session dir is kept, as before.
            kept = RayClient(ray_temp_dir=ray_tmp)
            kept.start()
            _assert_ray_cluster_started(kept)
            kept.stop()
            older = sessions()
            assert len(older) == 1

            client = RayClient(ray_temp_dir=ray_tmp, cleanup_ray_session_dir=True)
            latest = os.path.join(ray_tmp, "session_latest")
            # Stand-in for the session dir of another cluster sharing the temp dir.
            concurrent = os.path.join(ray_tmp, "session_2099-01-01_00-00-00_000000_1")
            for _ in range(2):  # the same client is started and stopped twice
                client.start()
                _assert_ray_cluster_started(client)
                own = glob.glob(os.path.join(ray_tmp, f"session_*_{client.ray_process.pid}"))
                assert len(own) == 1
                assert os.path.realpath(latest) == os.path.realpath(own[0])
                os.makedirs(concurrent, exist_ok=True)
                client.stop()
                client.stop()  # a second stop() is a no-op
                assert sessions() == sorted([*older, concurrent])
                assert not os.path.lexists(latest)
            time.sleep(15)  # no leftover Ray process recreates the dir
            assert sessions() == sorted([*older, concurrent])
    finally:
        for c in (kept, client):
            if c:
                c.stop()


def test_ray_client_stop_keeps_sessions_of_external_cluster(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch
):
    with tempfile.TemporaryDirectory(prefix="ray_test_external_") as ray_tmp:
        external = os.path.join(ray_tmp, "session_x_1")
        os.makedirs(external)
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6379")
        client = RayClient(ray_temp_dir=ray_tmp, cleanup_ray_session_dir=True)
        client.start()
        assert client.ray_process is None
        client.stop()
        assert os.path.isdir(external)
