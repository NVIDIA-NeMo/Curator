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
import subprocess
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from nemo_curator.core import client as client_module
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
            for _ in range(2):
                client.start()
                _assert_ray_cluster_started(client)
                own = glob.glob(os.path.join(ray_tmp, f"session_*_{client.ray_process.pid}"))
                assert len(own) == 1
                assert os.path.realpath(latest) == os.path.realpath(own[0])
                os.makedirs(concurrent, exist_ok=True)
                client.stop()
                client.stop()
                assert sessions() == sorted([*older, concurrent])
                assert not os.path.lexists(latest)
            deadline = time.monotonic() + 30
            while _live_processes_using(ray_tmp) and time.monotonic() < deadline:
                time.sleep(0.5)
            assert not _live_processes_using(ray_tmp)
            assert sessions() == sorted([*older, concurrent])
    finally:
        for c in (kept, client):
            if c:
                c.stop()


def _live_processes_using(path: str) -> list[int]:
    return [
        p.pid
        for p in psutil.process_iter(["cmdline", "status"])
        if p.info["status"] != psutil.STATUS_ZOMBIE and path in " ".join(p.info["cmdline"] or [])
    ]


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


def _fake_ray_start(monkeypatch: pytest.MonkeyPatch, session_roots: list[Path], on_sigterm: str = ":") -> None:
    """Stand in for `ray start`: a process that makes session_*_<its pid> under each root, runs `on_sigterm` on SIGTERM."""

    def fake_init_cluster(**kwargs) -> subprocess.Popen:
        roots = " ".join(f"'{r}'" for r in session_roots)
        script = (
            f"trap '{on_sigterm}; exit 0' TERM; for r in {roots}; do mkdir -p \"$r/session_2099-01-01_$$\"; done; "
        )
        script += "while :; do sleep 0.1; done"
        proc = subprocess.Popen(["bash", "-c", script], start_new_session=True)  # noqa: S603, S607
        for root in session_roots:
            while not (root / f"session_2099-01-01_{proc.pid}").is_dir():
                time.sleep(0.05)
        return proc

    monkeypatch.setattr(client_module, "init_cluster", fake_init_cluster)
    monkeypatch.setattr(client_module, "check_ray_responsive", lambda: True)


@pytest.mark.parametrize("temp_dir_name", ["ray[1]", "ray*", "ray?"])
def test_ray_client_stop_never_touches_sibling_of_temp_dir_with_glob_characters(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, temp_dir_name: str
):
    ray_tmp, sibling = tmp_path / temp_dir_name, tmp_path / "ray1"
    _fake_ray_start(monkeypatch, [ray_tmp, sibling])
    client = RayClient(ray_temp_dir=str(ray_tmp), include_dashboard=False, cleanup_ray_session_dir=True)
    client.start()
    pid = client.ray_process.pid
    client.stop()
    assert not (ray_tmp / f"session_2099-01-01_{pid}").exists()
    assert (sibling / f"session_2099-01-01_{pid}").is_dir()


def test_ray_client_stop_keeps_session_dir_that_appears_after_stop_begins(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Once the process is reaped its pid can be reused, so a dir showing up after the kill is not provably ours.
    _fake_ray_start(monkeypatch, [tmp_path], on_sigterm=f'mkdir "{tmp_path}/session_late_$$"')
    client = RayClient(ray_temp_dir=str(tmp_path), include_dashboard=False, cleanup_ray_session_dir=True)
    client.start()
    pid = client.ray_process.pid
    client.stop()
    assert not (tmp_path / f"session_2099-01-01_{pid}").exists()
    assert (tmp_path / f"session_late_{pid}").is_dir()


@pytest.mark.parametrize("leftover", ["same_named_session", "dangling_session_latest"])
def test_ray_client_stop_after_chdir_cleans_relative_temp_dir_it_started_in(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, leftover: str
):
    start_dir, other_dir = tmp_path / "start", tmp_path / "other"
    (other_dir / "ray").mkdir(parents=True)
    start_dir.mkdir()
    monkeypatch.chdir(start_dir)
    _fake_ray_start(monkeypatch, [Path("ray")])
    client = RayClient(ray_temp_dir="ray", include_dashboard=False, cleanup_ray_session_dir=True)
    client.start()
    pid = client.ray_process.pid
    # Something under the new cwd that only looks like ours.
    other = other_dir / "ray" / (f"session_2099-01-01_{pid}" if leftover == "same_named_session" else "session_latest")
    if leftover == "same_named_session":
        other.mkdir()
    else:
        other.symlink_to(other_dir / "ray" / "gone")
    monkeypatch.chdir(other_dir)
    client.stop()
    assert os.path.lexists(other)
    assert not (start_dir / "ray" / f"session_2099-01-01_{pid}").exists()


@pytest.mark.parametrize(
    ("caller_reaped", "expect_kept"),
    [
        pytest.param(False, False, id="process_alive_dir_is_ours"),
        pytest.param(True, True, id="caller_reaped_later_pid_match_is_kept"),
    ],
)
def test_ray_client_stop_removes_started_dir_but_not_later_pid_match_even_if_reaped(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caller_reaped: bool, expect_kept: bool
):
    _fake_ray_start(monkeypatch, [tmp_path])
    client = RayClient(ray_temp_dir=str(tmp_path), include_dashboard=False, cleanup_ray_session_dir=True)
    client.start()
    pid = client.ray_process.pid
    if caller_reaped:
        client.ray_process.kill()
        client.ray_process.wait()
    # With the process reaped, this stands in for a cluster that reused the pid after start().
    later = tmp_path / f"session_2099-01-02_{pid}"
    later.mkdir()
    client.stop()
    assert later.is_dir() is expect_kept
    assert not (tmp_path / f"session_2099-01-01_{pid}").exists()


def test_ray_client_restart_after_chdir_resolves_relative_temp_dir_from_new_cwd(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _fake_ray_start(monkeypatch, [Path("ray")])
    started_in: list[str] = []
    fake_init_cluster = client_module.init_cluster

    def recording_init_cluster(**kwargs) -> subprocess.Popen:
        started_in.append(os.path.abspath(kwargs["ray_temp_dir"]))
        return fake_init_cluster(**kwargs)

    monkeypatch.setattr(client_module, "init_cluster", recording_init_cluster)
    client = RayClient(ray_temp_dir="ray", include_dashboard=False, cleanup_ray_session_dir=True)
    for cwd in (first, second):
        monkeypatch.chdir(cwd)
        client.start()
        client.stop()
    assert started_in == [str(first / "ray"), str(second / "ray")]


@pytest.mark.parametrize("chdir_after_start", [False, True])
@pytest.mark.parametrize("leftover", ["same_named_session", "dangling_session_latest"])
def test_ray_client_stop_with_flag_enabled_after_start_never_touches_cwd(
    clean_env: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    leftover: str,
    chdir_after_start: bool,
):
    ray_tmp, start_dir, cwd = tmp_path / "ray", tmp_path / "start", tmp_path / "cwd"
    start_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(start_dir)
    _fake_ray_start(monkeypatch, [ray_tmp])
    client = RayClient(ray_temp_dir=str(ray_tmp), include_dashboard=False)
    client.start()
    pid = client.ray_process.pid
    client.cleanup_ray_session_dir = True  # flipped after start: nothing was recorded for cleanup
    if chdir_after_start:
        monkeypatch.chdir(cwd)
    else:
        cwd = start_dir
    other = cwd / (f"session_2099-01-01_{pid}" if leftover == "same_named_session" else "session_latest")
    if leftover == "same_named_session":
        other.mkdir()
    else:
        other.symlink_to(cwd / "gone")
    client.stop()
    assert os.path.lexists(other)


@pytest.mark.parametrize("leftover", ["same_named_session", "dangling_session_latest"])
def test_ray_client_stop_after_cleanup_free_restart_never_touches_previous_run_dir(
    clean_env: pytest.fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, leftover: str
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _fake_ray_start(monkeypatch, [Path("ray")])
    client = RayClient(ray_temp_dir="ray", include_dashboard=False, cleanup_ray_session_dir=True)
    monkeypatch.chdir(first)
    client.start()
    client.stop()
    # Second run starts with cleanup off, in another cwd; the flag is flipped on before its stop().
    client.cleanup_ray_session_dir = False
    monkeypatch.chdir(second)
    client.start()
    pid = client.ray_process.pid
    client.cleanup_ray_session_dir = True
    old = first / "ray" / (f"session_2099-01-01_{pid}" if leftover == "same_named_session" else "session_latest")
    if leftover == "same_named_session":
        old.mkdir()
    else:
        old.symlink_to(first / "ray" / "gone")
    client.stop()
    assert os.path.lexists(old)
