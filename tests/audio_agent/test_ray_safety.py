# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Ray bootstrap safety regressions; all Ray/process interactions are mocked."""

from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path  # noqa: TC003
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from nemo_curator.audio_agent import _ray


@pytest.fixture(autouse=True)
def _isolated_bootstrap_state(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
    saved = dict(_ray._STARTED)
    _ray._STARTED.clear()
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_MAX_LIMIT_FROM_API_SERVER", raising=False)
    yield
    _ray._STARTED.clear()
    _ray._STARTED.update(saved)
    # ``ensure_cluster`` sets RAY_ADDRESS through os.environ directly, and monkeypatch has
    # no undo recorded for a variable that was absent at delenv time. Leaving it set makes a
    # LATER test (e.g. smoke bounding) try to reach a cluster that no longer exists and hang.
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_MAX_LIMIT_FROM_API_SERVER", None)


@pytest.mark.parametrize(
    "address",
    ["auto", "ray://ray.example.test:10001", "10.20.30.40:6379"],
)
def test_every_external_ray_address_is_validated_and_never_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    address: str,
) -> None:
    probes: list[str] = []
    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda candidate: probes.append(candidate) or {"CPU": 4.0},
    )
    monkeypatch.setattr(
        "nemo_curator.core.client.RayClient",
        lambda **_kwargs: pytest.fail("external addresses must never reach local bootstrap"),
    )

    assert _ray.ensure_cluster() == address
    assert probes == [address]
    assert os.environ["RAY_ADDRESS"] == address
    assert _ray._STARTED == {}


def test_failed_external_connection_fails_closed_without_local_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = "ray://unreachable.example.test:10001"
    monkeypatch.setenv("RAY_ADDRESS", address)

    def fail_probe(_address: str) -> dict[str, float]:
        raise ConnectionError("unreachable")  # noqa: EM101

    monkeypatch.setattr(_ray, "cluster_resources", fail_probe)
    monkeypatch.setattr(
        "nemo_curator.core.client.RayClient",
        lambda **_kwargs: pytest.fail("failed external connection must not start a local head"),
    )

    with pytest.raises(RuntimeError, match="refusing to replace it"):
        _ray.ensure_cluster()

    assert os.environ["RAY_ADDRESS"] == address
    assert _ray._STARTED == {}


def test_preinitialized_ray_without_environment_is_reused_not_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = "127.0.0.1:6379"
    fake_ray = SimpleNamespace(is_initialized=lambda: True)
    probes: list[str] = []
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(_ray, "_connected_address", lambda _module: address)
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda candidate: probes.append(candidate) or {"CPU": 4.0},
    )
    monkeypatch.setattr(
        "nemo_curator.core.client.RayClient",
        lambda **_kwargs: pytest.fail("an initialized driver must not trigger local bootstrap"),
    )

    assert _ray.ensure_cluster() == address
    assert probes == [address]
    assert "RAY_ADDRESS" not in os.environ
    assert _ray._STARTED == {}


def test_cluster_probe_refuses_an_already_initialized_address_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = False

    def resources() -> dict[str, float]:
        nonlocal queried
        queried = True
        return {"CPU": 8.0}

    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=resources,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:6379",
    )

    with pytest.raises(RuntimeError, match="mismatched address"):
        _ray.cluster_resources("127.0.0.1:6380")

    assert queried is False


def test_cluster_probe_reads_matching_initialized_cluster_without_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"CPU": 8, "GPU": 1.0, "label": "ignored"},
        shutdown=lambda: pytest.fail("pre-existing Ray connection must stay initialized"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:6379",
    )

    assert _ray.cluster_resources("localhost:6379") == {"CPU": 8.0, "GPU": 1.0}


def test_cluster_probe_disconnects_a_new_connection_that_resolves_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdowns = 0
    queried = False

    def shutdown() -> None:
        nonlocal shutdowns
        shutdowns += 1

    def resources() -> dict[str, float]:
        nonlocal queried
        queried = True
        return {"CPU": 8.0}

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **_kwargs: None,
        cluster_resources=resources,
        shutdown=shutdown,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:9999",
    )

    with pytest.raises(RuntimeError, match="mismatched address"):
        _ray.cluster_resources("127.0.0.1:6379")

    assert queried is False
    assert shutdowns == 1


class _FakeRayClient:
    """Stand-in for the shared ``RayClient``, with the same externally visible contract.

    The real one starts ``ray start --block`` under ``Popen(start_new_session=True)`` and
    stops it with ``killpg`` on that process group. Everything this module needs to be
    correct about is at that seam: it owns ``RAY_ADDRESS``, and stopping is one call.
    """

    instances: ClassVar[list[_FakeRayClient]] = []
    address = "127.0.0.1:6381"

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        self.kwargs = kwargs
        self.started = False
        self.stops = 0
        self.start_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        _FakeRayClient.instances.append(self)

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        os.environ["RAY_ADDRESS"] = self.address

    def stop(self) -> None:
        self.stops += 1
        if self.stop_error is not None:
            raise self.stop_error
        os.environ.pop("RAY_ADDRESS", None)


@pytest.fixture(autouse=True)
def _reset_fake_clients():  # noqa: ANN202
    _FakeRayClient.instances.clear()
    yield
    _FakeRayClient.instances.clear()


def _mock_local_start(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    """Route the bootstrap through a fake RayClient and a known session directory."""
    temp_dir.mkdir(exist_ok=True)
    monkeypatch.setattr("nemo_curator.core.client.RayClient", _FakeRayClient)
    monkeypatch.setattr(_ray, "_detect_gpus", lambda: 0)
    # The stub ignores ``dir=``, so the root has to be pinned to the one the session
    # directory is actually under. Otherwise the harness records a root the directory does
    # not live in -- a disagreement no real bootstrap can produce, and one that would make
    # the containment gate look broken here while being right in production.
    monkeypatch.setattr(_ray, "_ray_temp_root", lambda: os.path.realpath(temp_dir.parent))
    monkeypatch.setattr(_ray.tempfile, "mkdtemp", lambda **_kwargs: str(temp_dir))


def test_the_bootstrap_goes_through_the_shared_ray_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Curator already solves starting and stopping Ray; the agent must not re-solve it.

    A private ``ray start`` without ``--block`` turns off Ray's own cleanup
    (``shutdown_at_exit``/``spawn_reaper``), which is what forced teardown to hunt for
    detached processes by scanning /proc -- where every real defect in this file lived.
    """
    temp_dir = tmp_path / "ray_audio_agent_shared"
    _mock_local_start(monkeypatch, temp_dir)

    address = _ray.ensure_cluster(num_cpus=3, object_store_memory=1234)

    assert address == _FakeRayClient.address
    client = _FakeRayClient.instances[-1]
    assert client.started
    assert client.kwargs["ray_temp_dir"] == str(temp_dir)
    assert client.kwargs["num_cpus"] == 3
    assert client.kwargs["object_store_memory"] == 1234
    # The agent is a library caller: registering with a shared Prometheus/Grafana install
    # would be a side effect on state it does not own.
    assert client.kwargs["include_dashboard"] is False


def test_owned_local_cluster_is_cleaned_and_environment_is_restored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_owned"
    _mock_local_start(monkeypatch, temp_dir)

    address = _ray.ensure_cluster()

    assert _ray.owns_cluster(address)
    assert _ray._STARTED["owner_pid"] == os.getpid()
    assert os.environ["RAY_ADDRESS"] == address
    assert temp_dir.exists()

    assert _ray.shutdown_cluster(address) is True

    assert _FakeRayClient.instances[-1].stops == 1
    assert not temp_dir.exists()
    assert "RAY_ADDRESS" not in os.environ
    assert "RAY_MAX_LIMIT_FROM_API_SERVER" not in os.environ
    assert _ray._STARTED == {}


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("did not become responsive"), KeyboardInterrupt()],
    ids=["unresponsive", "interrupted"],
)
def test_a_failed_start_stops_the_client_and_removes_the_temp_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    """RayClient stops what it started; the session directory is ours to remove.

    Ctrl-C is included deliberately: a ``KeyboardInterrupt`` is not an ``Exception``, so a
    bare ``except Exception`` would let it orphan a live head with no ownership recorded.
    """
    temp_dir = tmp_path / "ray_audio_agent_failed"
    _mock_local_start(monkeypatch, temp_dir)

    original_start = _FakeRayClient.start

    def failing_start(self: _FakeRayClient) -> None:
        self.start_error = failure
        original_start(self)

    monkeypatch.setattr(_FakeRayClient, "start", failing_start)

    with pytest.raises(type(failure)):
        _ray.ensure_cluster()

    assert _FakeRayClient.instances[-1].stops == 1
    assert not temp_dir.exists()
    assert "RAY_ADDRESS" not in os.environ
    assert _ray._STARTED == {}


def test_shutdown_refuses_address_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_mismatch"
    _mock_local_start(monkeypatch, temp_dir)
    address = _ray.ensure_cluster()

    assert _ray.shutdown_cluster("127.0.0.1:9999") is False
    assert _FakeRayClient.instances[-1].stops == 0
    assert _ray.owns_cluster(address)
    assert temp_dir.exists()


def test_shutdown_refuses_inherited_ownership_from_another_process(
    tmp_path: Path,
) -> None:
    """A forked child inherits ``_STARTED`` by copy-on-write but owns nothing."""
    temp_dir = tmp_path / "ray_audio_agent_inherited"
    temp_dir.mkdir()
    address = "127.0.0.1:6381"
    _ray._STARTED.update(
        {
            "address": address,
            "temp_dir": str(temp_dir),
            "client": _FakeRayClient(),
            "owned": True,
            "owner_pid": os.getpid() + 1,  # a different process started it
        }
    )
    os.environ["RAY_ADDRESS"] = address

    assert _ray.shutdown_cluster(address) is False
    assert _FakeRayClient.instances[-1].stops == 0
    assert temp_dir.exists()


def test_a_failed_stop_preserves_state_and_temp_directory_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_retry"
    _mock_local_start(monkeypatch, temp_dir)
    address = _ray.ensure_cluster()
    _FakeRayClient.instances[-1].stop_error = OSError("killpg failed")

    assert _ray.shutdown_cluster(address) is False
    assert _ray.owns_cluster(address)
    assert temp_dir.exists()
    assert os.environ["RAY_ADDRESS"] == address


def test_shutdown_refuses_a_driver_connected_to_a_different_cluster(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Disconnecting a driver aimed elsewhere would tear down someone else's session."""
    temp_dir = tmp_path / "ray_audio_agent_driver"
    _mock_local_start(monkeypatch, temp_dir)
    address = _ray.ensure_cluster()

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: True,
            shutdown=lambda: pytest.fail("a driver on another cluster must not be shut down"),
        ),
    )
    monkeypatch.setattr(_ray, "_connected_address", lambda _module: "127.0.0.1:7777")

    assert _ray.shutdown_cluster(address) is False
    assert _FakeRayClient.instances[-1].stops == 0
    assert _ray.owns_cluster(address)
    assert temp_dir.exists()


def test_an_unverifiable_session_dir_refuses_before_touching_the_callers_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal promises "nothing changed, retry safely" -- so it must cost nothing.

    ``ray.shutdown()`` is irreversible and closes the CALLER's session, dropping their actor
    handles and object refs. Running it before a gate that can still refuse broke that
    promise: shutdown reported failure while the caller's Ray connection was already gone.
    """
    shutdowns = 0

    def shutdown() -> None:
        nonlocal shutdowns
        shutdowns += 1

    address = "127.0.0.1:6399"
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(is_initialized=lambda: True, shutdown=shutdown))
    monkeypatch.setattr(_ray, "_connected_address", lambda _module: address)
    _ray._STARTED.update(
        {
            "address": address,
            "temp_dir": "/not/a/temp/root/ray_audio_agent_x",  # fails _safe_temp_dir
            "client": _FakeRayClient(),
            "owned": True,
            "owner_pid": os.getpid(),
        }
    )
    os.environ["RAY_ADDRESS"] = address

    assert _ray.shutdown_cluster(address) is False
    assert shutdowns == 0, "a refusal must not close the caller's Ray session"
    assert _ray.owns_cluster(address), "ownership must survive for a retry"


def test_cleanup_asks_about_the_root_it_used_not_the_one_configured_now(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Containment is a question about where the directory WAS created.

    Re-deriving the candidate roots at shutdown asks about the environment as it is then. A
    caller that set ``TMPDIR`` for the run and cleared it afterwards -- or an HPC job whose
    scratch variable is gone by teardown -- made that check fail on a directory this module
    had just created itself. Shutdown then refused identically on every retry, so the head
    stayed up and the tree leaked, on the one path whose entire job is cleanup.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    temp_dir = scratch / "ray_audio_agent_scratch"
    _mock_local_start(monkeypatch, temp_dir)
    address = _ray.ensure_cluster()

    monkeypatch.delenv("TMPDIR", raising=False)  # the run is over; the scratch var is gone

    assert _ray.shutdown_cluster(address) is True
    assert not temp_dir.exists()


def test_a_blank_ray_address_is_refused_rather_than_quietly_going_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set-but-empty is a broken instruction, and must not read as an absent one.

    Every branch asks "is it truthy" or "is it truthy and non-blank", so whitespace slid past
    the authoritative-external check and started a LOCAL head -- the exact substitution this
    function refuses to make for any other external value, performed without a word.
    """
    monkeypatch.setenv("RAY_ADDRESS", "   ")

    with pytest.raises(RuntimeError, match="names no cluster"):
        _ray.ensure_cluster()


def test_a_bootstrap_that_dies_before_starting_leaves_no_directory_behind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The import and the client construction sat outside the cleanup.

    An ImportError from a partial install, or a constructor rejecting its arguments, escaped
    with the temp directory already on disk and nothing holding a reference to remove it --
    and a broken install fails on every attempt, so it leaked one directory per try into the
    very root the object store then had to fit inside.
    """
    temp_dir = tmp_path / "ray_audio_agent_ctor"
    _mock_local_start(monkeypatch, temp_dir)

    def exploding_client(**_kwargs: Any) -> None:  # noqa: ANN401
        msg = "no acceptable RayClient"
        raise TypeError(msg)

    monkeypatch.setattr("nemo_curator.core.client.RayClient", exploding_client)

    with pytest.raises(TypeError, match="no acceptable RayClient"):
        _ray.ensure_cluster()

    assert not temp_dir.exists()
    assert _ray._STARTED == {}
    assert "RAY_MAX_LIMIT_FROM_API_SERVER" not in os.environ


def test_an_embedded_interpreter_does_not_put_the_working_directorys_parent_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Embedded interpreters leave ``sys.executable`` empty.

    ``abspath("")`` is the working directory, so its PARENT was prepended to PATH -- and an
    unrelated executable named ``ray`` sitting there would have been started as the head.
    """
    (tmp_path / "ray").write_text("#!/bin/sh\nexit 1\n")
    (tmp_path / "ray").chmod(0o755)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(sys, "executable", "")
    before = os.environ.get("PATH")

    with _ray._interpreter_ray_on_path():
        assert os.environ.get("PATH") == before


def test_the_socket_budget_leaves_room_for_the_path_ray_actually_binds() -> None:
    """The budget must cover Ray's whole suffix, not just our own directory.

    A root that passes this check but yields an over-long plasma socket does not fail at
    bind time -- ``ray start`` aborts with an AF_UNIX length error, which is a mysterious
    failure for exactly the long per-job scratch paths honouring $TMPDIR exists to support.
    Checked against Ray's own validator so its limit, not our copy of it, is the authority.
    """
    root = "/" + "x" * (_ray._AF_UNIX_ROOT_BUDGET - 1)
    assert len(root) == _ray._AF_UNIX_ROOT_BUDGET

    session = f"session_2026-08-06_18-16-46_301241_{4194304}"  # worst-case 7-digit pid
    socket_path = f"{root}/ray_audio_agent_{'a' * 8}/{session}/sockets/plasma_store"

    assert len(socket_path.encode()) <= _ray._AF_UNIX_MAX

    from ray._private.utils import validate_socket_filepath

    validate_socket_filepath(socket_path)  # Ray's own check must accept it


def test_the_reuse_probe_agrees_with_the_module_on_ipv6_addresses() -> None:
    """Ray brackets the IPv6 literals it advertises, and the probe must parse them.

    ``_address_identity`` accepted ``[::1]:6379`` while ``_reachable`` split on the last
    colon and handed ``getaddrinfo`` the bracketed host, so a live IPv6 head always probed
    as unreachable and cluster reuse always refused to reuse it.
    """
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.bind(("::1", 0))
    sock.listen(1)
    address = f"[::1]:{sock.getsockname()[1]}"
    try:
        assert _ray._address_identity(address) is not None, "the module accepts this form"
        assert _ray._reachable(address) is True, "so the probe must too"
    finally:
        sock.close()
    assert _ray._reachable(address) is False


class _RecordingRayClient(_FakeRayClient):
    """Records what a spawned child would resolve ``ray`` to, at the moment of the spawn."""

    resolved: ClassVar[list[str | None]] = []

    def start(self) -> None:
        _RecordingRayClient.resolved.append(shutil.which("ray"))
        super().start()


class TestTheInterpretersRayIsReachableByTheProcessWeSpawn:
    """``RayClient`` runs ``Popen(["ray", ...])``, so only PATH decides whether that resolves.

    Delegating the bootstrap to the shared client dropped the interpreter-relative lookup the
    agent's own bootstrap had (``dirname(sys.executable)/ray`` first, PATH second). The
    invocation this tool documents -- ``.venv/bin/python -m nemo_curator.audio_agent ...
    --bootstrap-ray`` -- runs the right interpreter without activating the venv, so the next
    real session died with ``FileNotFoundError: 'ray'`` before starting anything, against a
    flag that advertises "no manual setup".
    """

    def _venv_without_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """An interpreter with ``ray`` beside it, and a PATH that cannot see either."""
        bindir = tmp_path / "venv" / "bin"
        bindir.mkdir(parents=True)
        cli = bindir / "ray"
        cli.write_text("#!/bin/sh\nexit 0\n")
        cli.chmod(0o755)
        monkeypatch.setattr(sys, "executable", str(bindir / "python"))
        monkeypatch.setenv("PATH", "/nonexistent-bin")
        return bindir

    def _bootstrap(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        session = tmp_path / "session"
        session.mkdir(exist_ok=True)
        monkeypatch.setattr("nemo_curator.core.client.RayClient", _RecordingRayClient)
        monkeypatch.setattr(_ray, "_detect_gpus", lambda: 0)
        monkeypatch.setattr(_ray.tempfile, "mkdtemp", lambda **_kwargs: str(session))
        _RecordingRayClient.resolved.clear()
        _ray.ensure_cluster()

    def test_it_resolves_although_path_never_mentions_the_virtualenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bindir = self._venv_without_path(tmp_path, monkeypatch)

        self._bootstrap(tmp_path, monkeypatch)

        assert _RecordingRayClient.resolved == [str(bindir / "ray")]

    def test_the_interpreters_ray_wins_over_one_already_on_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Restores the old precedence: next to the interpreter first, PATH second."""
        bindir = self._venv_without_path(tmp_path, monkeypatch)
        other = tmp_path / "system" / "bin"
        other.mkdir(parents=True)
        (other / "ray").write_text("#!/bin/sh\nexit 0\n")
        (other / "ray").chmod(0o755)
        monkeypatch.setenv("PATH", str(other))

        self._bootstrap(tmp_path, monkeypatch)

        assert _RecordingRayClient.resolved == [str(bindir / "ray")]

    def test_the_callers_environment_is_left_as_it_was_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``smoke``/``run`` are library calls; a verb must not permanently edit PATH."""
        self._venv_without_path(tmp_path, monkeypatch)
        before = os.environ["PATH"]

        self._bootstrap(tmp_path, monkeypatch)

        assert os.environ["PATH"] == before

    def test_an_interpreter_with_no_ray_beside_it_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bindir = tmp_path / "bare" / "bin"
        bindir.mkdir(parents=True)
        monkeypatch.setattr(sys, "executable", str(bindir / "python"))
        monkeypatch.setenv("PATH", "/nonexistent-bin")

        self._bootstrap(tmp_path, monkeypatch)

        assert _RecordingRayClient.resolved == [None], "nothing to offer, so nothing is claimed"
        assert os.environ["PATH"] == "/nonexistent-bin"
