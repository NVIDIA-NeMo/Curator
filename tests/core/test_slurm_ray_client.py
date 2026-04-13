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

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, NoReturn

from nemo_curator.core.client import (
    RayClient,
    SlurmRayClient,
    _expand_slurm_nodelist,
    _find_ray_binary,
    _parse_slurm_nodelist,
)

# --------------------------------------------------------------------------- #
# Helper tests
# --------------------------------------------------------------------------- #


class TestFindRayBinary:
    def test_finds_ray_in_venv(self):
        binary = _find_ray_binary()
        assert os.path.isfile(binary)

    def test_raises_when_not_found(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr("os.path.isfile", lambda _: False)
        with pytest.raises(FileNotFoundError, match="ray"):
            _find_ray_binary()


class TestExpandSlurmNodelist:
    def test_single_hostname(self):
        result = _expand_slurm_nodelist("compute-001")
        assert result == ["compute-001"]

    def test_expands_with_scontrol(self, monkeypatch: pytest.MonkeyPatch):
        import nemo_curator.core.client as _client

        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="node-001\nnode-002\nnode-003\n")
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/scontrol")
        monkeypatch.setattr(_client.subprocess, "run", lambda *_args, **_kw: fake_result)
        result = _expand_slurm_nodelist("node-[001-003]")
        assert result == ["node-001", "node-002", "node-003"]

    def test_fallback_no_scontrol(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        result = _expand_slurm_nodelist("node-001")
        assert result == ["node-001"]


class TestParseSlurmNodelist:
    """Tests for the pure-Python fallback parser (no scontrol required)."""

    def test_single_node(self):
        assert _parse_slurm_nodelist("node-001") == ["node-001"]

    def test_comma_separated(self):
        assert _parse_slurm_nodelist("node-001,node-002,node-003") == [
            "node-001",
            "node-002",
            "node-003",
        ]

    def test_simple_range(self):
        assert _parse_slurm_nodelist("pool0-[01-05]") == [
            "pool0-01",
            "pool0-02",
            "pool0-03",
            "pool0-04",
            "pool0-05",
        ]

    def test_mixed_range_and_list(self):
        # prefix-[01-03,07,10-12] → 6 nodes
        result = _parse_slurm_nodelist("node-[01-03,07,10-12]")
        assert result == [
            "node-01",
            "node-02",
            "node-03",
            "node-07",
            "node-10",
            "node-11",
            "node-12",
        ]

    def test_zero_padded_range(self):
        result = _parse_slurm_nodelist("compute-[001-003]")
        assert result == ["compute-001", "compute-002", "compute-003"]

    def test_multiple_prefixes_with_ranges(self):
        # Two separate bracket groups in a comma-split list
        result = _parse_slurm_nodelist("gpu-[1-2],cpu-[3-4]")
        assert result == ["gpu-1", "gpu-2", "cpu-3", "cpu-4"]


# --------------------------------------------------------------------------- #
# SlurmRayClient unit tests
# --------------------------------------------------------------------------- #


class TestSlurmRayClientInit:
    def test_detects_slurm_cpus(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
        monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
        client = SlurmRayClient()
        assert client.num_cpus == 64
        assert client.num_gpus is None

    def test_detects_slurm_gpus(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
        client = SlurmRayClient()
        assert client.num_gpus == 8

    def test_explicit_overrides_slurm(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
        client = SlurmRayClient(num_cpus=32, num_gpus=4)
        assert client.num_cpus == 32
        assert client.num_gpus == 4

    def test_dashboard_host_defaults_to_all(self):
        client = SlurmRayClient()
        assert client.ray_dashboard_host == "0.0.0.0"  # noqa: S104


class TestSlurmRayClientFallback:
    def test_falls_back_without_slurm_job_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("RAY_ADDRESS", raising=False)

        with tempfile.TemporaryDirectory(prefix="ray_test_slurm_") as ray_tmp:
            client = SlurmRayClient(ray_temp_dir=ray_tmp)
            client.start()
            try:
                assert os.environ.get("RAY_ADDRESS") is not None
                assert client.ray_process is not None
                fn = os.path.join(ray_tmp, "ray_current_cluster")
                t0 = time.perf_counter()
                while not os.path.exists(fn) and time.perf_counter() - t0 < 30:
                    time.sleep(1)
                assert os.path.exists(fn)
            finally:
                client.stop()


class TestSlurmRayClientSingleNode:
    """Test single-node SLURM behaviour (no srun needed)."""

    def test_single_node_start_stop(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_NODELIST", os.uname().nodename)
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
        monkeypatch.delenv("RAY_ADDRESS", raising=False)

        with tempfile.TemporaryDirectory(prefix="ray_test_slurm_single_") as ray_tmp:
            client = SlurmRayClient(ray_temp_dir=ray_tmp, cleanup_on_start=False)
            client.start()
            try:
                assert os.environ.get("RAY_ADDRESS") is not None
                assert client._slurm_nodes == [os.uname().nodename]
            finally:
                client.stop()

    def test_context_manager(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_NODELIST", os.uname().nodename)
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
        monkeypatch.delenv("RAY_ADDRESS", raising=False)

        with tempfile.TemporaryDirectory(prefix="ray_test_slurm_ctx_") as ray_tmp:
            with SlurmRayClient(ray_temp_dir=ray_tmp, cleanup_on_start=False) as client:
                assert os.environ.get("RAY_ADDRESS") is not None

            assert client.ray_process is None


# --------------------------------------------------------------------------- #
# Port-file helpers
# --------------------------------------------------------------------------- #


class TestHeadPortFile:
    def test_default_uses_tmp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAY_PORT_BROADCAST_DIR", raising=False)
        client = SlurmRayClient()
        path = client._head_port_file("42")
        assert os.path.basename(path) == "ray_head_port_42"

    def test_custom_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAY_PORT_BROADCAST_DIR", str(tmp_path))
        client = SlurmRayClient()
        assert client._head_port_file("99") == str(tmp_path / "ray_head_port_99")


class TestWriteReadHeadPort:
    def test_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAY_PORT_BROADCAST_DIR", str(tmp_path))
        client = SlurmRayClient(ray_port=12345)
        client._write_head_port("job1")
        assert client._read_head_port("job1", timeout_s=5) == 12345

    def test_read_timeout_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAY_PORT_BROADCAST_DIR", str(tmp_path))
        monkeypatch.setattr(time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError, match="Timed out waiting"):
            SlurmRayClient()._read_head_port("no_such_job", timeout_s=0)

    def test_read_ignores_partial_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the port file exists but is empty/corrupt, _read_head_port retries until valid."""
        monkeypatch.setenv("RAY_PORT_BROADCAST_DIR", str(tmp_path))
        port_file = tmp_path / "ray_head_port_job2"
        write_calls = [0]
        real_sleep = time.sleep

        def patched_sleep(s: float) -> None:
            write_calls[0] += 1
            if write_calls[0] == 1:
                port_file.write_text("6379")
            real_sleep(min(s, 0.05))

        monkeypatch.setattr(time, "sleep", patched_sleep)
        port_file.write_text("")  # start with corrupt content
        assert SlurmRayClient()._read_head_port("job2", timeout_s=10) == 6379


# --------------------------------------------------------------------------- #
# _run_as_worker
# --------------------------------------------------------------------------- #


class TestRunAsWorker:
    def test_returns_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.core.client as _client

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ray")
        monkeypatch.setattr(
            _client.subprocess,
            "run",
            lambda _cmd, **_kw: subprocess.CompletedProcess(args=[], returncode=0),
        )
        assert SlurmRayClient()._run_as_worker("10.0.0.1") == 0

    def test_passes_gpu_and_cpu_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.core.client as _client

        captured: list[list[str]] = []

        def fake_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.append(cmd)
            return subprocess.CompletedProcess(args=[], returncode=0)

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ray")
        monkeypatch.setattr(_client.subprocess, "run", fake_run)
        SlurmRayClient(num_gpus=4, num_cpus=16)._run_as_worker("10.0.0.1")
        assert len(captured) == 1
        assert "--num-gpus" in captured[0]
        assert "--num-cpus" in captured[0]

    def test_nonzero_exit_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.core.client as _client

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ray")
        monkeypatch.setattr(
            _client.subprocess,
            "run",
            lambda _cmd, **_kw: subprocess.CompletedProcess(args=[], returncode=1),
        )
        assert SlurmRayClient()._run_as_worker("10.0.0.1") == 1


# --------------------------------------------------------------------------- #
# _cleanup_local_ray
# --------------------------------------------------------------------------- #


class TestCleanupLocalRay:
    def test_calls_ray_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.core.client as _client

        calls: list[list[str]] = []
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ray")
        monkeypatch.setattr(_client.subprocess, "run", lambda cmd, **_kw: calls.append(cmd))
        SlurmRayClient()._cleanup_local_ray()
        assert any("stop" in c for c in calls[0])

    def test_suppresses_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even if ray binary is missing, _cleanup_local_ray must not raise."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr("os.path.isfile", lambda _: False)
        SlurmRayClient()._cleanup_local_ray()  # should not raise


# --------------------------------------------------------------------------- #
# _expand_slurm_nodelist  (additional edge cases)
# --------------------------------------------------------------------------- #


class TestExpandSlurmNodelistEdgeCases:
    def test_scontrol_error_falls_back_to_parser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.core.client as _client

        def raise_error(*_a: object, **_kw: object) -> NoReturn:
            raise subprocess.CalledProcessError(1, "scontrol")

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/scontrol")
        monkeypatch.setattr(_client.subprocess, "run", raise_error)
        assert _expand_slurm_nodelist("node-001") == ["node-001"]

    def test_scontrol_empty_output_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_curator.core.client as _client

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/scontrol")
        monkeypatch.setattr(
            _client.subprocess,
            "run",
            lambda *_a, **_kw: subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        )
        assert _expand_slurm_nodelist("node-001") == ["node-001"]


# --------------------------------------------------------------------------- #
# _ray_init_with_timeout
# --------------------------------------------------------------------------- #


def _inject_fake_ray(
    monkeypatch: pytest.MonkeyPatch,
    init_fn: Callable[..., Any] | None = None,
    nodes_fn: Callable[[], list[Any]] | None = None,
) -> None:
    import sys
    import types

    fake_ray = types.ModuleType("ray")
    fake_ray.init = init_fn or (lambda *_a, **_kw: None)
    fake_ray.nodes = nodes_fn or list
    fake_ray.shutdown = lambda: None
    monkeypatch.setitem(sys.modules, "ray", fake_ray)


class TestRayInitWithTimeout:
    def test_main_thread_calls_ray_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        initted: list[str] = []
        _inject_fake_ray(monkeypatch, init_fn=lambda address, **_kw: initted.append(address))
        SlurmRayClient._ray_init_with_timeout("127.0.0.1:6379", timeout_s=10)
        assert initted == ["127.0.0.1:6379"]

    def test_non_main_thread_skips_sigalrm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        initted: list[str] = []
        _inject_fake_ray(monkeypatch, init_fn=lambda address, **_kw: initted.append(address))
        results: list[str] = []

        def _run() -> None:
            SlurmRayClient._ray_init_with_timeout("127.0.0.1:6379", timeout_s=5)
            results.append("done")

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=10)
        assert results == ["done"]
        assert initted == ["127.0.0.1:6379"]


# --------------------------------------------------------------------------- #
# _wait_for_workers
# --------------------------------------------------------------------------- #


class TestWaitForWorkers:
    def test_success_all_nodes_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _inject_fake_ray(
            monkeypatch,
            nodes_fn=lambda: [
                {"Alive": True, "Resources": {"CPU": 4.0, "GPU": 1.0}},
                {"Alive": True, "Resources": {"CPU": 4.0, "GPU": 1.0}},
            ],
        )
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6379")
        client = SlurmRayClient(worker_connect_timeout_s=30)
        client._slurm_nodes = ["node1", "node2"]
        client._wait_for_workers()  # must not raise

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _inject_fake_ray(monkeypatch, nodes_fn=list)  # workers never connect
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6379")
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = SlurmRayClient(worker_connect_timeout_s=0)
        client._slurm_nodes = ["node1", "node2"]
        with pytest.raises(TimeoutError, match="Timed out"):
            client._wait_for_workers()

    def test_partial_nodes_then_all_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Workers join across multiple polling iterations."""
        call_count = [0]

        def nodes_fn() -> list[dict[str, Any]]:
            call_count[0] += 1
            if call_count[0] < 3:
                return [{"Alive": True, "Resources": {}}]
            return [{"Alive": True, "Resources": {}}, {"Alive": True, "Resources": {}}]

        _inject_fake_ray(monkeypatch, nodes_fn=nodes_fn)
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6379")
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = SlurmRayClient(worker_connect_timeout_s=60)
        client._slurm_nodes = ["node1", "node2"]
        client._wait_for_workers()
        assert call_count[0] >= 3


# --------------------------------------------------------------------------- #
# SlurmRayClient.stop  (manages_cluster branch)
# --------------------------------------------------------------------------- #


class TestSlurmRayClientStopManagesCluster:
    def test_removes_port_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAY_PORT_BROADCAST_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        monkeypatch.setattr(RayClient, "stop", lambda _self: None)

        client = SlurmRayClient(ray_port=6379)
        client._manages_cluster = True
        client._write_head_port("99999")
        port_file = client._head_port_file("99999")
        assert os.path.exists(port_file)

        client.stop()
        assert not os.path.exists(port_file)

    def test_no_port_file_does_not_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAY_PORT_BROADCAST_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_JOB_ID", "88888")
        monkeypatch.setattr(RayClient, "stop", lambda _self: None)

        client = SlurmRayClient()
        client._manages_cluster = True
        client.stop()  # no port file — FileNotFoundError must be suppressed


# --------------------------------------------------------------------------- #
# SlurmRayClient.start  (additional branches)
# --------------------------------------------------------------------------- #


class TestSlurmRayClientStartBranches:
    def test_ray_address_already_set_delegates_to_super(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super_calls: list[str] = []
        monkeypatch.setattr(RayClient, "start", lambda _self: super_calls.append("start"))
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6379")
        SlurmRayClient().start()
        assert super_calls == ["start"]

    def test_head_node_multi_node_calls_write_and_wait(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Head node (SLURM_NODEID=0) with 2 nodes: must write port and wait for workers."""
        monkeypatch.setenv("SLURM_JOB_ID", "55555")
        monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[001-002]")
        monkeypatch.setenv("SLURM_NODEID", "0")
        monkeypatch.delenv("RAY_ADDRESS", raising=False)

        super_starts: list[str] = []

        def fake_super_start(_self: object) -> None:
            super_starts.append("start")
            os.environ["RAY_ADDRESS"] = "10.0.0.1:6379"

        wrote: list[str] = []
        waited: list[bool] = []
        monkeypatch.setattr(RayClient, "start", fake_super_start)
        monkeypatch.setattr(SlurmRayClient, "_cleanup_local_ray", lambda _self: None)
        monkeypatch.setattr(SlurmRayClient, "_write_head_port", lambda _self, jid: wrote.append(jid))
        monkeypatch.setattr(SlurmRayClient, "_wait_for_workers", lambda _self: waited.append(True))

        SlurmRayClient(cleanup_on_start=True).start()

        assert super_starts == ["start"]
        assert wrote == ["55555"]
        assert waited == [True]

    def test_worker_node_calls_sys_exit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Worker node (SLURM_NODEID=1) must call sys.exit with the worker's return code."""
        monkeypatch.setenv("SLURM_JOB_ID", "55556")
        monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[001-002]")
        monkeypatch.setenv("SLURM_NODEID", "1")
        monkeypatch.delenv("RAY_ADDRESS", raising=False)

        import nemo_curator.core.client as _client

        monkeypatch.setattr(SlurmRayClient, "_cleanup_local_ray", lambda _self: None)
        monkeypatch.setattr(_client.socket, "gethostbyname", lambda _: "10.0.0.1")
        monkeypatch.setattr(SlurmRayClient, "_read_head_port", lambda _self, _jid, **_kw: 6379)
        monkeypatch.setattr(SlurmRayClient, "_run_as_worker", lambda _self, _ip: 0)

        with pytest.raises(SystemExit) as exc:
            SlurmRayClient(cleanup_on_start=True).start()
        assert exc.value.code == 0
