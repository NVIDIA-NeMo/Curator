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
import subprocess
import tempfile
import time

import pytest

from nemo_curator.core.client import SlurmRayClient, _expand_slurm_nodelist, _find_ray_binary, _parse_slurm_nodelist

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

        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="node-001\nnode-002\nnode-003\n"
        )
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
