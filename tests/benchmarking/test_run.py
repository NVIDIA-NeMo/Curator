# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking"))

import run as benchmark_run
from runner.session import Session


def test_run_entry_forwards_only_resolved_configured_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CURATOR_BENCHMARK_PATH_MODE", "host")
    monkeypatch.setenv("UNCONFIGURED_PARENT_MARKER", "parent")
    monkeypatch.setenv("RAY_DATA_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("NEMO_CURATOR_RAY_DATA_DIAGNOSTICS", raising=False)
    session = Session.from_dict(
        {
            "paths": [{"name": "results_path", "host_path": str(tmp_path)}],
            "environment": {"SHARED": "session"},
            "entries": [
                {
                    "name": "test_entry",
                    "script": "benchmark.py",
                    "environment": {
                        "SHARED": "entry",
                        "UCX_TLS": "all",
                        "HF_HOME": "{session_entry_dir}/scratch/hf_cache",
                    },
                }
            ],
        }
    )
    setup = Mock(side_effect=RuntimeError("stop before cluster startup"))
    monkeypatch.setattr(benchmark_run, "setup_ray_cluster_and_env", setup)
    monkeypatch.setattr(benchmark_run, "teardown_ray_cluster_and_env", Mock())
    entry_path = tmp_path / "test_entry"

    with pytest.raises(RuntimeError, match="stop before cluster startup"):
        benchmark_run.run_entry(session.entries[0], session.path_resolver, session.dataset_resolver, entry_path, {})

    setup.assert_called_once()
    assert setup.call_args.kwargs["environment"] == {
        "SHARED": "entry",
        "UCX_TLS": "all",
        "HF_HOME": str(entry_path / "scratch" / "hf_cache"),
    }
    assert setup.call_args.kwargs["ray_data_log_level"] == "WARNING"
