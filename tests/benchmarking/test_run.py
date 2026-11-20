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

import errno
import fcntl
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

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


def _datasets_path_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("CURATOR_BENCHMARK_PATH_MODE", "host")
    return Session.from_dict(
        {
            "paths": [
                {"name": "results_path", "host_path": str(tmp_path / "results")},
                {"name": "datasets_path", "host_path": str(tmp_path / "datasets")},
            ],
            "entries": [{"name": "test_entry", "script": "benchmark.py"}],
        }
    ).path_resolver


def test_data_setup_lock_uses_a_lock_file_in_datasets_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with benchmark_run._data_setup_lock("dataset_a", _datasets_path_resolver(tmp_path, monkeypatch)):
        assert (tmp_path / "datasets" / ".dataset_a.setup.lock").is_file()


def test_data_setup_lock_times_out_while_another_job_holds_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path_resolver = _datasets_path_resolver(tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark_run, "_DATA_SETUP_LOCK_POLL_S", 0.01)
    lock_path = tmp_path / "datasets" / ".dataset_a.setup.lock"
    lock_path.parent.mkdir(parents=True)

    with lock_path.open("a") as other_job:
        fcntl.flock(other_job.fileno(), fcntl.LOCK_EX)
        with (
            pytest.raises(TimeoutError, match="dataset_a"),
            benchmark_run._data_setup_lock("dataset_a", path_resolver, timeout_s=0.05),
        ):
            pass


def test_data_setup_lock_runs_unlocked_when_flock_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_resolver = _datasets_path_resolver(tmp_path, monkeypatch)

    def unsupported(*_: object) -> None:
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(benchmark_run.fcntl, "flock", unsupported)
    entered = False
    with benchmark_run._data_setup_lock("dataset_a", path_resolver, timeout_s=1):
        entered = True

    assert entered


@pytest.mark.skipif(os.geteuid() == 0, reason="root can open a read-only file for writing")
def test_data_setup_lock_runs_unlocked_when_the_lock_file_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_resolver = _datasets_path_resolver(tmp_path, monkeypatch)
    lock_path = tmp_path / "datasets" / ".dataset_a.setup.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.touch(mode=0o444)
    entered = False
    with benchmark_run._data_setup_lock("dataset_a", path_resolver, timeout_s=1):
        entered = True

    assert entered


def test_run_data_setups_keeps_files_of_concurrent_jobs_in_the_shared_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CURATOR_BENCHMARK_PATH_MODE", "host")
    session = Session.from_dict(
        {
            "paths": [{"name": "results_path", "host_path": str(tmp_path / "results")}],
            "datasets": [{"name": "dataset_a", "formats": [{"type": "jsonl", "path": "a.jsonl"}]}],
            "data_setups": [{"name": "dataset_a", "script": "prepare_dataset.py"}],
            "entries": [{"name": "uses_a", "script": "benchmark.py", "args": "--input {dataset:dataset_a,jsonl}"}],
        }
    )
    monkeypatch.setattr(benchmark_run, "run_command_with_timeout", lambda **_: {"returncode": 0, "timed_out": False})
    other_jobs_files = [
        tmp_path / "data_setup" / "dataset_b" / "results.json",
        tmp_path / "data_setup" / "dataset_a" / "logs" / "stdouterr.log",
    ]
    for path in other_jobs_files:
        path.parent.mkdir(parents=True)
        path.write_text("another job")

    failed = benchmark_run.run_data_setups(
        session.data_setups, session.path_resolver, session.dataset_resolver, tmp_path
    )

    assert failed == set()
    assert [path.read_text() for path in other_jobs_files] == ["another job", "another job"]


def test_run_data_setups_runs_every_setup_and_returns_the_failed_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted = []

    def run_data_setup_entry(*, setup_entry: SimpleNamespace, **_: object) -> bool:
        attempted.append(setup_entry.name)
        return setup_entry.name == "good"

    monkeypatch.setattr(benchmark_run, "run_data_setup_entry", run_data_setup_entry)
    setups = [SimpleNamespace(name=name) for name in ("bad_a", "good", "bad_b")]

    failed = benchmark_run.run_data_setups(setups, Mock(), Mock(), tmp_path)

    assert attempted == ["bad_a", "good", "bad_b"]
    assert failed == {"bad_a", "bad_b"}


def test_failed_data_setup_blocks_only_the_entries_that_use_its_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CURATOR_BENCHMARK_PATH_MODE", "host")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": [
                    {"name": "results_path", "host_path": str(tmp_path / "results")},
                    {"name": "datasets_path", "host_path": str(tmp_path / "datasets")},
                ],
                "datasets": [{"name": "dataset_a", "formats": [{"type": "jsonl", "path": "{datasets_path}/a.jsonl"}]}],
                "data_setups": [{"name": "dataset_a", "script": "prepare_dataset.py"}],
                "entries": [
                    {"name": "uses_dataset", "script": "benchmark.py", "args": "--input {dataset:dataset_a,jsonl}"},
                    {"name": "independent", "script": "benchmark.py"},
                ],
            }
        )
    )
    ran = []
    sink = Mock()
    monkeypatch.setattr(benchmark_run, "run_data_setup_entry", lambda **_: False)
    monkeypatch.setattr(benchmark_run, "run_entry", lambda *, entry, **_: ran.append(entry.name) or True)
    monkeypatch.setattr(benchmark_run, "dump_env", lambda **_: {})
    monkeypatch.setattr(benchmark_run, "initialize_sinks", lambda *_, **__: [sink])
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path), "--session-name", "session"])

    assert benchmark_run.main() == 1
    assert ran == ["independent"]
    finished = {
        call.kwargs["result_dict"]["name"]: call.kwargs["result_dict"]
        for call in sink.register_benchmark_entry_finished.call_args_list
    }
    assert finished["uses_dataset"]["success"] is False
    assert "dataset_a" in finished["uses_dataset"]["error"]
    assert "error" not in finished["independent"]
