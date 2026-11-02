# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Exercise command ownership and telemetry failures without GPU hardware."""

import importlib
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


def test_sampler_failure_stops_command_and_marks_run_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("benchmark_run")

    def fail(*_args) -> None:
        msg = "nvidia-smi failed"
        raise OSError(msg)

    monkeypatch.setattr(module, "_sample", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        module.measure_command([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "run", {0})
    summary = json.loads((tmp_path / "run/run-summary.json").read_text())
    assert summary["exit_code"] == 1
    assert summary["gpu_sample_rows"] == 0
    assert "sampling failed" in summary["error"]


@pytest.mark.parametrize("exit_code", [3, -signal.SIGTERM])
def test_command_exit_and_observed_samples_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exit_code: int
):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("benchmark_run")

    def sample(stop: threading.Event, destination: Path, _indexes: set[int]) -> None:
        timestamp = datetime.now(UTC).isoformat()
        destination.write_text(",".join(module.FIELDS) + f"\n{timestamp},0,GPU-test,17,100,20\n")
        stop.wait(10)

    monkeypatch.setattr(module, "_sample", sample)
    original_stop = module.stop_command

    def stop_unreaped_leader(process: subprocess.Popen) -> None:
        assert process.returncode is None
        assert os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        original_stop(process)

    monkeypatch.setattr(module, "stop_command", stop_unreaped_leader)
    script = "raise SystemExit(3)" if exit_code == 3 else "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    result = module.measure_command([sys.executable, "-c", script], tmp_path / "run", {0})
    summary = json.loads((tmp_path / "run/run-summary.json").read_text())
    assert result == summary["exit_code"] == summary["command_exit_code"] == exit_code
    assert summary["per_gpu"]["0"]["samples"] == 1
    assert summary["mean_gpu_utilization_pct"] == 17
    assert summary["raw_gpu_sample_rows"] == 1
    assert summary["sample_window_started_utc"] == summary["started_utc"]
    assert summary["sample_window_finished_utc"] == summary["finished_utc"]
    assert summary["cleanup_elapsed_seconds"] >= 0
    assert summary["error"] is None


def test_group_cleanup_escalates_after_leader_exits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("benchmark_run")
    process = SimpleNamespace(pid=42, returncode=0, wait=Mock(return_value=0))
    signals = Mock()
    monkeypatch.setattr(module, "_group_has_live_processes", lambda _group: True, raising=False)
    monkeypatch.setattr(module, "_wait_for_group_exit", Mock(side_effect=[False, True]), raising=False)
    monkeypatch.setattr(module.os, "killpg", signals)
    module.stop_command(process)
    assert [call.args for call in signals.call_args_list] == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
    process.wait.assert_called_once_with(timeout=5)


@pytest.mark.parametrize("sampler_fails", [False, True])
def test_command_cleanup_failure_still_stops_sampler_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sampler_fails: bool
):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("benchmark_run")
    sampler = SimpleNamespace(
        start=Mock(),
        check=Mock(),
        stop=Mock(side_effect=RuntimeError("sampler stop failed") if sampler_fails else None),
    )

    def fail_stop(process: subprocess.Popen) -> None:
        process.wait(timeout=5)
        command = "owned group"
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(module, "GPUSampler", lambda *_args: sampler)
    monkeypatch.setattr(module, "stop_command", fail_stop)
    result = module.measure_command([sys.executable, "-c", "pass"], tmp_path / "run", {0})
    sampler.stop.assert_called_once()
    summary = json.loads((tmp_path / "run/run-summary.json").read_text())
    assert result == summary["exit_code"] == 1
    assert summary["command_exit_code"] == 0
    assert len(summary["cleanup_errors"]) == 1 + sampler_fails
    assert "Command cleanup failed" in summary["error"]
    if sampler_fails:
        assert "GPU sampler cleanup failed" in summary["error"]


def test_gpu_summary_excludes_outside_window_without_changing_raw_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("benchmark_run")
    samples = tmp_path / "gpu-samples.csv"
    samples.write_text(
        ",".join(module.FIELDS)
        + "\n2026-11-02T00:00:00+00:00,0,GPU-test,0,100,20"
        + "\n2026-11-02T00:00:01+00:00,0,GPU-test,20,100,20"
        + "\n2026-11-02T00:00:02+00:00,0,GPU-test,40,100,20"
        + "\n2026-11-02T00:00:03+00:00,0,GPU-test,100,100,20\n"
    )
    raw = samples.read_bytes()
    summary = module.summarize_gpus(
        samples,
        {0},
        started_utc=datetime(2026, 11, 2, 0, 0, 1, tzinfo=UTC),
        finished_utc=datetime(2026, 11, 2, 0, 0, 2, tzinfo=UTC),
    )
    assert summary["raw_gpu_sample_rows"] == 4
    assert summary["gpu_sample_rows"] == summary["per_gpu"]["0"]["samples"] == 2
    assert summary["excluded_gpu_sample_rows"] == 2
    assert summary["mean_gpu_utilization_pct"] == summary["per_gpu"]["0"]["mean_utilization_pct"] == 30
    assert samples.read_bytes() == raw
