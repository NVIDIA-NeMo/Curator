# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run a command while recording its wall time and one-second GPU samples."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import signal
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

FIELDS = ("timestamp_utc", "gpu_index", "gpu_uuid", "utilization_gpu_pct", "memory_used_mib", "power_draw_w")


def _sample(stop: threading.Event, destination: Path, gpu_indexes: set[int]) -> None:
    query = "index,uuid,utilization.gpu,memory.used,power.draw"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        while True:
            sampled_at = datetime.now(UTC).isoformat()
            result = subprocess.run(  # noqa: S603
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            observed = set()
            for line in result.stdout.splitlines():
                index, uuid, utilization, memory, power = (value.strip() for value in line.split(","))
                if int(index) not in gpu_indexes:
                    continue
                observed.add(int(index))
                writer.writerow(dict(zip(FIELDS, (sampled_at, index, uuid, utilization, memory, power), strict=True)))
            stream.flush()
            if observed != gpu_indexes:
                msg = f"Missing GPU samples: {sorted(gpu_indexes - observed)}"
                raise RuntimeError(msg)
            if stop.wait(1.0):
                return


class GPUSampler:
    """Own the sampler thread and propagate its failures to the caller."""

    def __init__(self, destination: Path, gpu_indexes: set[int]) -> None:
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(destination, gpu_indexes), daemon=True)

    def _run(self, destination: Path, gpu_indexes: set[int]) -> None:
        try:
            _sample(self.stop_event, destination, gpu_indexes)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def check(self) -> None:
        if self.error is not None:
            msg = f"GPU sampling failed: {self.error}"
            raise RuntimeError(msg) from self.error

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                msg = "GPU sampler did not stop"
                raise RuntimeError(msg)
        self.check()


def summarize_gpus(
    destination: Path,
    gpu_indexes: set[int],
    *,
    started_utc: datetime | None = None,
    finished_utc: datetime | None = None,
) -> dict:
    """Summarize the declared window while retaining every raw observation."""
    with destination.open(newline="") if destination.exists() else contextlib.nullcontext([]) as stream:
        raw_samples = list(csv.DictReader(stream)) if destination.exists() else []
    samples = [
        row
        for row in raw_samples
        if (started_utc is None or datetime.fromisoformat(row["timestamp_utc"]) >= started_utc)
        and (finished_utc is None or datetime.fromisoformat(row["timestamp_utc"]) <= finished_utc)
    ]
    per_gpu = {}
    for index in sorted(gpu_indexes):
        values = [float(row["utilization_gpu_pct"]) for row in samples if int(row["gpu_index"]) == index]
        per_gpu[str(index)] = {
            "samples": len(values),
            "mean_utilization_pct": statistics.fmean(values) if values else None,
            "min_utilization_pct": min(values) if values else None,
            "max_utilization_pct": max(values) if values else None,
        }
    values = [float(row["utilization_gpu_pct"]) for row in samples]
    return {
        "sample_interval_seconds": 1,
        "sample_window_started_utc": started_utc.isoformat() if started_utc else None,
        "sample_window_finished_utc": finished_utc.isoformat() if finished_utc else None,
        "gpu_indexes": sorted(gpu_indexes),
        "raw_gpu_sample_rows": len(raw_samples),
        "excluded_gpu_sample_rows": len(raw_samples) - len(samples),
        "gpu_sample_rows": len(samples),
        "mean_gpu_utilization_pct": statistics.fmean(values) if values else None,
        "per_gpu": per_gpu,
    }


def _group_has_live_processes(group_id: int) -> bool:
    """Check the Linux process group without treating zombies as running work."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == group_id and fields[0] not in {"Z", "X"}:
            return True
    return False


def _wait_for_group_exit(group_id: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while _group_has_live_processes(group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def stop_command(process: subprocess.Popen) -> None:
    """Stop the owned group before reaping its leader, which pins the PGID."""
    try:
        if _group_has_live_processes(process.pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            if not _wait_for_group_exit(process.pid, 5):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                if not _wait_for_group_exit(process.pid, 5):
                    command = f"process group {process.pid}"
                    raise subprocess.TimeoutExpired(command, 5)
    finally:
        process.wait(timeout=5)


def measure_command(command: list[str], run_dir: Path, gpu_indexes: set[int]) -> int:
    """Record a command, preserving failure summaries and owned cleanup."""
    run_dir.mkdir(parents=True, exist_ok=False)
    gpu_csv = run_dir / "gpu-samples.csv"
    sampler = GPUSampler(gpu_csv, gpu_indexes)
    process = None
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    exit_code = 1
    command_exit_code = None
    error = None
    cleanup_errors = []
    cleanup_interrupt = None
    try:
        sampler.start()
        with (run_dir / "command.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
            # WNOWAIT keeps the leader's PID reserved until its whole group is stopped.
            while (completed := os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)) is None:
                sampler.check()
                time.sleep(0.05)
            exit_code = completed.si_status if completed.si_code == os.CLD_EXITED else -completed.si_status
            command_exit_code = exit_code
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finished_utc = datetime.now(UTC)
        elapsed = time.perf_counter() - started
        cleanup_started = time.perf_counter()
        for label, cleanup in (
            ("Command cleanup", lambda: stop_command(process) if process is not None else None),
            ("GPU sampler cleanup", sampler.stop),
        ):
            try:
                cleanup()
            except BaseException as exc:  # noqa: BLE001 - finish all cleanup before reraising interrupts
                cleanup_errors.append(f"{label} failed: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception):
                    cleanup_interrupt = exc
        if cleanup_errors:
            exit_code = 1
            error = error or "; ".join(cleanup_errors)
        cleanup_elapsed = time.perf_counter() - cleanup_started
        summary = {
            "schema": "curator.parse35.benchmark.v1",
            "command": command,
            "exit_code": exit_code,
            "command_exit_code": command_exit_code,
            "error": error,
            "cleanup_errors": cleanup_errors,
            "started_utc": started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "elapsed_seconds": elapsed,
            "cleanup_elapsed_seconds": cleanup_elapsed,
            **summarize_gpus(gpu_csv, gpu_indexes, started_utc=started_utc, finished_utc=finished_utc),
        }
        (run_dir / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if cleanup_interrupt is not None:
            raise cleanup_interrupt
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        parser.error("separate a nonempty measured command with --")
    if any(index < 0 for index in args.gpu_index):
        parser.error("GPU indexes must be nonnegative")

    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(measure_command(args.command[1:], args.run_dir, set(args.gpu_index)))


if __name__ == "__main__":
    main()
