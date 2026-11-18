# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run a command while recording wall time and one-second NVIDIA GPU samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                index, uuid, utilization, memory, power = (value.strip() for value in line.split(","))
                if int(index) not in gpu_indexes:
                    continue
                writer.writerow(
                    dict(
                        zip(
                            FIELDS,
                            (sampled_at, index, uuid, utilization, memory, power),
                            strict=True,
                        )
                    )
                )
            stream.flush()
            if stop.wait(1.0):
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--":
        parser.error("separate the measured command with --")
    command = args.command[1:]
    if not command:
        parser.error("measured command is empty")

    args.run_dir.mkdir(parents=True, exist_ok=False)
    gpu_csv = args.run_dir / "gpu-samples.csv"
    stop = threading.Event()
    sampler = threading.Thread(target=_sample, args=(stop, gpu_csv, set(args.gpu_index)), daemon=True)
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    sampler.start()
    with (args.run_dir / "command.log").open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())  # noqa: S603
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join()
    finished_utc = datetime.now(UTC)

    with gpu_csv.open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    per_gpu = {}
    for index in args.gpu_index:
        values = [float(row["utilization_gpu_pct"]) for row in samples if int(row["gpu_index"]) == index]
        per_gpu[str(index)] = {
            "samples": len(values),
            "mean_utilization_pct": statistics.fmean(values) if values else None,
            "min_utilization_pct": min(values) if values else None,
            "max_utilization_pct": max(values) if values else None,
        }
    all_values = [float(row["utilization_gpu_pct"]) for row in samples]
    summary = {
        "schema": "curator.parse35.benchmark.v1",
        "command": command,
        "exit_code": process.returncode,
        "started_utc": started_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "elapsed_seconds": elapsed,
        "sample_interval_seconds": 1,
        "gpu_indexes": args.gpu_index,
        "gpu_sample_rows": len(samples),
        "mean_gpu_utilization_pct": statistics.fmean(all_values) if all_values else None,
        "per_gpu": per_gpu,
    }
    (args.run_dir / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
