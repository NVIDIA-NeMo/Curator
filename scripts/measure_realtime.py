# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.  # noqa: INP001
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

"""Measure realtime factor of an ongoing or completed pipeline run.

Three modes:
  Live:      Takes two snapshots separated by --interval seconds (default).
  Slurm log: Use --slurm-log to auto-extract inference time from log timestamps.
  Manual:    Use --wall-time to specify total wall time manually.

Usage:
    # Live (while job is running):
    python measure_realtime.py /path/to/output.jsonl

    # Completed run (auto-detect timing from slurm log):
    python measure_realtime.py /path/to/output.jsonl --slurm-log slurm-5046023.out

    # Completed run (manual wall time):
    python measure_realtime.py /path/to/output.jsonl --wall-time 4200
"""

import argparse
import json
import re
import time
from datetime import datetime


def measure(path: str, duration_key: str = "duration") -> tuple[int, float]:
    total_dur = 0.0
    count = 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            total_dur += entry.get(duration_key, 0)
            count += 1
    return count, total_dur


def parse_inference_time_from_log(log_path: str) -> float | None:
    """Extract inference duration by finding first and last 'generated' timestamps."""
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
    first_ts = None
    last_ts = None
    with open(log_path) as f:
        for line in f:
            if "generated" not in line and "model loaded" not in line:
                continue
            m = ts_pattern.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
            if first_ts is None:
                first_ts = ts
            last_ts = ts
    if first_ts and last_ts and last_ts > first_ts:
        return (last_ts - first_ts).total_seconds()
    return None


def print_results(count: int, audio: float, inference_time: float) -> None:
    realtime = audio / inference_time if inference_time > 0 else 0
    utt_per_sec = count / inference_time if inference_time > 0 else 0
    print()
    print("=" * 50)
    print(f"  Total utterances:   {count:,}")
    print(f"  Total audio:        {audio / 3600:.1f}h ({audio:,.0f}s)")
    print(f"  Inference time:     {inference_time:,.0f}s ({inference_time / 60:.1f}m)")
    print(f"  Realtime factor:    {audio:,.0f} / {inference_time:,.0f} = {realtime:.0f}x")
    print(f"  Utterances/sec:     {utt_per_sec:.1f}")
    print("=" * 50)


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure realtime factor from output JSONL")
    ap.add_argument("output_file", help="Path to the output JSONL file")
    ap.add_argument("--interval", type=int, default=60, help="Seconds between snapshots for live mode (default: 60)")
    ap.add_argument("--duration-key", type=str, default="duration", help="JSON key for audio duration in seconds")
    ap.add_argument(
        "--slurm-log", type=str, default=None, help="Slurm log file to auto-extract inference timing from timestamps"
    )
    ap.add_argument(
        "--wall-time", type=float, default=None, help="Total inference wall time in seconds (for completed runs)"
    )
    args = ap.parse_args()

    if args.slurm_log is not None:
        count, audio = measure(args.output_file, args.duration_key)
        inference_time = parse_inference_time_from_log(args.slurm_log)
        if inference_time is None:
            print("Could not extract timing from slurm log.")
            return
        print(f"Completed run: {args.output_file}")
        print(f"Timing from:   {args.slurm_log}")
        print_results(count, audio, inference_time)
        return

    if args.wall_time is not None:
        count, audio = measure(args.output_file, args.duration_key)
        print(f"Completed run: {args.output_file}")
        print_results(count, audio, args.wall_time)
        return

    print(f"Taking snapshot 1 of {args.output_file} ...")
    count1, audio1 = measure(args.output_file, args.duration_key)
    ts1 = time.time()
    print(f"  Utterances: {count1:,}  |  Audio: {audio1 / 3600:.1f}h")

    if count1 == 0:
        print("No data yet. Is the pipeline running?")
        return

    print(f"Waiting {args.interval}s ...")
    time.sleep(args.interval)

    print("Taking snapshot 2 ...")
    count2, audio2 = measure(args.output_file, args.duration_key)
    ts2 = time.time()
    print(f"  Utterances: {count2:,}  |  Audio: {audio2 / 3600:.1f}h")

    dt = ts2 - ts1
    d_count = count2 - count1
    d_audio = audio2 - audio1

    print()
    if d_count == 0:
        print("No new utterances. Pipeline may have finished or stalled.")
        print(f"Total: {count2:,} utterances, {audio2 / 3600:.1f}h audio")
        print("\nTip: For completed runs, use --slurm-log <file> or --wall-time <seconds>")
        return

    realtime = d_audio / dt
    utt_per_sec = d_count / dt

    print("=" * 50)
    print(f"  Wall time delta:    {dt:.0f}s")
    print(f"  Utterances delta:   {d_count:,}")
    print(f"  Audio delta:        {d_audio / 3600:.1f}h")
    print(f"  Realtime factor:    {realtime:.0f}x")
    print(f"  Utterances/sec:     {utt_per_sec:.1f}")
    print("=" * 50)
    print(f"  Total processed:    {count2:,} utterances, {audio2 / 3600:.1f}h audio")


if __name__ == "__main__":
    main()
