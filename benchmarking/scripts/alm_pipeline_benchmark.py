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

"""ALM (Audio Language Model) pipeline benchmarking script.

This script runs the ALM data curation pipeline (Builder + Overlap stages)
through the full Pipeline/Executor stack and collects performance metrics
for regression tracking.
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.tasks import AudioBatch


def load_manifest(manifest_path: str) -> list[dict]:
    """Load entries from a JSONL manifest file."""
    from fsspec.core import url_to_fs

    fs, path = url_to_fs(manifest_path)
    entries = []
    with fs.open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def run_alm_pipeline_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_manifest: str,
    executor: str,
    target_window_duration: float,
    tolerance: float,
    min_sample_rate: int,
    min_bandwidth: int,
    min_speakers: int,
    max_speakers: int,
    overlap_percentage: int,
    repeat_factor: int,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the ALM pipeline benchmark and collect comprehensive metrics."""
    benchmark_results_path = Path(benchmark_results_path)

    logger.info("Starting ALM pipeline benchmark")
    logger.info(f"Input manifest: {input_manifest}")
    logger.info(f"Executor: {executor}")
    logger.info(f"Repeat factor: {repeat_factor}")
    logger.info(f"Window duration: {target_window_duration}s (tolerance: {tolerance})")
    logger.info(f"Sample rate >= {min_sample_rate}, Bandwidth >= {min_bandwidth}")
    logger.info(f"Speakers: {min_speakers}-{max_speakers}")
    logger.info(f"Overlap percentage: {overlap_percentage}")

    entries = load_manifest(input_manifest)
    if repeat_factor > 1:
        entries = entries * repeat_factor
    num_input_entries = len(entries)
    logger.info(f"Loaded {num_input_entries} entries from manifest (repeat_factor={repeat_factor})")

    initial_tasks = [AudioBatch(data=[entry]) for entry in entries]

    pipeline = Pipeline(name="alm_benchmark", description="ALM Builder + Overlap benchmark pipeline")
    pipeline.add_stage(
        ALMDataBuilderStage(
            target_window_duration=target_window_duration,
            tolerance=tolerance,
            min_sample_rate=min_sample_rate,
            min_bandwidth=min_bandwidth,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    )
    pipeline.add_stage(
        ALMDataOverlapStage(
            overlap_percentage=overlap_percentage,
            target_duration=target_window_duration,
        )
    )

    exc = setup_executor(executor)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running ALM pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(exc, initial_tasks=initial_tasks)
        run_time_taken = time.perf_counter() - run_start_time

        output_entries = []
        for task in output_tasks or []:
            output_entries.extend(task.data)

        num_output_entries = len(output_entries)
        total_builder_windows = sum(len(e.get("windows", [])) for e in output_entries)
        total_filtered_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
        total_filtered_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
        entries_with_windows = sum(1 for e in output_entries if e.get("filtered_windows") or e.get("windows"))

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Input: {num_input_entries} entries -> Output: {num_output_entries} entries")
        logger.success(f"Builder windows: {total_builder_windows}, Filtered windows: {total_filtered_windows}")
        logger.success(f"Total filtered duration: {total_filtered_dur:.2f}s")
        success = True

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_output_entries = 0
        total_builder_windows = 0
        total_filtered_windows = 0
        total_filtered_dur = 0.0
        entries_with_windows = 0
        success = False

    return {
        "params": {
            "executor": executor,
            "input_manifest": input_manifest,
            "num_input_entries": num_input_entries,
            "target_window_duration": target_window_duration,
            "tolerance": tolerance,
            "min_sample_rate": min_sample_rate,
            "min_bandwidth": min_bandwidth,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "overlap_percentage": overlap_percentage,
            "repeat_factor": repeat_factor,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_input_entries": num_input_entries,
            "num_output_entries": num_output_entries,
            "entries_with_windows": entries_with_windows,
            "total_builder_windows": total_builder_windows,
            "total_filtered_windows": total_filtered_windows,
            "total_filtered_dur_s": total_filtered_dur,
            "throughput_entries_per_sec": num_input_entries / run_time_taken if run_time_taken > 0 else 0,
            "throughput_windows_per_sec": total_builder_windows / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM pipeline benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to write benchmark results")
    parser.add_argument("--input-manifest", required=True, help="Path to input JSONL manifest")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"], help="Executor")
    parser.add_argument("--target-window-duration", type=float, default=120.0, help="Target window duration (seconds)")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Window duration tolerance fraction")
    parser.add_argument("--min-sample-rate", type=int, default=16000, help="Minimum audio sample rate")
    parser.add_argument("--min-bandwidth", type=int, default=8000, help="Minimum segment bandwidth")
    parser.add_argument("--min-speakers", type=int, default=2, help="Minimum speakers per window")
    parser.add_argument("--max-speakers", type=int, default=5, help="Maximum speakers per window")
    parser.add_argument("--overlap-percentage", type=int, default=50, help="Overlap filter percentage (0-100)")
    parser.add_argument("--repeat-factor", type=int, default=1, help="Multiply manifest entries by this factor for scale testing")

    args = parser.parse_args()

    logger.info("=== ALM Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1

    result_dict = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        result_dict.update(run_alm_pipeline_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
