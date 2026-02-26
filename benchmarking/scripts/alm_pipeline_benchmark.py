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

"""ALM (Audio Language Model) pipeline benchmarking script.

This script runs the ALM data curation pipeline (Builder + Overlap stages)
through the full Pipeline/Executor stack and collects performance metrics
for regression tracking.

Can be invoked standalone with explicit args, or with --config to read
parameters from a benchmarking YAML (e.g. nightly-benchmark.yaml).
"""

import argparse
import re
import shlex
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm import (
    ALMDataBuilderStage,
    ALMDataOverlapStage,
    ALMManifestReader,
    ALMManifestWriterStage,
)
from utils import setup_executor, write_benchmark_results


def run_alm_pipeline_benchmark(  # noqa: PLR0913, PLR0915
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

    manifest_paths = [input_manifest] * max(repeat_factor, 1)
    logger.info(f"Manifest paths: {len(manifest_paths)} copies (repeat_factor={repeat_factor})")

    pipeline = Pipeline(name="alm_benchmark", description="ALM Reader + Builder + Overlap + Writer benchmark pipeline")
    pipeline.add_stage(ALMManifestReader(manifest_path=manifest_paths))
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
    output_path = str(benchmark_results_path / "alm_output.jsonl")
    pipeline.add_stage(ALMManifestWriterStage(output_path=output_path))

    exc = setup_executor(executor)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running ALM pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(exc)
        run_time_taken = time.perf_counter() - run_start_time

        output_entries = []
        for task in output_tasks or []:
            output_entries.extend(task.data)

        num_output_entries = len(output_entries)
        num_input_entries = num_output_entries
        total_builder_windows = sum(len(e.get("windows", [])) for e in output_entries)
        total_filtered_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
        total_filtered_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
        entries_with_windows = sum(1 for e in output_entries if e.get("filtered_windows") or e.get("windows"))

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Entries: {num_output_entries} (repeat_factor={repeat_factor})")
        logger.success(f"Builder windows: {total_builder_windows}, Filtered windows: {total_filtered_windows}")
        logger.success(f"Total filtered duration: {total_filtered_dur:.2f}s")
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_input_entries = 0
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


def _load_args_from_config(config_path: str, entry_name: str) -> list[str]:
    """Extract CLI args for a named entry from a benchmarking YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for entry in cfg.get("entries", []):
        if entry.get("name") == entry_name:
            raw_args = entry.get("args", "")
            curator_repo_dir = str(Path(config_path).resolve().parent.parent)
            resolved = re.sub(r"\{curator_repo_dir\}", curator_repo_dir, raw_args)
            import tempfile

            resolved = re.sub(r"\{session_entry_dir\}", tempfile.gettempdir() + "/alm_benchmark_results", resolved)
            return shlex.split(resolved)

    msg = f"Entry '{entry_name}' not found in {config_path}"
    raise ValueError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM pipeline benchmark for nightly benchmarking")
    parser.add_argument("--config", type=str, help="Path to benchmarking YAML config (e.g. nightly-benchmark.yaml)")
    parser.add_argument("--entry", type=str, default="alm_pipeline_xenna", help="Entry name in the YAML config")
    parser.add_argument("--benchmark-results-path", type=Path, help="Path to write benchmark results")
    parser.add_argument("--input-manifest", help="Path to input JSONL manifest")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"], help="Executor")
    parser.add_argument("--target-window-duration", type=float, default=120.0, help="Target window duration (seconds)")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Window duration tolerance fraction")
    parser.add_argument("--min-sample-rate", type=int, default=16000, help="Minimum audio sample rate")
    parser.add_argument("--min-bandwidth", type=int, default=8000, help="Minimum segment bandwidth")
    parser.add_argument("--min-speakers", type=int, default=2, help="Minimum speakers per window")
    parser.add_argument("--max-speakers", type=int, default=5, help="Maximum speakers per window")
    parser.add_argument("--overlap-percentage", type=int, default=50, help="Overlap filter percentage (0-100)")
    parser.add_argument("--repeat-factor", type=int, default=1, help="Multiply manifest entries by this factor for scale testing")

    pre_args, remaining = parser.parse_known_args()

    if pre_args.config:
        config_args = _load_args_from_config(pre_args.config, pre_args.entry)
        args = parser.parse_args(config_args + remaining)
    else:
        args = parser.parse_args()

    if not args.benchmark_results_path or not args.input_manifest:
        parser.error("--benchmark-results-path and --input-manifest are required (provide directly or via --config)")

    run_args = {k: v for k, v in vars(args).items() if k not in ("config", "entry")}

    logger.info("=== ALM Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {run_args}")

    success_code = 1

    result_dict = {
        "params": run_args,
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        result_dict.update(run_alm_pipeline_benchmark(**run_args))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
