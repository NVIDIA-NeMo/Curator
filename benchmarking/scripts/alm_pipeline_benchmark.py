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
parameters from a benchmarking YAML (e.g. benchmarks.yaml).
"""

import argparse
import json
import re
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader
from nemo_curator.stages.audio.alm import (
    ALMDataBuilderStage,
    ALMDataOverlapStage,
)
from nemo_curator.stages.audio.common import ManifestWriterStage


def _collect_output_metrics(
    output_tasks: list[Any], results_dir: Path, final_manifest: Path
) -> dict[str, int | float]:
    output_entries = [task.data for task in output_tasks]
    num_output_entries = len(output_entries)

    output_manifests = sorted(results_dir.glob("*.jsonl"))
    if output_manifests != [final_manifest]:
        msg = f"Expected only {final_manifest}, found {output_manifests}"
        raise RuntimeError(msg)
    num_manifest_entries = 0
    with final_manifest.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue
            if not isinstance(json.loads(line), dict):
                msg = f"ALM output line {line_number} is not a JSON object"
                raise TypeError(msg)
            num_manifest_entries += 1
    if num_manifest_entries != num_output_entries:
        msg = f"ALM output row mismatch: tasks={num_output_entries}, manifest={num_manifest_entries}"
        raise RuntimeError(msg)

    total_builder_windows = sum(len(entry.get("windows", [])) for entry in output_entries)
    return {
        "num_output_entries": num_output_entries,
        "entries_with_windows": sum(
            1 for entry in output_entries if entry.get("filtered_windows") or entry.get("windows")
        ),
        "total_builder_windows": total_builder_windows,
        "total_filtered_windows": sum(len(entry.get("filtered_windows", [])) for entry in output_entries),
        "total_filtered_dur_s": sum(entry.get("filtered_dur", 0) for entry in output_entries),
    }


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
    truncation: bool = True,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Run the ALM pipeline benchmark and collect comprehensive metrics."""
    benchmark_results_path = Path(benchmark_results_path)
    benchmark_results_path.mkdir(parents=True, exist_ok=True)
    results_dir = benchmark_results_path / "results"
    final_manifest = results_dir / "alm_output.jsonl"

    logger.info("Starting ALM pipeline benchmark")
    logger.info(f"Input manifest: {input_manifest}")
    logger.info(f"Executor: {executor}")
    if execution_mode:
        logger.info(f"Execution mode: {execution_mode}")
    logger.info(f"Window duration: {target_window_duration}s (tolerance: {tolerance})")
    logger.info(f"Sample rate >= {min_sample_rate}, Bandwidth >= {min_bandwidth}")
    logger.info(f"Speakers: {min_speakers}-{max_speakers}")
    logger.info(f"Overlap percentage: {overlap_percentage}")
    logger.info(f"Word-boundary truncation: {truncation}")

    pipeline = Pipeline(name="alm_benchmark", description="ALM Reader + Builder + Overlap benchmark pipeline")
    pipeline.add_stage(ManifestReader(manifest_path=input_manifest))
    pipeline.add_stage(
        ALMDataBuilderStage(
            target_window_duration=target_window_duration,
            tolerance=tolerance,
            min_sample_rate=min_sample_rate,
            min_bandwidth=min_bandwidth,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            truncation=truncation,
        )
    )
    pipeline.add_stage(
        ALMDataOverlapStage(
            overlap_percentage=overlap_percentage,
            target_duration=target_window_duration,
        )
    )
    pipeline.add_stage(ManifestWriterStage(output_path=str(final_manifest)))

    executor_config = {"execution_mode": execution_mode} if execution_mode else None
    exc = setup_executor(executor, config=executor_config)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running ALM pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(exc)
        output_metrics = _collect_output_metrics(output_tasks or [], results_dir, final_manifest)
        run_time_taken = time.perf_counter() - run_start_time

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Entries: {output_metrics['num_output_entries']}")
        logger.success(
            f"Builder windows: {output_metrics['total_builder_windows']}, "
            f"Filtered windows: {output_metrics['total_filtered_windows']}"
        )
        logger.success(f"Total filtered duration: {output_metrics['total_filtered_dur_s']:.2f}s")
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        output_metrics = {
            "num_output_entries": 0,
            "entries_with_windows": 0,
            "total_builder_windows": 0,
            "total_filtered_windows": 0,
            "total_filtered_dur_s": 0.0,
        }
        success = False

    return {
        "params": {
            "executor": executor,
            "input_manifest": input_manifest,
            "target_window_duration": target_window_duration,
            "tolerance": tolerance,
            "min_sample_rate": min_sample_rate,
            "min_bandwidth": min_bandwidth,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "overlap_percentage": overlap_percentage,
            "truncation": truncation,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            **output_metrics,
            "throughput_entries_per_sec": (
                output_metrics["num_output_entries"] / run_time_taken if run_time_taken > 0 else 0
            ),
            "throughput_windows_per_sec": (
                output_metrics["total_builder_windows"] / run_time_taken if run_time_taken > 0 else 0
            ),
        },
        "tasks": output_tasks or [],
    }


def _load_args_from_config(config_path: str, entry_name: str, datasets_path: Path | None = None) -> list[str]:
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
            if datasets_path is not None:
                for dataset in cfg.get("datasets", []):
                    for dataset_format in dataset.get("formats", []):
                        reference = f"{{dataset:{dataset['name']},{dataset_format['type']}}}"
                        path = dataset_format["path"].replace("{datasets_path}", str(datasets_path))
                        resolved = resolved.replace(reference, path)
            return shlex.split(resolved)

    msg = f"Entry '{entry_name}' not found in {config_path}"
    raise ValueError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM pipeline benchmark for nightly benchmarking")
    parser.add_argument("--config", type=str, help="Path to benchmarking YAML config (e.g. benchmarks.yaml)")
    parser.add_argument("--entry", type=str, default="alm_pipeline_xenna", help="Entry name in the YAML config")
    parser.add_argument("--datasets-path", type=Path, help="Dataset root used with --config")
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
    parser.add_argument(
        "--truncation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow the builder to truncate segments at word boundaries. Disable for manifests, "
            "such as AMI utterance annotations, that do not provide word timestamps."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default=None,
        choices=["streaming", "batch"],
        help="Xenna execution mode (streaming or batch). Only applies to xenna executor. Default: streaming.",
    )

    pre_args, _ = parser.parse_known_args()

    if pre_args.config:
        config_args = _load_args_from_config(pre_args.config, pre_args.entry, pre_args.datasets_path)
        args = parser.parse_args(config_args + sys.argv[1:])
    else:
        args = parser.parse_args()

    if not args.benchmark_results_path or not args.input_manifest:
        parser.error("--benchmark-results-path and --input-manifest are required (provide directly or via --config)")

    if args.input_manifest and re.search(r"\{[^}]+\}", args.input_manifest):
        parser.error("unresolved input-manifest placeholder; pass an explicit --input-manifest")

    run_args = {k: v for k, v in vars(args).items() if k not in ("config", "entry", "datasets_path")}

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
