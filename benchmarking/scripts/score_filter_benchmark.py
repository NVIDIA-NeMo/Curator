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

"""ScoreFilter benchmarking script.

This script runs heuristic- and model-based score filtering benchmarks
with comprehensive metrics collection using various executors and logs results to configured sinks.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline


def load_hydra_yaml(config_path: Path, overrides: list[str] | None = None) -> DictConfig:
    config_path = Path(config_path).resolve()

    with initialize_config_dir(
        config_dir=str(config_path.parent),
        job_name="app",
        version_base=None,
    ):
        return compose(config_name=config_path.stem, overrides=overrides)


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    pipeline = Pipeline(name="score_filter_pipeline")

    # Add stages to the pipeline
    for p in cfg.stages:
        stage = hydra.utils.instantiate(p)
        pipeline.add_stage(stage)

    return pipeline


def run_score_filter_benchmark(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    executor_name: str,
    benchmark_results_path: Path,
    yaml_config: Path,
    overrides: str | None = None,
) -> dict[str, Any]:
    """Run the ScoreFilter benchmark and collect comprehensive metrics."""

    executor = setup_executor(executor_name)

    input_path = input_path.absolute()

    # Ensure output directory
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.debug(f"Executor: {executor}")

    # Load YAML configuration and create pipeline
    overrides_list = [
        f"input_path={input_path}",
        f"output_path={output_path}",
    ]
    if overrides is not None:
        overrides_list.extend(overrides.split(","))

    cfg = load_hydra_yaml(yaml_config, overrides_list)
    pipeline = create_pipeline_from_yaml(cfg)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running ScoreFilter pipeline...")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # _stage_perf[0] is the file partitioning stage, so _stage_perf[1] is the file reading stage
        num_documents_processed = sum(task._stage_perf[1].num_items_processed for task in output_tasks)
        num_kept_documents = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_documents_processed} rows (documents)")
        logger.success(f"Kept {num_kept_documents} out of {num_documents_processed} rows (documents)")
        success = True

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_documents_processed = 0
        num_kept_documents = 0
        success = False

    return {
        "params": {
            "executor": executor_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "benchmark_results_path": str(benchmark_results_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "num_kept_documents": num_kept_documents,
            "num_output_tasks": len(output_tasks),
            "throughput_docs_per_sec": num_documents_processed / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ScoreFilter benchmark")
    # Paths
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input data")
    parser.add_argument(
        "--output-path", default=Path("./score_filter_output"), type=Path, help="Output directory for results"
    )
    # Executor
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    # Pipeline-specific
    parser.add_argument(
        "--yaml-config", required=True, type=Path, help="Path to YAML file containing pipeline configuration"
    )
    # example: --overrides="stages.0._target_=nemo_curator.stages.text.io.reader.ParquetReader,stages.0.files_per_partition=10"  # noqa: ERA001
    parser.add_argument("--overrides", type=str, help="Overrides to pass to the YAML configuration")

    args = parser.parse_args()

    logger.info("=== ScoreFilter Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_score_filter_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            executor_name=args.executor,
            benchmark_results_path=args.benchmark_results_path,
            yaml_config=args.yaml_config,
            overrides=args.overrides,
        )

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        print(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        results = {
            "params": vars(args),
            "metrics": {
                "is_success": False,
            },
            "tasks": [],
        }
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
