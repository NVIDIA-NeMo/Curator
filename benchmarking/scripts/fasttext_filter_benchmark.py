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

"""FastText Filter benchmarking script.

This script benchmarks FastText-based document filters (language ID and quality)
using a Hydra-configured pipeline and various executors.
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
    config_path = config_path.resolve()
    with initialize_config_dir(
        config_dir=str(config_path.parent),
        job_name="fasttext_filter_benchmark",
        version_base=None,
    ):
        return compose(config_name=config_path.stem, overrides=overrides)


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    pipeline = Pipeline(name="fasttext_filter_pipeline")
    for stage_cfg in cfg.stages:
        stage = hydra.utils.instantiate(stage_cfg)
        pipeline.add_stage(stage)
    return pipeline


def run_fasttext_filter_benchmark(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    executor_name: str,
    benchmark_results_path: Path,
    yaml_config: Path,
    fasttext_langid_model_path: Path,
    fasttext_quality_model_path: Path,
    overrides: str | None = None,
) -> dict[str, Any]:
    executor = setup_executor(executor_name)

    input_path = input_path.absolute()
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Executor: {executor_name}")
    logger.info(f"FastText pipeline config: {yaml_config}")
    logger.info(f"FastText language ID model: {fasttext_langid_model_path}")
    logger.info(f"FastText quality model: {fasttext_quality_model_path}")

    overrides_list = [
        f"input_path={input_path}",
        f"output_path={output_path}",
        f"fasttext_langid_model_path={fasttext_langid_model_path}",
        f"fasttext_quality_model_path={fasttext_quality_model_path}",
    ]
    if overrides:
        overrides_list.extend(overrides.split(","))

    cfg = load_hydra_yaml(yaml_config, overrides_list)
    pipeline = create_pipeline_from_yaml(cfg)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running FastText filter pipeline...")
        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # Stage assumptions:
        # 0 = partitioning (if any)
        # 1 = reader
        # -1 = writer (num_items_processed equals documents kept after all filters)
        num_documents_processed = sum(task._stage_perf[1].num_items_processed for task in output_tasks)
        num_kept_documents = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_documents_processed} documents")
        logger.success(f"Kept {num_kept_documents} documents")

        success = True

    except Exception as e:  # noqa: BLE001
        logger.error(f"Benchmark failed: {e}")
        logger.debug(traceback.format_exc())
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
            "yaml_config": str(yaml_config),
            "fasttext_langid_model_path": str(fasttext_langid_model_path),
            "fasttext_quality_model_path": str(fasttext_quality_model_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "num_kept_documents": num_kept_documents,
            "num_output_tasks": len(output_tasks),
            "throughput_docs_per_sec": (num_documents_processed / run_time_taken if run_time_taken > 0 else 0),
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FastText filter benchmark")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./fasttext_filter_output"),
    )
    parser.add_argument(
        "--executor",
        default="ray_data",
        choices=["ray_data", "xenna"],
    )
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument(
        "--fasttext-langid-model-path", type=Path, required=True, help="Path to FastText language ID model"
    )
    parser.add_argument(
        "--fasttext-quality-model-path", type=Path, required=True, help="Path to FastText quality model"
    )
    parser.add_argument("--overrides", type=str)

    args = parser.parse_args()

    logger.info("=== FastText Filter Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_fasttext_filter_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            executor_name=args.executor,
            benchmark_results_path=args.benchmark_results_path,
            yaml_config=args.yaml_config,
            fasttext_langid_model_path=args.fasttext_langid_model_path,
            fasttext_quality_model_path=args.fasttext_quality_model_path,
            overrides=args.overrides,
        )
    except Exception:  # noqa: BLE001
        results = {
            "params": vars(args),
            "metrics": {"is_success": False},
            "tasks": [],
        }
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
