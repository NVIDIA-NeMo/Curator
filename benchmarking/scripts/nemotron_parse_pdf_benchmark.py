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

"""Nemotron-Parse PDF pipeline benchmarking script.

Reuses the pipeline and argparser from
tutorials/interleaved/nemotron_parse_pdf/main.py with comprehensive
metrics collection.
"""

import argparse
import contextlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import collect_parquet_output_metrics, setup_executor, write_benchmark_results

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "interleaved" / "nemotron_parse_pdf"))

from main import (  # noqa: E402
    create_nemotron_parse_pdf_argparser,
    create_nemotron_parse_pdf_pipeline,
)


def run_nemotron_parse_pdf_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Nemotron-Parse PDF benchmark and collect metrics."""
    executor = setup_executor(args.executor)

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"PDF source: zip_base_dir={args.zip_base_dir}, pdf_dir={args.pdf_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {args.model_path}, backend={args.backend}")
    logger.info(f"PDFs per task: {args.pdfs_per_task}, max PDFs: {args.max_pdfs}")

    pipeline = create_nemotron_parse_pdf_pipeline(args)

    run_start_time = time.perf_counter()
    success = False
    output_tasks: list = []

    try:
        logger.info("Running Nemotron-Parse PDF pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        unique_samples: set[str] = set()
        for task in output_tasks:
            if hasattr(task, "data") and task.data is not None and hasattr(task.data, "column"):
                with contextlib.suppress(Exception):
                    unique_samples.update(task.data.column("sample_id").to_pylist())

        num_pdfs_processed = len(unique_samples)
        parquet_metrics = collect_parquet_output_metrics(output_dir)

        stage_perf: dict[str, list[float]] = {}
        for task in output_tasks:
            for perf in task._stage_perf:
                stage_perf.setdefault(perf.stage_name, []).append(perf.process_time)

        stage_summary = {}
        for stage_name, times in stage_perf.items():
            stage_summary[stage_name] = {
                "count": len(times),
                "total_s": sum(times),
                "mean_s": sum(times) / len(times) if times else 0,
                "min_s": min(times) if times else 0,
                "max_s": max(times) if times else 0,
            }

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_pdfs_processed} PDFs")
        logger.success(
            f"Output: {parquet_metrics.get('num_rows', 0)} rows in {parquet_metrics.get('num_output_files', 0)} files"
        )
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        run_time_taken = time.perf_counter() - run_start_time
        num_pdfs_processed = 0
        parquet_metrics = {}
        stage_summary = {}

    return {
        "params": {
            "executor": args.executor,
            "manifest": args.manifest,
            "pdf_dir": args.pdf_dir,
            "zip_base_dir": args.zip_base_dir,
            "output_dir": str(output_dir),
            "benchmark_results_path": str(args.benchmark_results_path),
            "model_path": args.model_path,
            "backend": args.backend,
            "pdfs_per_task": args.pdfs_per_task,
            "max_pdfs": args.max_pdfs,
            "dpi": args.dpi,
            "max_pages": args.max_pages,
            "inference_batch_size": args.inference_batch_size,
            "max_num_seqs": args.max_num_seqs,
            "resume": args.resume,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_pdfs_processed": num_pdfs_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_pdfs_per_sec": num_pdfs_processed / run_time_taken if run_time_taken > 0 else 0,
            **parquet_metrics,
            "stage_performance": stage_summary,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = create_nemotron_parse_pdf_argparser()

    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        required=True,
        help="Path to write benchmark results",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for pipeline execution",
    )

    args = parser.parse_args()

    logger.info("=== Nemotron-Parse PDF Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        results = run_nemotron_parse_pdf_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
