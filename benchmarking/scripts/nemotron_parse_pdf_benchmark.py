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
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "interleaved" / "nemotron_parse_pdf"))

from main import (  # noqa: E402
    create_nemotron_parse_pdf_argparser,
    create_nemotron_parse_pdf_pipeline,
)

from nemo_curator.tasks.utils import TaskPerfUtils  # noqa: E402


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _sample_ids_from_table(data: Any) -> set[str]:  # noqa: ANN401
    """Pull sample ids from an in-memory Arrow table, if that is what the task carries."""
    if data is None or not hasattr(data, "column"):
        return set()
    try:
        return set(data.column("sample_id").to_pylist())
    except Exception:  # column may be absent for non-interleaved payloads
        return set()


def _sample_ids_from_metadata(task: Any) -> set[str]:  # noqa: ANN401
    """Derive sample ids from the manifest entries recorded in task metadata.

    The parquet writer returns a ``FileGroupTask`` whose ``data`` is a list of
    written file paths, so the sample ids only survive in ``_metadata``.
    """
    metadata = getattr(task, "_metadata", None) or {}
    ids: set[str] = set()
    for entry in metadata.get("source_files") or []:
        if not isinstance(entry, str):
            continue
        name = entry
        try:
            record = json.loads(entry)
        except ValueError:
            pass  # plain filename rather than a serialized manifest record
        else:
            if isinstance(record, dict) and record.get("file_name"):
                name = record["file_name"]
        # Mirror PDFPreprocessStage's sample_id convention exactly: strip only the
        # extension, keeping any directory prefix. Using Path().stem here would
        # collapse "a/x.pdf" and "b/x.pdf" onto the same id and undercount.
        ids.add(name.rsplit(".", 1)[0])
    return ids


def _count_unique_pdfs(output_tasks: list) -> int:
    """Count distinct source PDFs represented in the pipeline output.

    Handles both shapes the pipeline emits: a materialized Arrow table carrying
    ``sample_id``, and the default parquet-writer output where the ids are only
    recoverable from ``_metadata['source_files']``.
    """
    unique: set[str] = set()
    for task in output_tasks:
        ids = _sample_ids_from_table(getattr(task, "data", None))
        if not ids:
            ids = _sample_ids_from_metadata(task)
        unique |= ids
    return len(unique)


def _compute_pdf_parse_metrics(output_tasks: list, run_time_taken: float) -> dict[str, float]:
    """Compute benchmark-level throughput metrics from additive task stats."""
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    metric_prefix = "task_nemotron_parse_inference_custom"

    num_valid_pages = task_metrics.get(f"{metric_prefix}.num_valid_pages_sum", 0.0)
    total_output_tokens = task_metrics.get(f"{metric_prefix}.total_output_tokens_sum", 0.0)

    return {
        # Surfaced as first-class metrics (not just throughput denominators) so
        # entries can assert on work actually completed rather than on wall-clock
        # rates, which vary with cluster load. Page count is exactly reproducible
        # for a fixed input; token count is not (dynamic batching shifts where the
        # model emits EOS), so it is asserted as a band rather than an exact value.
        "num_pages_processed": num_valid_pages,
        "num_output_tokens": total_output_tokens,
        "throughput_pages_per_sec": _safe_div(num_valid_pages, run_time_taken),
        "throughput_output_tokens_per_sec": _safe_div(total_output_tokens, run_time_taken),
    }


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

        num_pdfs_processed = _count_unique_pdfs(output_tasks)
        pdf_parse_metrics = _compute_pdf_parse_metrics(output_tasks, run_time_taken)

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_pdfs_processed} PDFs")
        logger.success(f"Page throughput: {pdf_parse_metrics['throughput_pages_per_sec']:.2f} pages/s")
        logger.success(
            f"Output token throughput: {pdf_parse_metrics['throughput_output_tokens_per_sec']:.2f} tokens/s"
        )
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        run_time_taken = time.perf_counter() - run_start_time
        num_pdfs_processed = 0
        # Keep the metric keys stable across success and failure so entry
        # requirements always have a value to compare against.
        pdf_parse_metrics = {
            "num_pages_processed": 0.0,
            "num_output_tokens": 0.0,
            "throughput_pages_per_sec": 0.0,
            "throughput_output_tokens_per_sec": 0.0,
        }

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
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_pdfs_processed": num_pdfs_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_pdfs_per_sec": num_pdfs_processed / run_time_taken if run_time_taken > 0 else 0,
            **pdf_parse_metrics,
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
