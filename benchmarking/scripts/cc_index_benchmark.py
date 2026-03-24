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

"""CC Index Lookup benchmark for nightly benchmarking.

Imports stages and helpers from tutorials/math/1_cc_index_lookup.py
(CCIndexLookupStage, collect_unique_urls, get_cc_index_files) and
orchestrates them with timing and metrics collection.

Example usage::

    python cc_index_benchmark.py \\
        --benchmark-results-path=/tmp/results \\
        --query-dataset-path=/data/cc_index_benchmark/dataset/openwebmath/data \\
        --cc-index-path=/data/cc_index_benchmark/cc_index \\
        --output-path=/tmp/output
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import ray
from loguru import logger
from utils import write_benchmark_results

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.file_utils import get_all_file_paths_under

_TUTORIAL_PATH = Path(__file__).resolve().parent.parent.parent / "tutorials" / "math" / "1_cc_index_lookup.py"
_spec = importlib.util.spec_from_file_location("cc_index_lookup", _TUTORIAL_PATH)
if _spec is None or _spec.loader is None:
    msg = (
        f"Could not load tutorial module from {_TUTORIAL_PATH}. "
        "Ensure the tutorials directory is present relative to the benchmarking scripts."
    )
    raise FileNotFoundError(msg)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CCIndexLookupStage = _mod.CCIndexLookupStage
collect_unique_urls = _mod.collect_unique_urls
get_cc_index_files = _mod.get_cc_index_files


def compute_output_metrics(output_path: str) -> dict:
    """Compute post-hoc metrics from output enriched parquet files."""
    metrics: dict = {}
    try:
        output_files = get_all_file_paths_under(output_path, keep_extensions=[".parquet"])
        metrics["num_output_files"] = len(output_files)
        total_bytes = sum(os.path.getsize(f) for f in output_files)
        metrics["output_total_mb"] = total_bytes / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not compute output metrics: {e}")
    return metrics


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the CC Index lookup benchmark following the tutorial pattern."""
    query_path = Path(args.query_dataset_path).absolute()
    cc_index_path = Path(args.cc_index_path).absolute()
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    # We need ray.put() below to broadcast query URLs, which triggers an
    # implicit ray.init() if Ray isn't initialized yet.  That bare init
    # would lack the runtime_env that XennaExecutor normally sets (see
    # XennaExecutor.run()), causing Xenna's GPU probe to fail.  Mirror
    # what XennaExecutor does: explicit ray.init with the env var BEFORE
    # any other Ray calls.
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    ray_client = RayClient()
    ray_client.start()

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            },
        },
    )

    try:
        # Step 1: Collect query URLs
        logger.info(f"Collecting unique URLs from {query_path}...")
        url_start = time.perf_counter()
        query_urls = collect_unique_urls(str(query_path), args.url_col)
        url_elapsed = time.perf_counter() - url_start
        num_query_urls = len(query_urls)
        logger.info(f"Collected {num_query_urls:,} unique URLs in {url_elapsed:.2f}s")

        # Step 2: Broadcast URLs via Ray object store
        broadcast_start = time.perf_counter()
        query_urls_ref = ray.put(query_urls)
        broadcast_elapsed = time.perf_counter() - broadcast_start
        logger.info(f"Broadcast URLs to Ray object store in {broadcast_elapsed:.2f}s")

        # Step 3: Collect CC Index files
        cc_files = get_cc_index_files(str(cc_index_path), args.crawls)
        num_cc_files = len(cc_files)
        cc_total_bytes = sum(os.path.getsize(f) for f in cc_files)
        logger.info(f"CC Index: {num_cc_files} files, {cc_total_bytes / (1024**3):.2f} GB")

        # Step 4: Build pipeline
        pipeline = Pipeline(
            name="cc_index_lookup_benchmark",
            stages=[
                FilePartitioningStage(file_paths=cc_files, blocksize=args.blocksize),
                CCIndexLookupStage(
                    query_urls_ref=query_urls_ref,
                    output_path=str(output_path),
                    url_col=args.url_col,
                ),
            ],
        )

        logger.info(f"Pipeline description:\n{pipeline.describe()}")
        logger.info("Starting CC Index lookup pipeline...")

        # Step 5: Run pipeline
        pipeline_start = time.perf_counter()
        try:
            results = pipeline.run()
            success = True
        except Exception:
            logger.exception("CC Index lookup pipeline failed")
            results = []
            success = False
        pipeline_elapsed = time.perf_counter() - pipeline_start

    finally:
        ray_client.stop()

    return _build_results(
        results=results,
        success=success,
        timings={"url": url_elapsed, "broadcast": broadcast_elapsed, "pipeline": pipeline_elapsed},
        num_query_urls=num_query_urls,
        num_cc_files=num_cc_files,
        cc_total_bytes=cc_total_bytes,
        output_path=str(output_path),
        args=args,
    )


def _build_results(  # noqa: PLR0913
    *,
    results: list,
    success: bool,
    timings: dict[str, float],
    num_query_urls: int,
    num_cc_files: int,
    cc_total_bytes: int,
    output_path: str,
    args: argparse.Namespace,
) -> dict:
    """Aggregate task results into benchmark metrics."""
    total_elapsed = sum(timings.values())
    pipeline_elapsed = timings["pipeline"]

    total_input_rows = 0
    total_matched_rows = 0
    for task in results or []:
        meta = getattr(task, "_metadata", {}) or {}
        total_input_rows += meta.get("input_rows", 0)
        total_matched_rows += meta.get("matched_rows", 0)

    metrics = {
        "is_success": success,
        "time_taken_s": total_elapsed,
        "url_collection_time_s": timings["url"],
        "broadcast_time_s": timings["broadcast"],
        "pipeline_time_s": pipeline_elapsed,
        "num_query_urls": num_query_urls,
        "num_cc_index_files": num_cc_files,
        "cc_index_total_gb": cc_total_bytes / (1024**3),
        "num_output_tasks": len(results) if results else 0,
        "total_cc_index_rows_scanned": total_input_rows,
        "total_matched_rows": total_matched_rows,
        "match_rate": total_matched_rows / total_input_rows if total_input_rows > 0 else 0,
        "throughput_cc_rows_per_sec": total_input_rows / pipeline_elapsed if pipeline_elapsed > 0 else 0,
        "throughput_gb_per_sec": (cc_total_bytes / (1024**3)) / pipeline_elapsed if pipeline_elapsed > 0 else 0,
    }

    task_metrics = TaskPerfUtils.aggregate_task_metrics(results, prefix="task")
    metrics.update(task_metrics)
    metrics.update(compute_output_metrics(output_path))

    logger.success(f"Benchmark completed in {total_elapsed:.2f}s (pipeline: {pipeline_elapsed:.2f}s)")
    logger.success(f"CC Index rows scanned: {total_input_rows:,}")
    logger.success(f"Matched rows: {total_matched_rows:,} ({metrics['match_rate']:.4%})")
    logger.success(f"Throughput: {metrics['throughput_gb_per_sec']:.2f} GB/s")

    return {
        "params": {
            "args": vars(args),
            "num_query_urls": num_query_urls,
            "num_cc_index_files": num_cc_files,
        },
        "metrics": metrics,
        "tasks": results or [],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="CC Index Lookup benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")
    p.add_argument("--query-dataset-path", required=True, help="Path to query dataset parquet (e.g. OPENWEBMATH)")
    p.add_argument("--cc-index-path", required=True, help="Path to CC Index partitions (hive layout)")
    p.add_argument("--output-path", required=True, help="Output directory for enriched parquet")
    p.add_argument("--url-col", default="url", help="URL column name in query dataset")
    p.add_argument("--blocksize", default="512MiB", help="File block size for partitioning CC Index files")
    p.add_argument(
        "--crawls",
        nargs="+",
        default=None,
        help="Crawl IDs to include (default: auto-detect all)",
    )

    args = p.parse_args()

    logger.info("=== CC Index Lookup Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results = {
        "params": {"args": vars(args)},
        "metrics": {
            "is_success": False,
            "time_taken_s": 0,
            "pipeline_time_s": 0,
            "num_query_urls": 0,
            "num_cc_index_files": 0,
            "total_cc_index_rows_scanned": 0,
            "total_matched_rows": 0,
            "throughput_cc_rows_per_sec": 0,
            "throughput_gb_per_sec": 0,
        },
        "tasks": [],
    }
    try:
        results = run_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
