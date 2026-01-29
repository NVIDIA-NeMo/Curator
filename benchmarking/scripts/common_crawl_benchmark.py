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

"""Common Crawl download+extract benchmark for nightly benchmarking.

Runs the text Common Crawl pipeline and writes params/metrics/tasks to the
benchmark results directory, compatible with the nightly driver.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.download.common_crawl.stage import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


def create_common_crawl_pipeline(  # noqa: PLR0913
    download_dir: Path,
    output_dir: Path,
    output_format: Literal["parquet", "jsonl"],
    crawl_type: Literal["main", "news"],
    start_snapshot: str,
    end_snapshot: str,
    html_extraction_algorithm: str = "justext",
    use_aws_to_download: bool = False,
    verbose: bool = False,
    url_limit: int | None = None,
    record_limit: int | None = None,
    add_filename_column: bool = False,
    ray_data_cast_as_actor: bool = False,
) -> Pipeline:
    if ray_data_cast_as_actor:
        os.environ["CAST_AS_ACTOR"] = "true"

    pipeline = Pipeline(name="common_crawl_processing", description="Download and process Common Crawl data")

    pipeline.add_stage(
        CommonCrawlDownloadExtractStage(
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            download_dir=str(download_dir),
            crawl_type=crawl_type,
            html_extraction=html_extraction_algorithm,
            use_aws_to_download=use_aws_to_download,
            verbose=verbose,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
    )

    if output_format == "jsonl":
        writer = JsonlWriter(path=str(output_dir))
    elif output_format == "parquet":
        writer = ParquetWriter(path=str(output_dir))
    else:
        msg = f"Invalid output format: {output_format}"
        raise ValueError(msg)

    pipeline.add_stage(writer)

    return pipeline


def run_benchmark(args: argparse.Namespace) -> dict:
    download_dir = Path(args.download_path).resolve()
    download_dir.mkdir(exist_ok=True, parents=True)

    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    pipeline = create_common_crawl_pipeline(
        download_dir=download_dir,
        output_dir=output_dir,
        output_format=args.output_format,
        crawl_type=args.crawl_type,
        start_snapshot=args.start_snapshot,
        end_snapshot=args.end_snapshot,
        html_extraction_algorithm=args.html_extraction,
        use_aws_to_download=args.aws,
        verbose=args.verbose,
        url_limit=args.url_limit,
        record_limit=args.record_limit,
        add_filename_column=args.add_filename_column,
        ray_data_cast_as_actor=args.ray_data_cast_as_actor,
    )

    executor = setup_executor(args.executor)

    logger.info("Starting Common Crawl pipeline execution...")
    start = time.perf_counter()

    try:
        results = pipeline.run(executor, initial_tasks=None)
        success = True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed: {e}")
        results = []
        success = False
    elapsed = time.perf_counter() - start

    total_documents = sum(task.num_items for task in results) if results else 0

    return {
        "params": {
            "download_path": str(download_dir),
            "output_path": str(output_dir),
            "output_format": args.output_format,
            "crawl_type": args.crawl_type,
            "start_snapshot": args.start_snapshot,
            "end_snapshot": args.end_snapshot,
            "html_extraction": args.html_extraction,
            "aws": args.aws,
            "verbose": args.verbose,
            "url_limit": args.url_limit,
            "record_limit": args.record_limit,
            "add_filename_column": args.add_filename_column,
            "ray_data_cast_as_actor": args.ray_data_cast_as_actor,
            "executor": args.executor,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": elapsed,
            "num_output_tasks": len(results) if results else 0,
            "total_documents": total_documents,
        },
        "tasks": results or [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Common Crawl download/extract benchmark")
    # Contract arg for nightly driver
    p.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")
    # Pipeline configuration
    p.add_argument("--download_path", type=str, default="./common_crawl_downloads")
    p.add_argument("--output_path", type=str, default="./common_crawl_output")
    p.add_argument("--output_format", type=str, default="parquet", choices=["parquet", "jsonl"])
    p.add_argument("--crawl_type", type=str, default="main", choices=["main", "news"])
    p.add_argument("--start_snapshot", type=str, default="2023-01")
    p.add_argument("--end_snapshot", type=str, default="2023-10")
    p.add_argument("--html_extraction", type=str, default="justext", choices=["justext", "resiliparse", "trafilatura"])
    p.add_argument("--aws", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--url_limit", type=int, default=5)
    p.add_argument("--record_limit", type=int, default=None)
    p.add_argument("--add_filename_column", action="store_true")
    # Executor selection
    p.add_argument("--executor", type=str, default="xenna", choices=["xenna", "ray_data", "ray_actors"])
    p.add_argument("--ray_data_cast_as_actor", action="store_true")

    args = p.parse_args()
    results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
