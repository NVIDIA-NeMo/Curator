"""Example script for downloading and processing Common Crawl data."""

import argparse
import time
from pathlib import Path
from pprint import pprint
from typing import Literal

from loguru import logger

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.download.text.common_crawl import CommonCrawlDownloadExtractStage
from ray_curator.stages.io.writer import JsonlWriter, ParquetWriter
from ray_curator.tasks import _EmptyTask


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
) -> Pipeline:
    """Create a pipeline for downloading and processing Common Crawl data.

    Args:
        download_dir: Directory to download WARC files to
        output_dir: Directory to write output files to
        output_format: Format of output files (parquet or jsonl)
        crawl_type: Type of Common Crawl (main or news)
        start_snapshot: Start snapshot string (YYYY-WW for main, YYYY-MM for news)
        end_snapshot: End snapshot string (YYYY-WW for main, YYYY-MM for news)
        html_extraction_algorithm: Algorithm to use for HTML extraction
        use_aws_to_download: Whether to use AWS S3 for downloading
        verbose: Whether to print verbose output
        limit: Limit the number of WARC files to process

    Returns:
        Pipeline: Configured pipeline
    """
    # Define pipeline
    pipeline = Pipeline(name="common_crawl_processing", description="Download and process Common Crawl data")

    # Add Common Crawl pipeline stage
    # The CommonCrawlPipeline is a CompositeStage that will be decomposed
    # into its constituent stages during pipeline building
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

    # Add writer stage
    if output_format == "jsonl":
        writer = JsonlWriter(output_dir=str(output_dir))
    elif output_format == "parquet":
        writer = ParquetWriter(output_dir=str(output_dir))
    else:
        msg = f"Invalid output format: {output_format}"
        raise ValueError(msg)

    pipeline.add_stage(writer)

    return pipeline


def main(args: argparse.Namespace) -> None:
    """Main function to run the pipeline."""
    # Create directories
    download_dir = Path(args.download_path).resolve()
    download_dir.mkdir(exist_ok=True, parents=True)

    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create pipeline
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
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    if args.executor == "xenna":
        executor = XennaExecutor()
    elif args.executor == "ray_data":
        from ray_curator.backends.experimental.ray_data import RayDataExecutor

        executor = RayDataExecutor()
    elif args.executor == "ray_actors":
        from ray_curator.backends.experimental.ray_actors import RayActorExecutor

        executor = RayActorExecutor()
    else:
        msg = f"Invalid executor type: {args.executor}"
        raise ValueError(msg)

    # Create initial task
    initial_task = _EmptyTask(task_id="common_crawl_task", dataset_name="common_crawl", data=None)

    # Execute pipeline
    print("Starting pipeline execution...")
    results = pipeline.run(executor, initial_tasks=[initial_task])

    # Print results summary
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    total_documents = sum(task.num_items for task in results) if results else 0
    print(f"Total documents processed: {total_documents}")

    # Print sample of results
    if results:
        print("\nSample of processed documents:")
        for task in results:
            pprint(task)
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process Common Crawl data")

    # Pipeline configuration
    parser.add_argument(
        "--download_path", type=str, default="./common_crawl_downloads", help="Directory to download WARC files to"
    )
    parser.add_argument(
        "--output_path", type=str, default="./common_crawl_output", help="Directory to write output files to"
    )
    parser.add_argument(
        "--output_format", type=str, default="parquet", choices=["parquet", "jsonl"], help="Format of output files"
    )

    # Common Crawl configuration
    parser.add_argument(
        "--crawl_type", type=str, default="main", choices=["main", "news"], help="Type of Common Crawl (main or news)"
    )
    parser.add_argument(
        "--start_snapshot",
        type=str,
        default="2023-01",
        help="Start snapshot string (YYYY-WW for main, YYYY-MM for news)",
    )
    parser.add_argument(
        "--end_snapshot", type=str, default="2023-10", help="End snapshot string (YYYY-WW for main, YYYY-MM for news)"
    )
    parser.add_argument(
        "--html_extraction",
        type=str,
        default="justext",
        choices=["justext", "resiliparse", "trafilatura"],
        help="Algorithm to use for HTML extraction",
    )
    parser.add_argument("--aws", action="store_true", help="Use AWS S3 for downloading")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--url_limit", type=int, default=5, help="Limit the number of WARC files to process")
    parser.add_argument("--record_limit", type=int, default=None, help="Limit the number of records to process")
    parser.add_argument("--add_filename_column", action="store_true", help="Add filename column to output")

    # Executor configuration
    parser.add_argument(
        "--executor",
        type=str,
        default="xenna",
        choices=["xenna", "ray_data", "ray_actors"],
        help="Executor to use for pipeline",
    )

    args = parser.parse_args()
    start_time = time.perf_counter()
    main(args)
    end_time = time.perf_counter()
    logger.success(f"Time taken: {end_time - start_time} seconds")
