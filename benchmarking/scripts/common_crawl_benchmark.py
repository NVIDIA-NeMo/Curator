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
import json
import os
import time
from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path
from typing import Literal

import requests
from loguru import logger
from text_metrics import MeasureText
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.download.base.iterator import DocumentIterateExtractStage
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader
from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor
from nemo_curator.stages.text.download.common_crawl.stage import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.tasks import FileGroupTask

MAX_WARC_FILES = 3


def verify_warc_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    objects = manifest["objects"]
    prefix = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-18/"
    if manifest["snapshot"] != "CC-MAIN-2024-18" or not 1 <= len(objects) <= MAX_WARC_FILES:
        msg = "Expected 1-3 explicit CC-MAIN-2024-18 WARC objects"
        raise ValueError(msg)
    if len({obj["url"] for obj in objects}) != len(objects) or any(
        not obj["url"].startswith(prefix)
        or not obj["url"].endswith(".warc.gz")
        or type(obj["size_bytes"]) is not int
        or obj["size_bytes"] <= 0
        for obj in objects
    ):
        msg = "Invalid or duplicate WARC object in sample manifest"
        raise ValueError(msg)
    if sum(obj["size_bytes"] for obj in objects) > 5 * 1024**3:
        msg = "WARC sample exceeds the 5 GiB compressed-object cap"
        raise ValueError(msg)
    for obj in objects:
        response = requests.head(obj["url"], timeout=30, allow_redirects=False)
        response.raise_for_status()
        if (
            response.status_code != HTTPStatus.OK
            or int(response.headers.get("Content-Length", -1)) != obj["size_bytes"]
        ):
            msg = f"Remote WARC size/status changed: {obj['url']}"
            raise ValueError(msg)
        if obj.get("etag") and response.headers.get("ETag") != obj["etag"]:
            msg = f"Remote WARC ETag changed: {obj['url']}"
            raise ValueError(msg)
    return manifest


class CompletedWarcIterator(CommonCrawlWarcIterator):
    def iterate(self, file_path: str) -> Iterator[dict]:
        count = 0
        for record in super().iterate(file_path):
            count += 1
            yield record
        # The extraction stage catches iterator errors, so require this EOF receipt.
        Path(str(file_path) + ".iteration.json").write_text(json.dumps({"response_records": count}))


class BoundedWarcDownloader(CommonCrawlWARCDownloader):
    def __init__(self, directory: str, objects: list[dict]):
        super().__init__(directory)
        self.objects = {obj["url"]: obj for obj in objects}
        self.receipts = []

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        obj = self.objects[url]
        start = time.perf_counter()
        size = 0
        with requests.get(
            url, stream=True, timeout=(30, 120), allow_redirects=False, headers={"Accept-Encoding": "identity"}
        ) as response:
            response.raise_for_status()
            if (
                response.status_code != HTTPStatus.OK
                or int(response.headers.get("Content-Length", -1)) != obj["size_bytes"]
            ):
                msg = "GET status/size differs from the verified WARC object"
                raise ValueError(msg)
            if obj.get("etag") and response.headers.get("ETag") != obj["etag"]:
                msg = "GET ETag differs from the verified WARC object"
                raise ValueError(msg)
            with Path(path).open("xb") as stream:
                for chunk in response.iter_content(chunk_size=1024**2):
                    size += len(chunk)
                    if size > obj["size_bytes"]:
                        msg = "WARC response exceeded its verified byte cap"
                        raise ValueError(msg)
                    stream.write(chunk)
            if size != obj["size_bytes"]:
                msg = "Incomplete WARC response"
                raise ValueError(msg)
            self.receipts.append(
                {
                    "url": url,
                    "downloaded_bytes": size,
                    "seconds": time.perf_counter() - start,
                    "cache_headers": {
                        k: response.headers[k] for k in ["X-Cache", "Age", "Via", "ETag"] if k in response.headers
                    },
                }
            )
        return True, None


def run_manifest_benchmark(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.warc_manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[1] / "fixtures" / manifest_path
    manifest = verify_warc_manifest(manifest_path)
    download_dir, output_dir = Path(args.download_path).resolve(), Path(args.output_path).resolve()
    # Exclusive directories prevent a repeated submission being labeled a fresh download.
    download_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    downloader = BoundedWarcDownloader(str(download_dir), manifest["objects"])
    params = {
        "warc_manifest": manifest,
        "download_path": str(download_dir),
        "output_path": str(output_dir),
        "output_format": args.output_format,
        "html_extraction": args.html_extraction,
        "executor": args.executor,
        "record_limit": None,
        "cache_semantics": "Fresh exclusive local directory; no local cache reuse. Upstream/CDN cache unknown. Download bytes are completed compressed object bodies, not wire/retry traffic.",
        "timing_boundary": "Sequential acquisition then full-file extraction/measurement/write; excludes HEAD preflight, construction and final validation/serialization. No record limit. Existing iterator skips corrupt records and extractor can reject unsupported/empty content.",
    }
    metrics = {
        "is_success": False,
        "time_taken_s": None,
        "input_compressed_bytes": None,
        "num_output_documents": None,
        "output_text_bytes": None,
        "cached_files_before": 0,
    }
    tasks = []
    try:
        start = time.perf_counter()
        paths = []
        for obj in manifest["objects"]:
            path = downloader.download(obj["url"])
            if path is None or Path(path).stat().st_size != obj["size_bytes"]:
                msg = f"Missing or incomplete WARC download: {obj['url']}"
                raise ValueError(msg)  # noqa: TRY301
            paths.append(path)
        acquisition_seconds = time.perf_counter() - start
        params["acquisition_receipts"] = downloader.receipts
        writer = JsonlWriter if args.output_format == "jsonl" else ParquetWriter
        pipeline = Pipeline(
            name="explicit_warc_extract",
            stages=[
                DocumentIterateExtractStage(
                    iterator=CompletedWarcIterator(),
                    extractor=CommonCrawlHTMLExtractor(algorithm=args.html_extraction),
                    record_limit=None,
                    add_filename_column=True,
                ),
                MeasureText("output"),
                writer(path=str(output_dir)),
            ],
        )
        executor = setup_executor(args.executor)
        initial = [
            FileGroupTask(dataset_name=manifest["snapshot"], data=[path], _metadata={"source_files": [path]})
            for path in paths
        ]
        start = time.perf_counter()
        tasks = pipeline.run(executor, initial_tasks=initial)
        extraction_seconds = time.perf_counter() - start
        receipts = [json.loads(Path(path + ".iteration.json").read_text()) for path in paths]
        documents = sum(t._metadata["text_volume"]["num_output_documents"] for t in tasks)
        text_bytes = sum(t._metadata["text_volume"]["output_text_bytes"] for t in tasks)
        if len(tasks) != len(paths) or documents <= 0 or text_bytes <= 0:
            msg = "WARC sample must finish every file and produce nonzero text"
            raise ValueError(msg)  # noqa: TRY301
        metrics.update(
            is_success=True,
            time_taken_s=acquisition_seconds + extraction_seconds,
            acquisition_time_s=acquisition_seconds,
            extraction_write_time_s=extraction_seconds,
            input_compressed_bytes=sum(Path(p).stat().st_size for p in paths),
            downloaded_compressed_bytes=sum(obj["size_bytes"] for obj in manifest["objects"]),
            num_input_warc_files=len(paths),
            num_completed_warc_files=len(receipts),
            input_response_records=sum(r["response_records"] for r in receipts),
            num_output_documents=documents,
            output_text_bytes=text_bytes,
            output_file_bytes=sum(Path(p).stat().st_size for t in tasks for p in t.data),
        )
    except Exception as exc:
        logger.exception("Bounded WARC sample failed")
        params["failure"] = str(exc)
    return {"params": params, "metrics": metrics, "tasks": tasks}


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
    except Exception as e:
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
    p.add_argument(
        "--warc-manifest", help="Explicit bounded sample; relative paths resolve under benchmarking/fixtures"
    )
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

    p.add_argument(
        "--save-tasks", action=argparse.BooleanOptionalAction, default=True, help="Save completed tasks in tasks.pkl"
    )
    args = p.parse_args()
    save_tasks = vars(args).pop("save_tasks")
    results = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results = run_manifest_benchmark(args) if args.warc_manifest else run_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path, save_tasks=save_tasks)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
