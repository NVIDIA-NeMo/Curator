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

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cudf
import ray
from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_all_file_paths_under, get_fs


@dataclass
class CCIndexLookupConfig:
    """Configuration for CC Index lookup."""

    input_path: str
    output_path: str
    cc_index_path: str
    crawls: list[str] | None = None
    url_col: str = "url"
    blocksize: str = "512MiB"


CC_INDEX_COLS = [
    "url",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "content_mime_type",
    "http_status",
]


class CCIndexLookupStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that joins CC Index files against broadcast query URLs using cuDF:
        - cudf.read_parquet() for GPU-native reading
        - cudf.merge() for GPU-accelerated join
        - Query URLs broadcast via Ray object store
    """

    name = "CCIndexLookup"
    resources = Resources(gpus=1.0)

    def __init__(
        self,
        query_urls_ref: ray.ObjectRef,
        output_path: str,
        url_col: str = "url",
        write_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.query_urls_ref = query_urls_ref
        self.output_path = output_path
        self.url_col = url_col
        self.write_kwargs = write_kwargs or {}
        self._query_df: cudf.DataFrame | None = None

        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options"))
        self.output_fs.makedirs(output_path, exist_ok=True)

    def setup(self, _worker_metadata: dict | None = None) -> None:
        """Load broadcast query URLs from Ray object store."""
        if self._query_df is None:
            self._query_df = ray.get(self.query_urls_ref)
            logger.info(f"Loaded {len(self._query_df):,} query URLs from broadcast")

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process CC Index files and join against query URLs."""
        if self._query_df is None:
            msg = "Query URLs not loaded. Call setup() first."
            raise RuntimeError(msg)

        output_files = []
        total_input = 0
        total_matched = 0

        for cc_index_file in task.data:
            cc_df = cudf.read_parquet(cc_index_file, columns=CC_INDEX_COLS)
            total_input += len(cc_df)

            matched = cc_df.merge(self._query_df, on="url", how="inner")
            total_matched += len(matched)

            if len(matched) == 0:
                continue

            # Convert category columns to strings (cudf parquet writer limitation)
            for col in matched.columns:
                if matched[col].dtype.name == "category":
                    matched[col] = matched[col].astype(str)

            # Write output
            output_file = self.output_fs.sep.join([self.output_path, Path(cc_index_file).stem + "_enriched.parquet"])
            matched.to_parquet(output_file, **self.write_kwargs)
            output_files.append(output_file)

        logger.debug(f"Processed {len(task.data)} files: {total_input:,} -> {total_matched:,} rows")

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=output_files,
            _metadata={
                "input_rows": total_input,
                "matched_rows": total_matched,
            },
        )


def collect_unique_urls(input_path: str, url_col: str = "url") -> cudf.DataFrame:
    """Collect unique URLs from input dataset using cuDF."""
    logger.info(f"Collecting unique URLs from: {input_path}")

    input_files = get_all_file_paths_under(input_path, keep_extensions=[".parquet"])
    if not input_files:
        msg = f"No parquet files found at {input_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Found {len(input_files)} input files")

    dfs = [cudf.read_parquet(f, columns=[url_col]) for f in input_files]
    combined = cudf.concat(dfs, ignore_index=True)
    unique_urls = combined.drop_duplicates(subset=[url_col])

    if url_col != "url":
        unique_urls = unique_urls.rename(columns={url_col: "url"})

    logger.info(f"Collected {len(unique_urls):,} unique URLs")
    return unique_urls


def get_available_crawls(cc_index_base: str) -> list[str]:
    """Auto-detect available crawl directories."""
    if not os.path.exists(cc_index_base):
        return []
    crawl_dirs = [d for d in os.listdir(cc_index_base) if d.startswith("crawl=")]
    return sorted([d.split("=")[1] for d in crawl_dirs])


def get_cc_index_files(cc_index_base: str, crawls: list[str] | None = None) -> list[str]:
    """Collect all CC Index parquet files for specified crawls (or all if None)."""
    if crawls is None:
        crawls = get_available_crawls(cc_index_base)
        if not crawls:
            msg = f"No crawl directories found at {cc_index_base}"
            raise FileNotFoundError(msg)
        logger.info(f"Auto-detected {len(crawls)} crawls: {crawls}")

    all_files = []
    for crawl in crawls:
        crawl_path = os.path.join(cc_index_base, f"crawl={crawl}", "subset=warc")
        if os.path.exists(crawl_path):
            files = get_all_file_paths_under(crawl_path, keep_extensions=[".parquet"])
            all_files.extend(files)
            logger.info(f"Found {len(files)} CC Index files for {crawl}")
        else:
            logger.warning(f"CC Index path not found: {crawl_path}")

    if not all_files:
        msg = f"No CC Index files found for crawls {crawls}"
        raise FileNotFoundError(msg)

    return all_files


def run_cc_index_lookup(config: CCIndexLookupConfig) -> None:
    ray_client = RayClient()
    ray_client.start()

    try:
        # Step 1: Collect query URLs and broadcast
        query_urls = collect_unique_urls(config.input_path, config.url_col)
        query_urls_ref = ray.put(query_urls)
        logger.info("Query URLs broadcast to Ray object store")

        # Step 2: Collect CC Index files from all crawls
        cc_files = get_cc_index_files(config.cc_index_path, config.crawls)
        logger.info(f"Total CC Index files: {len(cc_files)}")

        # Step 3: Build pipeline
        pipeline = Pipeline(
            name="cc_index_lookup",
            stages=[
                FilePartitioningStage(
                    file_paths=cc_files,
                    blocksize=config.blocksize,
                ),
                CCIndexLookupStage(
                    query_urls_ref=query_urls_ref,
                    output_path=config.output_path,
                    url_col=config.url_col,
                ),
            ],
        )

        logger.info(pipeline.describe())

        # Step 4: Run pipeline
        result_tasks = pipeline.run()

        # Summarize
        total_input = sum(t._metadata.get("input_rows", 0) for t in result_tasks)
        total_matched = sum(t._metadata.get("matched_rows", 0) for t in result_tasks)

        logger.info("=" * 60)
        logger.info(f"Query URLs:    {len(query_urls):,}")
        logger.info(f"CC Index rows: {total_input:,}")
        logger.info(f"Matched rows:  {total_matched:,}")
        logger.info(f"Output:        {config.output_path}")
        logger.info("=" * 60)

    finally:
        ray_client.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich dataset with WARC metadata using Curator's cuDF pattern.",
    )
    parser.add_argument("--input", required=True, help="Input directory with parquet files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--cc-index-path",
        required=True,
        help="CC Index path (<path>/crawl=CC-MAIN-YYYY-WW/subset=warc/)",
    )
    parser.add_argument(
        "--crawls",
        nargs="+",
        default=None,
        help="Crawl IDs (default: auto-detect all available crawls)",
    )
    parser.add_argument("--url-col", default="url", help="URL column name")
    parser.add_argument("--blocksize", default="512MiB", help="File block size")

    args = parser.parse_args()

    config = CCIndexLookupConfig(
        input_path=args.input,
        output_path=args.output,
        cc_index_path=args.cc_index_path,
        crawls=args.crawls,
        url_col=args.url_col,
        blocksize=args.blocksize,
    )
    run_cc_index_lookup(config)

    logger.info(f"Next: python 2_text_preprocess.py --input {args.output}/*.parquet --fetch-cc")


if __name__ == "__main__":
    main()
