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
from dataclasses import dataclass

import ray.data
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.math.download.extract import MathContentExtractor
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.extract import DocumentExtractStage
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter


@dataclass
class TextPreprocessConfig:
    """Configuration for text preprocessing."""

    input_glob: str
    output_dir: str
    fetch_cc: bool = False
    warc_filename_col: str = "warc_filename"
    warc_record_offset_col: str = "warc_record_offset"
    warc_record_length_col: str = "warc_record_length"


def build_pipeline(config: TextPreprocessConfig) -> Pipeline:
    p = Pipeline(name="math_text_preprocess", description="Decode (binary) → type → html via lynx → text")

    p.add_stage(
        ParquetReader(file_paths=config.input_glob).with_(
            {
                "file_partitioning": {"resources": Resources(cpus=1.0)},
                "parquet_reader": {"resources": Resources(cpus=1.0)},
            }
        )
    )

    if config.fetch_cc:
        logger.info("Adding CommonCrawlWARCReader stage to fetch content from S3.")
        p.add_stage(
            CommonCrawlWARCReader(
                warc_filename_col=config.warc_filename_col,
                warc_record_offset_col=config.warc_record_offset_col,
                warc_record_length_col=config.warc_record_length_col,
            ).with_(resources=Resources(cpus=0.5))  # Lightweight network op
        )

    p.add_stage(
        DocumentExtractStage(
            extractor=MathContentExtractor(
                binary_column="binary_content", url_column="url", mime_type_column="content_mime_type"
            ),
            add_filename_column=False,
        ).with_(resources=Resources(cpus=1.0))
    )

    p.add_stage(JsonlWriter(path=config.output_dir).with_(resources=Resources(cpus=1.0)))

    return p


def report_extraction_stats(output_dir: str) -> None:
    """Optional: Report extraction statistics by reading output with Ray Data."""
    try:
        from nemo_curator.utils.file_utils import get_all_file_paths_under

        jsonl_files = get_all_file_paths_under(output_dir, keep_extensions=[".jsonl"])
        if not jsonl_files:
            logger.debug(f"No JSONL files found in {output_dir}")
            return

        ds = ray.data.read_json(jsonl_files)
        total = ds.count()
        html_docs = ds.filter(lambda row: row.get("type") == "html").count()
        html_failed = ds.filter(
            lambda row: row.get("type") == "html" and (not row.get("text") or row.get("text").strip() == "")
        ).count()

        logger.info(
            f"Extraction stats: {total} total documents, {html_docs} HTML, {html_failed} HTML extraction failures"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not compute stats (optional): {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run math text preprocessing on Parquet files")
    parser.add_argument("--input", required=True, help="Glob or directory for Parquet input files")
    parser.add_argument("--output", required=True, help="Output directory for JSONL results")
    parser.add_argument("--report-stats", action="store_true", help="Report extraction statistics after processing")
    parser.add_argument(
        "--fetch-cc",
        action="store_true",
        help="Fetch raw content from Common Crawl S3 using WARC metadata (requires 'warc_filename', 'warc_record_offset', 'warc_record_length' columns).",
    )
    parser.add_argument(
        "--warc-filename-col",
        default="warc_filename",
        help="Column name for WARC filename (default: 'warc_filename')",
    )
    parser.add_argument(
        "--offset-col",
        default="warc_record_offset",
        help="Column name for WARC record offset (default: 'warc_record_offset')",
    )
    parser.add_argument(
        "--length-col",
        default="warc_record_length",
        help="Column name for WARC record length (default: 'warc_record_length')",
    )

    args = parser.parse_args()

    ray_client = RayClient()
    ray_client.start()

    config = TextPreprocessConfig(
        input_glob=args.input,
        output_dir=args.output,
        fetch_cc=args.fetch_cc,
        warc_filename_col=args.warc_filename_col,
        warc_record_offset_col=args.offset_col,
        warc_record_length_col=args.length_col,
    )
    pipeline = build_pipeline(config)
    logger.info(pipeline.describe())

    executor = XennaExecutor()
    pipeline.run(executor)

    logger.info("Pipeline completed successfully.")

    # Optional: Report extraction statistics
    if args.report_stats:
        report_extraction_stats(args.output)

    ray_client.stop()


if __name__ == "__main__":
    main()
