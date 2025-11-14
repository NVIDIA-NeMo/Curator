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

import ray.data
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.math import MathContentExtractor
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.extract import DocumentExtractStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter


def build_pipeline(input_glob: str, output_dir: str) -> Pipeline:
    p = Pipeline(name="math_text_preprocess", description="Decode (binary) → type → html via lynx → text")

    p.add_stage(
        ParquetReader(file_paths=input_glob).with_(
            {
                "file_partitioning": {"resources": Resources(cpus=0.1)},
                "parquet_reader": {"resources": Resources(cpus=0.1)},
            }
        )
    )

    p.add_stage(
        DocumentExtractStage(extractor=MathContentExtractor(), add_filename_column=False).with_(
            resources=Resources(cpus=1)
        )
    )

    p.add_stage(JsonlWriter(path=output_dir).with_(resources=Resources(cpus=0.1)))

    return p


def report_extraction_stats(output_dir: str) -> None:
    """Optional: Report extraction statistics by reading output with Ray Data."""
    try:
        ds = ray.data.read_json(f"{output_dir}/*.jsonl")
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
    args = parser.parse_args()

    ray_client = RayClient()
    ray_client.start()

    pipeline = build_pipeline(args.input, args.output)
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
