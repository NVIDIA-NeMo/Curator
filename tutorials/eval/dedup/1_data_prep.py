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

"""
Download and extract a small Common Crawl sample, for exercising the
fuzzy-dedup-eval example end to end.

Example:
    python tutorials/eval/dedup/1_data_prep.py \
        --download-dir output/dedup_eval/cc_warcs \
        --output-path output/dedup_eval/raw_corpus
"""

from __future__ import annotations

import argparse

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.download.common_crawl.stage import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-snapshot", default="2025-30", help="CC-MAIN snapshot in YYYY-WW format.")
    parser.add_argument("--end-snapshot", default="2025-30", help="CC-MAIN snapshot in YYYY-WW format.")
    parser.add_argument("--download-dir", required=True, help="Local directory for downloaded WARC files.")
    parser.add_argument("--output-path", required=True, help="Directory for the extracted JSONL corpus.")
    parser.add_argument("--url-limit", type=int, default=5, help="Maximum WARC files to download.")
    parser.add_argument("--record-limit", type=int, default=2000, help="Maximum records to extract per WARC file.")
    parser.add_argument("--use-aws-to-download", action="store_true", help="Use s5cmd against Common Crawl S3.")
    parser.add_argument("--verbose", action="store_true", help="Show Common Crawl downloader output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    ray_client = RayClient()
    ray_client.start()
    try:
        stage = CommonCrawlDownloadExtractStage(
            start_snapshot=args.start_snapshot,
            end_snapshot=args.end_snapshot,
            download_dir=args.download_dir,
            crawl_type="main",
            use_aws_to_download=args.use_aws_to_download,
            verbose=args.verbose,
            url_limit=args.url_limit,
            record_limit=args.record_limit,
        )
        pipeline = Pipeline(
            "dedup_eval_data_prep",
            stages=[stage, JsonlWriter(path=args.output_path, write_kwargs={"force_ascii": False})],
        )
        pipeline.run()
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
