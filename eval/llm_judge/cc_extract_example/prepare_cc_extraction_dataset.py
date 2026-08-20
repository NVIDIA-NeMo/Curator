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
Prepare a Common Crawl extraction-comparison dataset for this LLM-judge example.

Example:
    python eval/llm_judge/cc_extract_example/prepare_cc_extraction_dataset.py \
        --download-dir data/cc_warcs --output-path data/cc_extractions
"""

from __future__ import annotations

import argparse

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import DocumentDownloadExtractStage, DocumentExtractor
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader
from nemo_curator.stages.text.download.common_crawl.url_generation import MainCommonCrawlUrlGenerator
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator
from nemo_curator.stages.text.download.html_extractors import JusTextExtractor, TrafilaturaExtractor
from nemo_curator.stages.text.download.html_extractors.utils import get_stop_list_dict
from nemo_curator.stages.text.download.utils import decode_html, lang_detect
from nemo_curator.stages.text.io.writer import JsonlWriter

OUTPUT_FIELDS = [
    "url",
    "warc_id",
    "source_id",
    "language",
    "raw_text",
    "justext_text",
    "trafilatura_text",
]


def _extract_text(
    extractor: JusTextExtractor | TrafilaturaExtractor,
    html: str,
    stop_words: frozenset[str],
    language: str,
) -> str | None:
    """Run a Curator HTML extractor and normalize its paragraph output."""
    paragraphs = extractor.extract_text(html, stop_words, language)
    return "\n\n".join(paragraphs) if paragraphs else None


class JusTextTrafilaturaExtractor(DocumentExtractor):
    """
    Preserve decoded HTML and run jusText plus Trafilatura on each WARC record.

    ``raw_text`` is decoded raw HTML, not plain visible-page text;
    it intentionally gives the LLM judge the source that the two extractors processed.
    """

    def __init__(self) -> None:
        self.justext = JusTextExtractor()
        self.trafilatura = TrafilaturaExtractor()
        self.stop_lists = get_stop_list_dict()

    def extract(self, record: dict[str, object]) -> dict[str, object] | None:
        html = decode_html(record.get("content", b""))
        if html is None:
            return None

        language: str | None = None
        justext_text: str | None = None
        trafilatura_text: str | None = None
        try:
            language = lang_detect(html)
            stop_words = self.stop_lists.get(language)
            if stop_words is not None:
                justext_text = _extract_text(self.justext, html, stop_words, language)
                trafilatura_text = _extract_text(self.trafilatura, html, stop_words, language)
        except Exception:  # noqa: BLE001, S110
            # Keep the raw HTML row even when language detection or one extractor fails.
            pass

        return {
            "url": record["url"],
            "warc_id": record["warc_id"],
            "source_id": record["source_id"],
            "language": language,
            "raw_text": html,
            "justext_text": justext_text,
            "trafilatura_text": trafilatura_text,
        }

    def input_columns(self) -> list[str]:
        return ["url", "warc_id", "source_id", "content"]

    def output_columns(self) -> list[str]:
        return OUTPUT_FIELDS


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build Common Crawl download -> dual extraction -> JSONL writer."""
    url_generator = MainCommonCrawlUrlGenerator(
        start_snapshot_str=args.start_snapshot,
        end_snapshot_str=args.end_snapshot,
        limit=args.url_limit,
    )

    extraction_stage = DocumentDownloadExtractStage(
        url_generator=url_generator,
        downloader=CommonCrawlWARCDownloader(
            download_dir=args.download_dir,
            use_aws_to_download=args.use_aws_to_download,
            verbose=args.verbose,
        ),
        iterator=CommonCrawlWarcIterator(),
        extractor=JusTextTrafilaturaExtractor(),
        url_limit=args.url_limit,
        record_limit=args.record_limit,
        add_filename_column=False,
        # jusText uses lxml and benefits from Curator's worker recycling.
        extractor_max_calls_per_worker=args.extractor_max_calls_per_worker,
    )
    return Pipeline(
        name="common_crawl_extraction_comparison",
        description="Download Common Crawl WARC files and compare jusText with Trafilatura extraction.",
        stages=[extraction_stage, JsonlWriter(path=args.output_path, fields=OUTPUT_FIELDS)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--start-snapshot",
        default="2026-30",
        help="CC-MAIN snapshot in YYYY-WW format (default: 2026-30).",
    )
    parser.add_argument(
        "--end-snapshot",
        default="2026-30",
        help="CC-MAIN snapshot in YYYY-WW format (default: 2026-30).",
    )
    parser.add_argument("--download-dir", required=True, help="Local directory for downloaded WARC files.")
    parser.add_argument("--output-path", required=True, help="Directory for JSONL output partitions.")
    parser.add_argument("--url-limit", type=int, default=1, help="Maximum WARC files to download (default: 1).")
    parser.add_argument("--record-limit", type=int, default=100, help="Maximum records per WARC file (default: 100).")
    parser.add_argument("--extractor-max-calls-per-worker", type=int, default=2)
    parser.add_argument("--use-aws-to-download", action="store_true", help="Use s5cmd against Common Crawl S3.")
    parser.add_argument("--verbose", action="store_true", help="Show Common Crawl downloader output.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = RayClient()
    client.start()
    try:
        build_pipeline(args).run(executor=RayDataExecutor())
    finally:
        client.stop()


if __name__ == "__main__":
    main()
