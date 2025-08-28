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

from typing import Literal

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.html_extractors import HTMLExtractorAlgorithm

from .download import CommonCrawlWARCDownloader
from .extract import CommonCrawlHTMLExtractor
from .url_generation import MainCommonCrawlUrlGenerator, NewsCommonCrawlUrlGenerator
from .warc_iterator import CommonCrawlWarcIterator


class CommonCrawlDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite stage for downloading and processing Common Crawl data.

    This pipeline:
    1. Generates WARC URLs (either from main or news crawls)
    2. Downloads WARC files
    3. Extracts content from WARC files
    4. Extracts text from HTML content
    """

    def __init__(  # noqa: PLR0913
        self,
        start_snapshot: str,
        end_snapshot: str,
        download_dir: str,
        crawl_type: Literal["main", "news"] = "main",
        html_extraction: HTMLExtractorAlgorithm | str | None = None,
        html_extraction_kwargs: dict | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
        use_aws_to_download: bool = False,
        verbose: bool = False,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ):
        self.crawl_type = crawl_type
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot

        if crawl_type == "main":
            self.url_generator = MainCommonCrawlUrlGenerator(
                start_snapshot_str=start_snapshot, end_snapshot_str=end_snapshot, limit=url_limit
            )
        else:
            self.url_generator = NewsCommonCrawlUrlGenerator(
                start_snapshot_str=start_snapshot, end_snapshot_str=end_snapshot, limit=url_limit
            )

        self.downloader = CommonCrawlWARCDownloader(
            download_dir=download_dir, use_aws_to_download=use_aws_to_download, verbose=verbose
        )
        self.iterator = CommonCrawlWarcIterator()
        self.extractor = CommonCrawlHTMLExtractor(
            algorithm=html_extraction,
            algorithm_kwargs=html_extraction_kwargs,
            stop_lists=stop_lists,
        )
        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self._name = f"common_crawl_{self.crawl_type}_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return f"Common Crawl {self.crawl_type} pipeline: {self.start_snapshot} to {self.end_snapshot}"
