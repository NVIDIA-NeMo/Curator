# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from .arxiv import ArxivDownloader, ArxivExtractor, ArxivIterator, download_arxiv
from .commoncrawl import (
    CommonCrawlWARCDownloader,
    CommonCrawlWARCDownloaderExtractOnly,
    CommonCrawlWARCExtractor,
    CommonCrawlWARCIterator,
    JusTextExtractor,
    ResiliparseExtractor,
    download_common_crawl,
)
from .doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    batch_download,
    download_and_extract,
    import_downloader,
    import_extractor,
    import_iterator,
)
from .wikipedia import (
    WikipediaDownloader,
    WikipediaExtractor,
    WikipediaIterator,
    download_wikipedia,
)

__all__ = [
    "DocumentDownloader",
    "DocumentIterator",
    "DocumentExtractor",
    "download_and_extract",
    "import_downloader",
    "import_extractor",
    "import_iterator",
    "download_common_crawl",
    "CommonCrawlWARCDownloader",
    "CommonCrawlWARCExtractor",
    "CommonCrawlWARCIterator",
    "CommonCrawlWARCDownloaderExtractOnly",
    "JusTextExtractor",
    "ResiliparseExtractor",
    "download_wikipedia",
    "WikipediaDownloader",
    "WikipediaIterator",
    "WikipediaExtractor",
    "batch_download",
    "download_arxiv",
    "ArxivDownloader",
    "ArxivIterator",
    "ArxivExtractor",
]
