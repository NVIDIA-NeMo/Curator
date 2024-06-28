from pathlib import Path

import pytest

from nemo_curator.download import download_and_extract
from nemo_curator.download.commoncrawl import (
    CommonCrawlWARCDownloader,
    CommonCrawlWARCExtractor,
    CommonCrawlWARCIterator,
    get_common_crawl_urls,
)


class TestDownload:
    def test_imports(self):
        from nemo_curator.download import (
            download_arxiv,
            download_common_crawl,
            download_wikipedia,
        )

        assert True

    def test_common_crawl_urls(self):
        start_snapshot = "2021-04"
        end_snapshot = "2021-10"
        urls = get_common_crawl_urls(start_snapshot, end_snapshot)

        assert (
            urls[0]
            == "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/segments/1614178347293.1/warc/CC-MAIN-20210224165708-20210224195708-00000.warc.gz"
        )
        assert (
            urls[-1]
            == "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-04/segments/1610704847953.98/warc/CC-MAIN-20210128134124-20210128164124-00799.warc.gz"
        )
        assert len(urls) == 143840

    def test_incorrect_snapshot_order(self):
        with pytest.raises(ValueError):
            end_snapshot = "2021-04"
            start_snapshot = "2021-10"
            urls = get_common_crawl_urls(start_snapshot, end_snapshot)

    def test_common_crawl_news_urls(self):
        start_snapshot = "2021-04"
        end_snapshot = "2021-10"
        urls = get_common_crawl_urls(start_snapshot, end_snapshot, news=True)

        assert (
            urls[0]
            == "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/04/CC-NEWS-20210401004522-01022.warc.gz"
        )
        assert (
            urls[-1]
            == "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/10/CC-NEWS-20211031225258-00089.warc.gz"
        )
        assert len(urls) == 3838

    def test_incorrect_snapshot_order_news(self):
        with pytest.raises(ValueError):
            end_snapshot = "2021-04"
            start_snapshot = "2021-10"
            urls = get_common_crawl_urls(start_snapshot, end_snapshot, news=True)

    def test_uneven_common_crawl_range(self):
        start_snapshot = "2021-03"
        end_snapshot = "2021-11"
        urls = get_common_crawl_urls(start_snapshot, end_snapshot)

        assert (
            urls[0]
            == "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/segments/1614178347293.1/warc/CC-MAIN-20210224165708-20210224195708-00000.warc.gz"
        )
        assert (
            urls[-1]
            == "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-04/segments/1610704847953.98/warc/CC-MAIN-20210128134124-20210128164124-00799.warc.gz"
        )
        assert len(urls) == 143840

    def test_no_urls(self):
        with pytest.raises(ValueError):
            output_format = {
                "text": str,
            }
            download_and_extract(
                [],
                [],
                CommonCrawlWARCDownloader(download_dir="."),
                CommonCrawlWARCIterator(),
                CommonCrawlWARCExtractor(),
                output_format,
            )

    def test_url_path_mismatch(self):
        with pytest.raises(ValueError):
            output_format = {
                "text": str,
            }
            download_and_extract(
                ["one", "two", "three"],
                ["one"],
                CommonCrawlWARCDownloader(download_dir="."),
                CommonCrawlWARCIterator(),
                CommonCrawlWARCExtractor(),
                output_format,
            )
