from pathlib import Path

import pytest

from nemo_curator.download import ResiliparseExtractor, download_and_extract
from nemo_curator.download.commoncrawl import (
    CommonCrawlWARCDownloader,
    CommonCrawlWARCExtractor,
    CommonCrawlWARCIterator,
    get_common_crawl_urls,
    get_stop_list_dict,
)


class TestDownload:
    def test_imports(self):
        from nemo_curator.download import (
            JusTextExtractor,
            ResiliparseExtractor,
            download_arxiv,
            download_common_crawl,
            download_wikipedia,
        )

        assert True

    def test_resiliparse_extract_text(self):
        # Modified from https://github.com/chatnoir-eu/chatnoir-resiliparse/blob/abdf1966fb3cefe3e0790e510ab5cb1446f99a79/tests/resiliparse/extract/test_html2text.py
        html = """<!doctype html>
            <head>
                <title>My Title</title>
                <meta charset="utf-8">
                <style>* { margin: 0; }</style>
            </head>
            <body>
                <section id="wrapper">
                    <nav>
                        <ul>
                            <li>Nav 1</li>
                            <li>
                                <p>Nav 2</p>
                                <ul>
                                    <li><p>Nav 3</p></li>
                                </ul>
                            </li>
                        </ul>
                    </nav>
                    <main>
                        This is a sample paragraph. In it we write words.
                        These are stopwords: because did than has near we almost while what still.
                        <a href="#foo" hidden>bar</a>

                        <p>
                        This paragraph doesn't have many stopwords. Remove it.
                        <br>Let's keep this paragraph: either came does last new took taken making became from.
                        </p>

                        <button aria-hidden="true">Click here</button>
                        <input type="hidden" value="foo">
                        <input type="text" value="Some text" placeholder="Insert text">
                        <input type="text" placeholder="Insert text">
                        <img src="" alt="Some image">
                        <object data="" class="some-class hidden">Cannot display object</object>
                    </main>
                    <script language="vbscript" type="text/vbscript">MsgBox("Hello World!")</script>
                    <noscript>Sorry, your browser doesn't support VB Script!</noscript>
                    <div><div><div><footer id="global-footer">
                        Copyright (C) 2021 Foo Bar
                    </footer></div></div></div>
                </section>
            </body>
        </html>"""

        algorithm = ResiliparseExtractor()
        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(html, stop_words["ENGLISH"])

        expected = [
            "This is a sample paragraph. In it we write words. These are stopwords: because did than has near we almost while what still.",
            "Let's keep this paragraph: either came does last new took taken making became from.",
        ]

        assert result == expected

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

    @pytest.mark.skip(
        reason="Skipping until we figure out how to get this to a non flaky state"
    )
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
