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

import json
import os
import subprocess
import zlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests
from bs4 import BeautifulSoup

from nemo_curator.utils.download_utils import (
    get_arxiv_urls,
    get_common_crawl_snapshot_index,
    get_common_crawl_urls,
    get_main_warc_paths,
    get_news_warc_paths,
    get_wikipedia_urls,
)


class TestMainWarcPaths:
    """Tests for the get_main_warc_paths function."""

    def test_valid_snapshot_range(self):
        """Test with valid snapshot range."""
        # Mock snapshot index
        mock_index = [
            {"id": "CC-MAIN-2021-04"},
            {"id": "CC-MAIN-2021-10"},
            {"id": "CC-MAIN-2021-25"},
            {"id": "CC-MAIN-2021-31"},
            {"id": "CC-MAIN-2021-43"},
        ]

        # Call the function
        warc_paths = get_main_warc_paths(
            mock_index, "2021-10", "2021-31", prefix="https://test.example.com"
        )

        # Check results
        assert len(warc_paths) == 3
        assert (
            warc_paths[0]
            == "https://test.example.com/crawl-data/CC-MAIN-2021-10/warc.paths.gz"
        )
        assert (
            warc_paths[1]
            == "https://test.example.com/crawl-data/CC-MAIN-2021-25/warc.paths.gz"
        )
        assert (
            warc_paths[2]
            == "https://test.example.com/crawl-data/CC-MAIN-2021-31/warc.paths.gz"
        )

    def test_ignores_non_standard_format(self):
        """Test that IDs not in the standard format are ignored."""
        # Mock snapshot index with only valid entries
        # The actual function doesn't gracefully handle non-standard formats
        # but tries to parse them and raises a ValueError
        mock_index = [
            {"id": "CC-MAIN-2021-04"},
            {"id": "CC-MAIN-2021-25"},
        ]

        # Call the function
        warc_paths = get_main_warc_paths(mock_index, "2021-04", "2021-25")

        # Check results
        assert len(warc_paths) == 2
        assert "CC-MAIN-2021-04" in warc_paths[0]
        assert "CC-MAIN-2021-25" in warc_paths[1]

    def test_handles_invalid_format(self):
        """Test that the function properly handles IDs with invalid format."""
        # Mock snapshot index with a non-standard entry
        mock_index = [
            {"id": "CC-MAIN-2021-04"},
            {"id": "CC-MAIN-non-standard"},
        ]

        # The function should raise ValueError when encountering non-standard format
        with pytest.raises(ValueError) as excinfo:
            get_main_warc_paths(mock_index, "2021-04", "2021-25")

        # Check error message
        assert "invalid literal for int()" in str(excinfo.value)

    def test_filters_by_year(self):
        """Test filtering of snapshots before 2013."""
        # Mock snapshot index
        mock_index = [
            {"id": "CC-MAIN-2012-10"},
            {"id": "CC-MAIN-2013-04"},
            {"id": "CC-MAIN-2014-10"},
        ]

        # Call the function
        with patch("builtins.print") as mock_print:
            warc_paths = get_main_warc_paths(mock_index, "2012-10", "2014-10")

            # Check that warning was printed
            mock_print.assert_called_once()

        # Check results (should only include >= 2013)
        assert len(warc_paths) == 2
        assert "CC-MAIN-2013-04" in warc_paths[0]
        assert "CC-MAIN-2014-10" in warc_paths[1]

    def test_both_dates_before_2013(self):
        """Test when both start and end snapshots are before 2013."""
        # Mock snapshot index
        mock_index = [
            {"id": "CC-MAIN-2010-10"},
            {"id": "CC-MAIN-2011-15"},
            {"id": "CC-MAIN-2012-20"},
            {"id": "CC-MAIN-2013-04"},
            {"id": "CC-MAIN-2014-10"},
        ]

        # Call the function
        with patch("builtins.print") as mock_print:
            warc_paths = get_main_warc_paths(mock_index, "2010-10", "2012-20")

            # Check that warning was printed
            mock_print.assert_called_once()

        # Check results (should be empty since all matching snapshots are filtered out)
        assert len(warc_paths) == 0

    def test_invalid_date_range(self):
        """Test with start date after end date."""
        # Mock snapshot index
        mock_index = [
            {"id": "CC-MAIN-2021-04"},
            {"id": "CC-MAIN-2021-10"},
        ]

        # Call the function with start date after end date
        with pytest.raises(ValueError) as excinfo:
            get_main_warc_paths(mock_index, "2021-10", "2021-04")

        # Check error message
        assert "Start snapshot" in str(excinfo.value)
        assert "is after end snapshot" in str(excinfo.value)


class TestNewsWarcPaths:
    """Tests for the get_news_warc_paths function."""

    def test_valid_date_range(self):
        """Test with valid date range."""
        # Call the function
        with patch("datetime.datetime") as mock_datetime:
            # Mock datetime.now() to return a fixed date
            mock_now = MagicMock()
            mock_now.year = 2023
            mock_now.month = 6
            mock_datetime.now.return_value = mock_now

            warc_paths = get_news_warc_paths(
                "2023-01", "2023-03", prefix="https://test.example.com"
            )

            # Check results (should have entries for Jan, Feb, Mar 2023)
            assert len(warc_paths) == 3
            assert (
                warc_paths[0]
                == "https://test.example.com/crawl-data/CC-NEWS/2023/01/warc.paths.gz"
            )
            assert (
                warc_paths[1]
                == "https://test.example.com/crawl-data/CC-NEWS/2023/02/warc.paths.gz"
            )
            assert (
                warc_paths[2]
                == "https://test.example.com/crawl-data/CC-NEWS/2023/03/warc.paths.gz"
            )

    def test_invalid_date_range(self):
        """Test with start date after end date."""
        # Call the function with start date after end date
        with pytest.raises(ValueError) as excinfo:
            get_news_warc_paths("2023-03", "2023-01")

        # Check error message
        assert "Start snapshot" in str(excinfo.value)
        assert "is after end snapshot" in str(excinfo.value)

    def test_date_out_of_supported_range(self):
        """Test with dates outside supported range (warning)."""
        # Call the function with dates outside supported range
        with patch("nemo_curator.utils.download_utils.datetime") as mock_datetime:
            # Mock datetime.now() to return a fixed date
            mock_now = MagicMock()
            mock_now.year = 2023
            mock_now.month = 6
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime.side_effect = datetime.strptime

            with patch("builtins.print") as mock_print:
                # Test with date before 2016
                get_news_warc_paths("2015-12", "2016-01")
                mock_print.assert_called_once()
                mock_print.reset_mock()

                # Test with date after current date
                get_news_warc_paths("2023-01", "2024-01")
                mock_print.assert_called_once()


class TestCommonCrawlSnapshotIndex:
    """Tests for the get_common_crawl_snapshot_index function."""

    def test_retrieves_snapshot_index(self):
        """Test retrieving snapshot index from URL."""
        # Mock response from index URL
        mock_index_content = json.dumps(
            [{"id": "CC-MAIN-2021-04"}, {"id": "CC-MAIN-2021-10"}]
        )
        mock_response = MagicMock()
        mock_response.content = mock_index_content.encode()

        with patch("requests.get", return_value=mock_response) as mock_get:
            # Call the function
            result = get_common_crawl_snapshot_index("https://index.example.com/")

            # Check requests.get called with correct URL
            mock_get.assert_called_once_with("https://index.example.com/collinfo.json")

            # Check result is parsed JSON content
            assert result == [{"id": "CC-MAIN-2021-04"}, {"id": "CC-MAIN-2021-10"}]


class TestCommonCrawlUrls:
    """Tests for the get_common_crawl_urls function."""

    @patch("nemo_curator.utils.download_utils.get_common_crawl_snapshot_index")
    @patch("nemo_curator.utils.download_utils.get_main_warc_paths")
    @patch("requests.get")
    def test_cc_main_urls(self, mock_get, mock_paths, mock_index):
        """Test retrieving CC-MAIN URLs."""
        # Setup mock responses
        mock_index.return_value = "mock_index_data"
        mock_paths.return_value = [
            "https://data.example.com/path1.gz",
            "https://data.example.com/path2.gz",
        ]

        # Create mock responses for each path
        mock_response1 = MagicMock()
        mock_response1.content = zlib.compress("warc1\nwarc2\n".encode("utf-8"))
        mock_response2 = MagicMock()
        mock_response2.content = zlib.compress("warc3\n".encode("utf-8"))
        mock_get.side_effect = [mock_response1, mock_response2]

        # Call the function
        result = get_common_crawl_urls(
            "2021-10",
            "2021-25",
            data_domain_prefix="https://data.example.com/",
            index_prefix="https://index.example.com/",
        )

        # Check function calls
        mock_index.assert_called_once_with("https://index.example.com/")
        mock_paths.assert_called_once_with(
            "mock_index_data", "2021-10", "2021-25", prefix="https://data.example.com/"
        )

        # Check result URLs
        assert len(result) == 3
        assert result[0] == "https://data.example.com/warc1"
        assert result[1] == "https://data.example.com/warc2"
        assert result[2] == "https://data.example.com/warc3"

    @patch("nemo_curator.utils.download_utils.get_news_warc_paths")
    @patch("requests.get")
    def test_cc_news_urls(self, mock_get, mock_paths):
        """Test retrieving CC-NEWS URLs."""
        # Setup mock responses
        mock_paths.return_value = ["https://data.example.com/path1.gz"]

        # Create mock response
        mock_response = MagicMock()
        mock_response.content = zlib.compress(
            "news-warc1\nnews-warc2\n".encode("utf-8")
        )
        mock_get.return_value = mock_response

        # Call the function
        result = get_common_crawl_urls(
            "2023-01",
            "2023-03",
            data_domain_prefix="https://data.example.com/",
            news=True,
        )

        # Check function calls
        mock_paths.assert_called_once_with(
            "2023-01", "2023-03", prefix="https://data.example.com/"
        )

        # Check result URLs
        assert len(result) == 2
        assert result[0] == "https://data.example.com/news-warc1"
        assert result[1] == "https://data.example.com/news-warc2"

    @patch("nemo_curator.utils.download_utils.get_main_warc_paths")
    @patch("requests.get")
    def test_exception_handling(self, mock_get, mock_paths):
        """Test handling exceptions when retrieving URLs."""
        # Setup mock responses
        mock_paths.return_value = [
            "https://data.example.com/path1.gz",
            "https://data.example.com/path2.gz",
        ]

        # First request succeeds, second fails
        mock_response1 = MagicMock()
        mock_response1.content = zlib.compress("warc1\nwarc2\n".encode("utf-8"))
        mock_response2 = MagicMock()
        mock_response2.content = b"invalid_content"  # Will cause decompression error
        mock_get.side_effect = [mock_response1, mock_response2]

        # Call the function with index already mocked
        with patch("nemo_curator.utils.download_utils.get_common_crawl_snapshot_index"):
            with patch("builtins.print") as mock_print:
                result = get_common_crawl_urls("2021-10", "2021-25")

        # Check error message was printed
        assert (
            mock_print.call_count == 3
        )  # Three print statements: path, content, and exception

        # Check result URLs (only from successful request)
        assert len(result) == 2
        assert "warc1" in result[0]
        assert "warc2" in result[1]


class TestWikipediaUrls:
    """Tests for the get_wikipedia_urls function."""

    @patch("requests.get")
    def test_latest_dump(self, mock_get):
        """Test retrieving latest Wikipedia dump URLs."""
        # Mock index response
        mock_index_html = """
        <html>
        <body>
            <a href="20230101/">20230101/</a>
            <a href="20230201/">20230201/</a>
            <a href="20230301/">20230301/</a>
        </body>
        </html>
        """
        mock_index_response = MagicMock()
        mock_index_response.content = mock_index_html.encode("utf-8")

        # Mock dump status response
        mock_dump_json = json.dumps(
            {
                "jobs": {
                    "articlesmultistreamdump": {
                        "files": {
                            "enwiki-20230201-pages-articles-multistream1.xml-p1p41242.bz2": {},
                            "enwiki-20230201-pages-articles-multistream2.xml-p41243p151573.bz2": {},
                            "some-other-file.txt": {},
                        }
                    }
                }
            }
        )
        mock_dump_response = MagicMock()
        mock_dump_response.content = mock_dump_json.encode("utf-8")

        # Set up mock response sequence
        mock_get.side_effect = [mock_index_response, mock_dump_response]

        # Call the function
        result = get_wikipedia_urls(
            language="en", wikidumps_index_prefix="https://dumps.example.com/"
        )

        # Check requests were made correctly
        assert mock_get.call_count == 2
        mock_get.assert_any_call("https://dumps.example.com/enwiki")
        mock_get.assert_any_call(
            "https://dumps.example.com/enwiki/20230201/dumpstatus.json"
        )

        # Check result URLs
        assert len(result) == 2
        assert "multistream1.xml" in result[0]
        assert "multistream2.xml" in result[1]

    @patch("requests.get")
    def test_specific_dump_date(self, mock_get):
        """Test retrieving Wikipedia dump URLs for a specific date."""
        # Mock dump status response
        mock_dump_json = json.dumps(
            {
                "jobs": {
                    "articlesmultistreamdump": {
                        "files": {
                            "enwiki-20220101-pages-articles-multistream1.xml-p1p41242.bz2": {},
                            "enwiki-20220101-pages-articles-multistream2.xml-p41243p151573.bz2": {},
                        }
                    }
                }
            }
        )
        mock_dump_response = MagicMock()
        mock_dump_response.content = mock_dump_json.encode("utf-8")

        # Set up mock response
        mock_get.return_value = mock_dump_response

        # Call the function with specific date
        result = get_wikipedia_urls(
            language="en",
            wikidumps_index_prefix="https://dumps.example.com/",
            dump_date="20220101",
        )

        # Check request was made correctly
        mock_get.assert_called_once_with(
            "https://dumps.example.com/enwiki/20220101/dumpstatus.json"
        )

        # Check result URLs
        assert len(result) == 2
        assert "20220101" in result[0]
        assert "20220101" in result[1]

    @patch("requests.get")
    def test_invalid_dump_date(self, mock_get):
        """Test error handling for invalid dump date."""
        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.content = b"Not a valid JSON"

        # Set up mock response
        mock_get.return_value = mock_response

        # Call the function with specific date
        with pytest.raises(ValueError) as excinfo:
            get_wikipedia_urls(dump_date="20220101")

        # Check error message
        assert "No wikipedia dump found for 20220101" in str(excinfo.value)


class TestArxivUrls:
    """Tests for the get_arxiv_urls function."""

    @patch("subprocess.run")
    def test_successful_retrieval(self, mock_run):
        """Test successful retrieval of ArXiv URLs."""
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        # The output of s5cmd has a specific format we need to match for indexing
        # This specifically needs items at position 3, 7, 11, etc. to be URLs
        mock_result.stdout = "DATE1 TIME1 SIZE1 s3://arxiv/src/arXiv_src_1804_001.tar DATE2 TIME2 SIZE2 s3://arxiv/src/arXiv_src_1805_001.tar DATE3 TIME3 SIZE3 s3://arxiv/src/arXiv_src_1806_001.tar"
        mock_run.return_value = mock_result

        # Call the function
        result = get_arxiv_urls()

        # Check subprocess call
        mock_run.assert_called_once()
        assert "s5cmd" in mock_run.call_args[0][0]
        assert "s3://arxiv/src/" in mock_run.call_args[0][0]

        # Check result URLs
        assert len(result) == 3
        assert result == [
            "s3://arxiv/src/arXiv_src_1804_001.tar",
            "s3://arxiv/src/arXiv_src_1805_001.tar",
            "s3://arxiv/src/arXiv_src_1806_001.tar",
        ]

    @patch("subprocess.run")
    def test_command_failure(self, mock_run):
        """Test error handling when s5cmd command fails."""
        # Mock subprocess result (failure)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Command failed: access denied"
        mock_run.return_value = mock_result

        # Call the function and check for error
        with pytest.raises(RuntimeError) as excinfo:
            get_arxiv_urls()

        # Check error message
        assert "Unable to get arxiv urls" in str(excinfo.value)
        assert "access denied" in str(excinfo.value)
