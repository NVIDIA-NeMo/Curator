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
from unittest.mock import Mock, patch

import pytest
import requests
from bs4 import BeautifulSoup

from nemo_curator.stages.text.download.wikipedia.url_generation import WikipediaUrlGenerator


class TestWikipediaUrlGenerator:
    """Test suite for WikipediaUrlGenerator."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        generator = WikipediaUrlGenerator()
        assert generator.language == "en"
        assert generator.dump_date is None
        assert generator.wikidumps_index_prefix == "https://dumps.wikimedia.org"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        generator = WikipediaUrlGenerator(
            language="es", dump_date="20230101", wikidumps_index_prefix="https://custom.dumps.org"
        )
        assert generator.language == "es"
        assert generator.dump_date == "20230101"
        assert generator.wikidumps_index_prefix == "https://custom.dumps.org"

    @patch("requests.get")
    def test_get_latest_dump_date_success(self, mock_get: Mock):
        """Test successful retrieval of latest dump date."""
        # Mock HTML response with dump directories
        mock_html = """
        <html>
        <body>
        <a href="../">../</a>
        <a href="20230401/">20230401/</a>
        <a href="20230420/">20230420/</a>
        <a href="20230501/">20230501/</a>
        <a href="20230520/">20230520/</a>
        <a href="latest/">latest/</a>
        </body>
        </html>
        """

        # Mock dumpstatus.json response
        mock_dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "enwiki-20230501-pages-articles-multistream1.xml.bz2": {},
                        "enwiki-20230501-pages-articles-multistream2.xml.bz2": {},
                    },
                }
            }
        }

        def mock_get_side_effect(url: str, **kwargs) -> Mock:  # noqa: ARG001
            response = Mock()
            if url == "https://dumps.wikimedia.org/enwiki":
                response.content = mock_html.encode("utf-8")
            elif url == "https://dumps.wikimedia.org/enwiki/20230520/dumpstatus.json":
                # Mock 20230520 as not finished, so it tries the next one
                response.content = json.dumps({"jobs": {"articlesmultistreamdump": {"status": "running"}}}).encode(
                    "utf-8"
                )
            elif url == "https://dumps.wikimedia.org/enwiki/20230501/dumpstatus.json":
                response.content = json.dumps(mock_dump_data).encode("utf-8")
            else:
                error_msg = f"Unexpected URL: {url}"
                raise ValueError(error_msg)
            return response

        mock_get.side_effect = mock_get_side_effect

        generator = WikipediaUrlGenerator(language="en")
        urls = generator._get_wikipedia_urls()

        expected_urls = [
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream2.xml.bz2",
        ]
        assert sorted(urls) == sorted(expected_urls)

    @patch("requests.get")
    def test_get_wikipedia_urls_with_specified_date(self, mock_get: Mock):
        """Test URL generation with specified dump date."""
        # Mock dumpstatus.json response
        mock_dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "enwiki-20230501-pages-articles-multistream1.xml.bz2": {},
                        "enwiki-20230501-pages-articles-multistream2.xml.bz2": {},
                        "enwiki-20230501-index.txt.bz2": {},  # Should be filtered out
                    },
                }
            }
        }
        mock_response = Mock()
        mock_response.content = json.dumps(mock_dump_data).encode("utf-8")
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(language="en", dump_date="20230501")
        urls = generator._get_wikipedia_urls()

        expected_urls = [
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream2.xml.bz2",
        ]
        assert sorted(urls) == sorted(expected_urls)

    @patch("requests.get")
    def test_get_wikipedia_urls_network_error(self, mock_get: Mock):
        """Test handling of network errors when fetching dump data."""
        mock_get.side_effect = requests.RequestException("Network error")

        generator = WikipediaUrlGenerator(language="en")

        with pytest.raises(requests.RequestException):
            generator._get_wikipedia_urls()

    @patch("requests.get")
    def test_get_wikipedia_urls_invalid_json(self, mock_get: Mock):
        """Test handling of invalid JSON in dump status."""
        mock_response = Mock()
        mock_response.content = b"Invalid JSON content"
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(language="en", dump_date="20230501")

        with pytest.raises(ValueError, match="Unable to load dump data for 20230501"):
            generator._get_wikipedia_urls()

    @patch("requests.get")
    def test_get_wikipedia_urls_empty_files(self, mock_get: Mock):
        """Test handling when dump has no XML files."""
        mock_dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "enwiki-20230501-index.txt.bz2": {},
                        "enwiki-20230501-other.txt.bz2": {},
                    },
                }
            }
        }
        mock_response = Mock()
        mock_response.content = json.dumps(mock_dump_data).encode("utf-8")
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(language="en", dump_date="20230501")
        urls = generator._get_wikipedia_urls()

        assert urls == []

    @patch("requests.get")
    def test_get_wikipedia_urls_missing_articles_job(self, mock_get: Mock):
        """Test handling when dump status doesn't have articlesmultistreamdump job."""
        mock_dump_data = {"jobs": {"other_job": {"files": {}}}}
        mock_response = Mock()
        mock_response.content = json.dumps(mock_dump_data).encode("utf-8")
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(language="en", dump_date="20230501")

        with pytest.raises(KeyError):
            generator._get_wikipedia_urls()

    @patch("requests.get")
    def test_get_wikipedia_urls_different_languages(self, mock_get: Mock):
        """Test URL generation for different languages."""
        mock_dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "eswiki-20230501-pages-articles-multistream1.xml.bz2": {},
                    },
                }
            }
        }
        mock_response = Mock()
        mock_response.content = json.dumps(mock_dump_data).encode("utf-8")
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(language="es", dump_date="20230501")
        urls = generator._get_wikipedia_urls()

        expected_urls = [
            "https://dumps.wikimedia.org/eswiki/20230501/eswiki-20230501-pages-articles-multistream1.xml.bz2",
        ]
        assert urls == expected_urls

    @patch("requests.get")
    def test_get_wikipedia_urls_custom_prefix(self, mock_get: Mock):
        """Test URL generation with custom wikidumps prefix."""
        mock_dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "enwiki-20230501-pages-articles-multistream1.xml.bz2": {},
                    },
                }
            }
        }
        mock_response = Mock()
        mock_response.content = json.dumps(mock_dump_data).encode("utf-8")
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(
            language="en", dump_date="20230501", wikidumps_index_prefix="https://custom.dumps.org"
        )
        urls = generator._get_wikipedia_urls()

        expected_urls = [
            "https://custom.dumps.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2",
        ]
        assert urls == expected_urls

    @patch.object(WikipediaUrlGenerator, "_get_wikipedia_urls")
    def test_generate_urls(self, mock_get_urls: Mock):
        """Test the generate_urls method."""
        mock_urls = [
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream2.xml.bz2",
        ]
        mock_get_urls.return_value = mock_urls

        generator = WikipediaUrlGenerator(language="en")
        urls = generator.generate_urls()

        assert urls == mock_urls
        mock_get_urls.assert_called_once()

    def test_beautiful_soup_parsing(self):
        """Test that BeautifulSoup parsing works correctly."""
        html_content = """
        <html>
        <body>
        <a href="../">../</a>
        <a href="20230401/">20230401/</a>
        <a href="20230420/">20230420/</a>
        <a href="20230501/">20230501/</a>
        <a href="latest/">latest/</a>
        </body>
        </html>
        """

        soup = BeautifulSoup(html_content, "lxml")
        dumps = soup.find_all("a")

        # Test that we get the expected number of links
        assert len(dumps) == 5

        # Test that we get the expected third-to-last dump date
        dump_date = dumps[-3].text
        assert dump_date == "20230420/"


class TestWikipediaUrlGeneratorIntegration:
    """Integration tests for WikipediaUrlGenerator."""

    @patch("requests.get")
    def test_realistic_scenario_latest_dump(self, mock_get: Mock):
        """Test a realistic scenario getting the latest dump."""
        # First call - getting index page
        index_html = """
        <html>
        <body>
        <a href="../">../</a>
        <a href="20230401/">20230401/</a>
        <a href="20230420/">20230420/</a>
        <a href="20230501/">20230501/</a>
        <a href="20230520/">20230520/</a>
        <a href="latest/">latest/</a>
        </body>
        </html>
        """

        # Second call - getting dumpstatus.json
        dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "enwiki-20230501-pages-articles-multistream1.xml.bz2": {},
                        "enwiki-20230501-pages-articles-multistream2.xml.bz2": {},
                        "enwiki-20230501-pages-articles-multistream3.xml.bz2": {},
                        "enwiki-20230501-index.txt.bz2": {},
                    },
                }
            }
        }

        def mock_get_side_effect(url: str, **kwargs) -> Mock:  # noqa: ARG001
            response = Mock()
            if url == "https://dumps.wikimedia.org/enwiki":
                response.content = index_html.encode("utf-8")
            elif url == "https://dumps.wikimedia.org/enwiki/20230520/dumpstatus.json":
                # Mock 20230520 as not finished, so it tries the next one
                response.content = json.dumps({"jobs": {"articlesmultistreamdump": {"status": "running"}}}).encode(
                    "utf-8"
                )
            elif url == "https://dumps.wikimedia.org/enwiki/20230501/dumpstatus.json":
                response.content = json.dumps(dump_data).encode("utf-8")
            else:
                error_msg = f"Unexpected URL: {url}"
                raise ValueError(error_msg)
            return response

        mock_get.side_effect = mock_get_side_effect

        generator = WikipediaUrlGenerator(language="en")
        urls = generator.generate_urls()

        expected_urls = [
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream2.xml.bz2",
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream3.xml.bz2",
        ]
        assert sorted(urls) == sorted(expected_urls)
        assert mock_get.call_count == 3  # Index page + 2 dump status calls

    @patch("requests.get")
    def test_realistic_scenario_specific_dump(self, mock_get: Mock):
        """Test a realistic scenario with specific dump date."""
        dump_data = {
            "jobs": {
                "articlesmultistreamdump": {
                    "status": "done",
                    "files": {
                        "eswiki-20230301-pages-articles-multistream1.xml.bz2": {},
                        "eswiki-20230301-pages-articles-multistream2.xml.bz2": {},
                    },
                }
            }
        }

        mock_response = Mock()
        mock_response.content = json.dumps(dump_data).encode("utf-8")
        mock_get.return_value = mock_response

        generator = WikipediaUrlGenerator(language="es", dump_date="20230301")
        urls = generator.generate_urls()

        expected_urls = [
            "https://dumps.wikimedia.org/eswiki/20230301/eswiki-20230301-pages-articles-multistream1.xml.bz2",
            "https://dumps.wikimedia.org/eswiki/20230301/eswiki-20230301-pages-articles-multistream2.xml.bz2",
        ]
        assert sorted(urls) == sorted(expected_urls)
        mock_get.assert_called_once_with("https://dumps.wikimedia.org/eswiki/20230301/dumpstatus.json", timeout=30)
