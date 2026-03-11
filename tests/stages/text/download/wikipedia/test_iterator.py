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

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from nemo_curator.stages.text.download.wikipedia.iterator import WikipediaIterator


class TestWikipediaIterator:
    """Test suite for WikipediaIterator."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        iterator = WikipediaIterator()
        assert iterator._language == "en"
        assert iterator._log_frequency == 1000
        assert iterator._counter == 0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        iterator = WikipediaIterator(language="es", log_frequency=500)
        assert iterator._language == "es"
        assert iterator._log_frequency == 500
        assert iterator._counter == 0

    def test_output_columns(self):
        """Test that output_columns returns the expected column names."""
        iterator = WikipediaIterator()
        columns = iterator.output_columns()

        expected_columns = ["title", "id", "url", "language", "source_id", "raw_content"]
        assert columns == expected_columns

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_single_article(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test iteration over a single Wikipedia article."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create a mock element with proper namespace structure
        mock_page = mock.Mock()
        mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

        # Mock the sub-elements with proper text values
        mock_title = mock.Mock()
        mock_title.text = "Test Article"
        mock_ns = mock.Mock()
        mock_ns.text = "0"
        mock_id = mock.Mock()
        mock_id.text = "12345"
        mock_revision = mock.Mock()
        mock_text = mock.Mock()
        mock_text.text = "This is the article content."

        # Set up the find method to return appropriate elements
        def mock_find(path: str) -> Any:  # noqa: PLR0911, ANN401
            if path.endswith("title"):
                return mock_title
            elif path.endswith("ns"):
                return mock_ns
            elif path.endswith("id"):
                return mock_id
            elif path.endswith("redirect"):
                return None
            elif path.endswith("revision"):
                return mock_revision
            elif path.endswith("text"):
                return mock_text
            return None

        mock_page.find = mock_find
        mock_revision.find = lambda path: mock_text if path.endswith("text") else None

        # Set up the iterparse to return our mock page
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = [(None, mock_page)]

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            assert len(results) == 1
            result = results[0]

            assert result["title"] == "Test Article"
            assert result["id"] == "12345"
            assert result["url"] == "https://en.wikipedia.org/wiki/Test%20Article"
            assert result["language"] == "en"
            assert result["source_id"] == "test.xml.bz2"
            assert result["raw_content"] == "This is the article content."

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_filters_non_main_namespace(
        self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path
    ):
        """Test that iteration filters out non-main namespace pages."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create test data with different namespaces
        test_data = [
            ("Main Article", "0", "11111", "Main article content"),
            ("Talk:Main Article", "1", "22222", "Talk page content"),
            ("User:TestUser", "2", "33333", "User page content"),
            ("Another Article", "0", "44444", "Another main article"),
        ]

        mock_pages = []
        for title, ns, id_, content in test_data:
            mock_page = mock.Mock()
            mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

            mock_title = mock.Mock()
            mock_title.text = title
            mock_ns_elem = mock.Mock()
            mock_ns_elem.text = ns
            mock_id = mock.Mock()
            mock_id.text = id_
            mock_revision = mock.Mock()
            mock_text = mock.Mock()
            mock_text.text = content

            def mock_find(path: str, title: str = title, ns: str = ns, id_: str = id_) -> Any:  # noqa: PLR0911, ANN401
                if path.endswith("title"):
                    mock_title.text = title  # noqa: B023
                    return mock_title  # noqa: B023
                elif path.endswith("ns"):
                    mock_ns_elem.text = ns  # noqa: B023
                    return mock_ns_elem  # noqa: B023
                elif path.endswith("id"):
                    mock_id.text = id_  # noqa: B023
                    return mock_id  # noqa: B023
                elif path.endswith("redirect"):
                    return None
                elif path.endswith("revision"):
                    return mock_revision  # noqa: B023
                elif path.endswith("text"):
                    return mock_text  # noqa: B023
                return None

            mock_page.find = mock_find
            mock_pages.append((None, mock_page))

        # Set up the iterparse to return our mock pages
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = mock_pages

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            # Only articles from namespace 0 should be returned
            assert len(results) == 2
            assert results[0]["title"] == "Main Article"
            assert results[1]["title"] == "Another Article"

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_filters_redirects(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test that iteration filters out redirect pages."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create test data with redirects
        test_data = [
            ("Regular Article", "11111", "Regular article content", False),
            ("Redirect Page", "22222", "Redirect content", True),
            ("Another Article", "33333", "Another article content", False),
        ]

        mock_pages = []
        for title, id_, content, is_redirect in test_data:
            mock_page = mock.Mock()
            mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

            mock_title = mock.Mock()
            mock_title.text = title
            mock_ns = mock.Mock()
            mock_ns.text = "0"
            mock_id = mock.Mock()
            mock_id.text = id_
            mock_revision = mock.Mock()
            mock_text = mock.Mock()
            mock_text.text = content
            mock_redirect = mock.Mock() if is_redirect else None

            def mock_find(  # noqa: PLR0911
                path: str, _: str = title, __: str = id_, ___: str = content, ____: bool = is_redirect
            ) -> Any:  # noqa: ANN401
                if path.endswith("title"):
                    return mock_title  # noqa: B023
                elif path.endswith("ns"):
                    return mock_ns  # noqa: B023
                elif path.endswith("id"):
                    return mock_id  # noqa: B023
                elif path.endswith("redirect"):
                    return mock_redirect  # noqa: B023
                elif path.endswith("revision"):
                    return mock_revision  # noqa: B023
                elif path.endswith("text"):
                    return mock_text  # noqa: B023
                return None

            mock_page.find = mock_find
            mock_revision.find = lambda path: mock_text if path.endswith("text") else None  # noqa: B023
            mock_pages.append((None, mock_page))

        # Set up the iterparse to return our mock pages
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = mock_pages

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            # Only non-redirect articles should be returned
            assert len(results) == 3
            assert results[1]["title"] == "Another Article"

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_handles_empty_content(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test that iteration handles empty content properly."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create a mock element with empty content
        mock_page = mock.Mock()
        mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

        mock_title = mock.Mock()
        mock_title.text = "Empty Article"
        mock_ns = mock.Mock()
        mock_ns.text = "0"
        mock_id = mock.Mock()
        mock_id.text = "12345"
        mock_revision = mock.Mock()
        mock_text = mock.Mock()
        mock_text.text = ""

        def mock_find(path: str) -> Any:  # noqa: PLR0911, ANN401
            if path.endswith("title"):
                return mock_title
            elif path.endswith("ns"):
                return mock_ns
            elif path.endswith("id"):
                return mock_id
            elif path.endswith("redirect"):
                return None
            elif path.endswith("revision"):
                return mock_revision
            elif path.endswith("text"):
                return mock_text
            return None

        mock_page.find = mock_find
        mock_revision.find = lambda path: mock_text if path.endswith("text") else None

        # Set up the iterparse to return our mock page
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = [(None, mock_page)]

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            # Empty content should still be returned (filtering is done at extraction level)
            assert len(results) == 1
            result = results[0]
            assert result["title"] == "Empty Article"
            assert result["raw_content"] == ""

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_handles_missing_elements(
        self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path
    ):
        """Test that iteration handles missing required elements."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create test data with missing elements
        test_data = [
            ("Good Article", "0", "11111", "Good content"),
            (None, "0", "22222", "Missing title"),  # Missing title
            ("No ID", "0", None, "No ID content"),  # Missing ID
            ("No NS", None, "33333", "No NS content"),  # Missing namespace
        ]

        mock_pages = []
        for title, ns, id_, content in test_data:
            mock_page = mock.Mock()
            mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

            mock_title = mock.Mock()
            mock_title.text = title
            mock_ns = mock.Mock()
            mock_ns.text = ns
            mock_id = mock.Mock()
            mock_id.text = id_
            mock_revision = mock.Mock()
            mock_text = mock.Mock()
            mock_text.text = content

            def mock_find(path: str, title: str = title, ns: str = ns, id_: str = id_) -> Any:  # noqa: ANN401
                if path.endswith("title"):
                    return mock_title if title is not None else None  # noqa: B023
                elif path.endswith("ns"):
                    return mock_ns if ns is not None else None  # noqa: B023
                elif path.endswith("id"):
                    return mock_id if id_ is not None else None  # noqa: B023
                elif path.endswith("revision"):
                    return mock_revision  # noqa: B023
                elif path.endswith("text"):
                    return mock_text  # noqa: B023
                return None

            mock_page.find = mock_find
            mock_revision.find = lambda path: mock_text if path.endswith("text") else None  # noqa: B023
            mock_pages.append((None, mock_page))

        # Set up the iterparse to return our mock pages
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = mock_pages

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            # Only the complete article should be returned
            assert len(results) == 0

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_different_languages(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test iteration with different language settings."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create a mock element for Spanish Wikipedia
        mock_page = mock.Mock()
        mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

        mock_title = mock.Mock()
        mock_title.text = "Artículo de prueba"
        mock_ns = mock.Mock()
        mock_ns.text = "0"
        mock_id = mock.Mock()
        mock_id.text = "12345"
        mock_revision = mock.Mock()
        mock_text = mock.Mock()
        mock_text.text = "'''Artículo de prueba''' es un artículo de prueba."

        def mock_find(path: str) -> Any:  # noqa: PLR0911, ANN401
            if path.endswith("title"):
                return mock_title
            elif path.endswith("ns"):
                return mock_ns
            elif path.endswith("id"):
                return mock_id
            elif path.endswith("redirect"):
                return None
            elif path.endswith("revision"):
                return mock_revision
            elif path.endswith("text"):
                return mock_text
            return None

        mock_page.find = mock_find

        # Set up the iterparse to return our mock page
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = [(None, mock_page)]

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator(language="es")
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            assert len(results) == 1
            result = results[0]

            assert result["title"] == "Artículo de prueba"
            assert result["language"] == "es"
            assert result["url"] == "https://es.wikipedia.org/wiki/Art%C3%ADculo%20de%20prueba"

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_url_encoding(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test that URLs are properly encoded."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create a mock element with special characters in title
        mock_page = mock.Mock()
        mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

        mock_title = mock.Mock()
        mock_title.text = "Article with spaces & special chars"
        mock_ns = mock.Mock()
        mock_ns.text = "0"
        mock_id = mock.Mock()
        mock_id.text = "12345"
        mock_revision = mock.Mock()
        mock_text = mock.Mock()
        mock_text.text = "Content with special characters."

        def mock_find(path: str) -> Any:  # noqa: PLR0911, ANN401
            if path.endswith("title"):
                return mock_title
            elif path.endswith("ns"):
                return mock_ns
            elif path.endswith("id"):
                return mock_id
            elif path.endswith("redirect"):
                return None
            elif path.endswith("revision"):
                return mock_revision
            elif path.endswith("text"):
                return mock_text
            return None

        mock_page.find = mock_find

        # Set up the iterparse to return our mock page
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = [(None, mock_page)]

        # Patch ET.iterparse
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            results = list(iterator.iterate(str(test_file)))

            assert len(results) == 1
            result = results[0]

            assert result["title"] == "Article with spaces & special chars"
            assert result["url"] == "https://en.wikipedia.org/wiki/Article%20with%20spaces%20%26%20special%20chars"

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_with_logging(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test that logging works correctly during iteration."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Create mock elements for multiple articles
        mock_pages = []
        for _i, (title, id_, content) in enumerate(
            [
                ("Article 1", "11111", "Content 1"),
                ("Article 2", "22222", "Content 2"),
                ("Article 3", "33333", "Content 3"),
            ]
        ):
            mock_page = mock.Mock()
            mock_page.tag = "{http://www.mediawiki.org/xml/export-0.10/}page"

            mock_title = mock.Mock()
            mock_title.text = title
            mock_ns = mock.Mock()
            mock_ns.text = "0"
            mock_id = mock.Mock()
            mock_id.text = id_
            mock_revision = mock.Mock()
            mock_text = mock.Mock()
            mock_text.text = content

            def mock_find(path: str, title: str = title, id_: str = id_, content: str = content) -> Any:  # noqa: PLR0911, ANN401
                if path.endswith("title"):
                    mock_title.text = title  # noqa: B023
                    return mock_title  # noqa: B023
                elif path.endswith("ns"):
                    return mock_ns  # noqa: B023
                elif path.endswith("id"):
                    mock_id.text = id_  # noqa: B023
                    return mock_id  # noqa: B023
                elif path.endswith("redirect"):
                    return None
                elif path.endswith("revision"):
                    return mock_revision  # noqa: B023
                elif path.endswith("text"):
                    mock_text.text = content  # noqa: B023
                    return mock_text  # noqa: B023
                return None

            mock_page.find = mock_find
            mock_pages.append((None, mock_page))

        # Set up the iterparse to return our mock pages
        mock_iterparse = mock.Mock()
        mock_iterparse.return_value = mock_pages

        # Patch ET.iterparse and logger
        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):  # noqa: SIM117
            with mock.patch("nemo_curator.stages.text.download.wikipedia.iterator.logger") as mock_logger:
                iterator = WikipediaIterator(log_frequency=2)
                test_file = tmp_path / "test.xml.bz2"
                test_file.write_text("dummy content")

                results = list(iterator.iterate(str(test_file)))

                assert len(results) == 3
                # Log should be called once when counter reaches 2
                mock_logger.debug.assert_called_once()

    @mock.patch("bz2.BZ2File")
    def test_iterate_file_error_handling(self, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test error handling when file cannot be opened."""
        mock_bz2file.side_effect = OSError("Cannot open file")

        iterator = WikipediaIterator()
        test_file = tmp_path / "nonexistent.xml.bz2"

        with pytest.raises(OSError, match="Cannot open file"):
            list(iterator.iterate(str(test_file)))

    @mock.patch("bz2.BZ2File")
    @mock.patch("codecs.getreader")
    def test_iterate_xml_parsing_error(self, mock_getreader: mock.Mock, mock_bz2file: mock.Mock, tmp_path: Path):
        """Test error handling for XML parsing errors."""
        # Mock the BZ2File and reader
        mock_file = mock.Mock()
        mock_bz2file.return_value.__enter__.return_value = mock_file
        mock_reader = mock.Mock()
        mock_getreader.return_value = mock_reader

        # Make iterparse raise an exception
        mock_iterparse = mock.Mock()
        mock_iterparse.side_effect = ET.ParseError("Invalid XML")

        with mock.patch("xml.etree.ElementTree.iterparse", mock_iterparse):
            iterator = WikipediaIterator()
            test_file = tmp_path / "test.xml.bz2"
            test_file.write_text("dummy content")

            with pytest.raises(ET.ParseError):
                list(iterator.iterate(str(test_file)))
