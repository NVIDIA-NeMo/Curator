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

import pytest

from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor
from nemo_curator.stages.text.download.html_extractors import (
    JusTextExtractor,
    ResiliparseExtractor,
    TrafilaturaExtractor,
)


class TestCommonCrawlHTMLExtractor:
    """Test suite for CommonCrawlHTMLExtractor - focused on core logic correctness."""

    def test_extract_empty_content_returns_none(self) -> None:
        """Test that records with empty content return None."""
        html_extractor = CommonCrawlHTMLExtractor(algorithm="justext")

        # Record with empty content
        record_empty_content = {
            "url": "http://example.com",
            "warc_id": "test123",
            "source_id": "test.warc",
            "content": b"",
        }

        result = html_extractor.extract(record_empty_content)
        assert result is None

        # Record without content field
        record_without_content = {
            "url": "http://example.com",
            "warc_id": "test123",
            "source_id": "test.warc",
        }

        result = html_extractor.extract(record_without_content)
        assert result is None

    @pytest.mark.parametrize("algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_with_english_different_algorithms(
        self, algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        """Test extraction with different HTML extraction algorithms."""
        test_record = {
            "url": "http://example.com",
            "warc_id": "test123",
            "source_id": "test.warc",
            "content": b"<html><body><p>This is a comprehensive test paragraph with many important words and meaningful content. "
            b"We are testing the HTML extraction functionality to ensure that it works properly and efficiently "
            b"with various types of content. The system should be able to process this text correctly.</p></body></html>",
        }

        html_extractor = CommonCrawlHTMLExtractor(algorithm=algorithm)
        result = html_extractor.extract(test_record.copy())

        assert result is not None
        assert result["url"] == test_record["url"]
        assert result["warc_id"] == test_record["warc_id"]
        assert result["source_id"] == test_record["source_id"]
        assert result["language"] == "ENGLISH"
        assert result["text"] == test_record["content"].decode("utf-8").replace("<html><body><p>", "").replace(
            "</p></body></html>", ""
        )

        assert isinstance(result["text"], str)
        assert "content" not in result

    @pytest.mark.parametrize("algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_non_english_different_algorithms(
        self, algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        """Test language detection with non-English content."""
        html_extractor = CommonCrawlHTMLExtractor(algorithm=algorithm)

        # Record with Thai content (should be detected as THAI)
        record = {
            "url": "http://example.com",
            "warc_id": "test123",
            "source_id": "test.warc",
            "content": b"<html><body><p>\xe0\xb8\x99\xe0\xb8\xb5\xe0\xb9\x88\xe0\xb8\x84\xe0\xb8\xb7\xe0\xb8\xad\xe0\xb8\x95\xe0\xb8\xb1\xe0\xb8\xa7\xe0\xb8\xad\xe0\xb8\xa2\xe0\xb9\x88\xe0\xb8\xb2\xe0\xb8\x87\xe0\xb8\xa2\xe0\xb9\x88\xe0\xb8\xad\xe0\xb8\xab\xe0\xb8\x99\xe0\xb9\x89\xe0\xb8\xb2 \xe0\xb9\x83\xe0\xb8\x99\xe0\xb8\x99\xe0\xb8\xb1\xe0\xb9\x89\xe0\xb8\x99\xe0\xb9\x80\xe0\xb8\xa3\xe0\xb8\xb2\xe0\xb9\x80\xe0\xb8\x82\xe0\xb8\xb5\xe0\xb8\xa2\xe0\xb8\x99\xe0\xb8\x84\xe0\xb8\xb3\xe0\xb8\x95\xe0\xb9\x88\xe0\xb8\xb2\xe0\xb8\x87\xe0\xb9\x86</p></body></html>",
        }

        result = html_extractor.extract(record)
        assert result is not None
        assert result["language"] in ["THAI"]  # Common detected languages
        assert result["text"] == record["content"].decode("utf-8").replace("<html><body><p>", "").replace(
            "</p></body></html>", ""
        )
        assert "content" not in result

    def test_input_columns(self) -> None:
        """Test that input_columns returns the expected column names."""
        html_extractor = CommonCrawlHTMLExtractor(algorithm="justext")
        columns = html_extractor.input_columns()

        expected_columns = ["url", "warc_id", "source_id", "content"]
        assert columns == expected_columns

    def test_output_columns(self) -> None:
        """Test that output_columns returns the expected column names."""
        html_extractor = CommonCrawlHTMLExtractor(algorithm="justext")
        columns = html_extractor.output_columns()

        expected_columns = ["url", "warc_id", "source_id", "language", "text"]
        assert columns == expected_columns

    def test_algorithm_instantiation_with_string(self) -> None:
        """Test that string algorithm names work correctly."""
        # Test justext
        html_extractor_justext = CommonCrawlHTMLExtractor(algorithm="justext")
        assert isinstance(html_extractor_justext.algorithm, JusTextExtractor)

        # Test resiliparse
        html_extractor_resiliparse = CommonCrawlHTMLExtractor(algorithm="resiliparse")
        assert isinstance(html_extractor_resiliparse.algorithm, ResiliparseExtractor)

        # Test trafilatura
        html_extractor_trafilatura = CommonCrawlHTMLExtractor(algorithm="trafilatura")
        assert isinstance(html_extractor_trafilatura.algorithm, TrafilaturaExtractor)

    def test_algorithm_instantiation_with_object(self) -> None:
        """Test that algorithm objects can be passed directly."""
        custom_algorithm = JusTextExtractor(length_low=50, length_high=150)
        html_extractor = CommonCrawlHTMLExtractor(algorithm=custom_algorithm)

        assert html_extractor.algorithm is custom_algorithm
        # Type cast to access JusTextExtractor-specific attributes
        assert isinstance(html_extractor.algorithm, JusTextExtractor)
        assert html_extractor.algorithm.length_low == 50
        assert html_extractor.algorithm.length_high == 150

    def test_default_algorithm_instantiation(self) -> None:
        """Test that default algorithm (justext) is used when none specified."""
        html_extractor = CommonCrawlHTMLExtractor()
        assert isinstance(html_extractor.algorithm, JusTextExtractor)

    def test_invalid_algorithm_raises_error(self) -> None:
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Invalid algorithm"):
            CommonCrawlHTMLExtractor(algorithm="invalid_algorithm")

        with pytest.raises(ValueError, match="Invalid algorithm"):
            # Test with an invalid type (ignoring type checking for test purposes)
            CommonCrawlHTMLExtractor(algorithm=123)  # type: ignore[arg-type]

    def test_algorithm_kwargs_passed_correctly(self) -> None:
        """Test that algorithm kwargs are passed to the algorithm constructor."""
        algorithm_kwargs = {"length_low": 80, "length_high": 180, "stopwords_low": 0.25}
        html_extractor = CommonCrawlHTMLExtractor(algorithm="justext", algorithm_kwargs=algorithm_kwargs)

        assert isinstance(html_extractor.algorithm, JusTextExtractor)
        # Access JusTextExtractor-specific attributes after isinstance check
        justext_algorithm = html_extractor.algorithm
        assert justext_algorithm.length_low == 80
        assert justext_algorithm.length_high == 180
        assert justext_algorithm.stopwords_low == 0.25

    def test_custom_stop_lists(self) -> None:
        """Test that custom stop lists can be provided."""
        custom_stop_lists = {
            "ENGLISH": frozenset(["the", "and", "or", "but"]),
            "CUSTOM": frozenset(["foo", "bar", "baz"]),
        }

        html_extractor = CommonCrawlHTMLExtractor(algorithm="justext", stop_lists=custom_stop_lists)

        assert html_extractor._stop_lists == custom_stop_lists
        assert "ENGLISH" in html_extractor._stop_lists
        assert "CUSTOM" in html_extractor._stop_lists
