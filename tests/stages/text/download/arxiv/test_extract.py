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

from nemo_curator.stages.text.download.arxiv.extract import ArxivExtractor


class TestArxivExtractor:
    """Test suite for ArxivExtractor."""

    def test_extract_empty_content_returns_none(self) -> None:
        """Test that records with empty content return None."""
        extractor = ArxivExtractor()

        # Record with empty content
        record_empty_content = {
            "id": "1234567890",
            "source_id": "test.tar",
            "content": b"",
        }

        result = extractor.extract(record_empty_content)
        assert result is None

        # Record without content field
        record_without_content = {
            "id": "1234567890",
            "source_id": "test.tar",
        }

        result = extractor.extract(record_without_content)
        assert result is None

    def test_input_columns(self) -> None:
        """Test that input_columns returns the expected column names."""
        extractor = ArxivExtractor()
        columns = extractor.input_columns()

        expected_columns = ["id", "source_id", "content"]
        assert columns == expected_columns

    def test_output_columns(self) -> None:
        """Test that output_columns returns the expected column names."""
        extractor = ArxivExtractor()
        columns = extractor.output_columns()

        expected_columns = ["text"]
        assert columns == expected_columns

    def test_arxiv_extractor(self) -> None:
        extractor = ArxivExtractor()
        # Create a minimal LaTeX document including comments and a section header.
        content = r"""
        % This is a comment line that should be removed.
        \section{Introduction}
        This is the introduction of the paper.
        % Another comment that should vanish.
        """
        result = extractor.extract({"content": [content]})
        assert isinstance(result, dict)
        extracted_text = result.get("text", "")
        # Verify that comments have been removed.
        assert "% This is a comment" not in extracted_text
        # Verify that the section header content is retained.
        assert "Introduction" in extracted_text
        assert "This is the introduction" in extracted_text
