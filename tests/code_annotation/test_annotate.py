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

"""Tests for the code_annotation library."""

import pandas as pd
import pytest
from code_annotation import LANGUAGE_COL_NAME, annotate


class TestDetectLanguage:
    """Tests for detect_language annotation."""

    def test_detect_python(self):
        """Test Python language detection."""
        df = pd.DataFrame(
            {
                "content": ["def hello(): pass"],
                "representative_filename": ["test.py"],
            }
        )
        result = annotate({"detect_language": {}}, df)

        assert LANGUAGE_COL_NAME in result.columns
        assert "language_detector" in result.columns
        assert result[LANGUAGE_COL_NAME].iloc[0] == "Python"

    def test_detect_rust(self):
        """Test Rust language detection."""
        df = pd.DataFrame(
            {
                "content": ['fn main() { println!("hello"); }'],
                "representative_filename": ["main.rs"],
            }
        )
        result = annotate({"detect_language": {}}, df)

        assert result[LANGUAGE_COL_NAME].iloc[0] == "Rust"

    def test_detect_java(self):
        """Test Java language detection."""
        df = pd.DataFrame(
            {
                "content": ["public class Test { public static void main(String[] args) {} }"],
                "representative_filename": ["Test.java"],
            }
        )
        result = annotate({"detect_language": {}}, df)

        assert result[LANGUAGE_COL_NAME].iloc[0] == "Java"

    def test_detect_multiple_languages(self):
        """Test detection of multiple languages in batch."""
        df = pd.DataFrame(
            {
                "content": [
                    "def hello(): pass",
                    "fn main() {}",
                    "public class Test {}",
                    "console.log('hello');",
                ],
                "representative_filename": ["test.py", "main.rs", "Test.java", "index.js"],
            }
        )
        result = annotate({"detect_language": {}}, df)

        assert len(result) == 4
        assert result[LANGUAGE_COL_NAME].iloc[0] == "Python"
        assert result[LANGUAGE_COL_NAME].iloc[1] == "Rust"
        assert result[LANGUAGE_COL_NAME].iloc[2] == "Java"
        assert result[LANGUAGE_COL_NAME].iloc[3] == "JavaScript"

    def test_detect_without_filename(self):
        """Test language detection without filename hint."""
        df = pd.DataFrame(
            {
                "content": ["#!/usr/bin/env python\nprint('hello')"],
            }
        )
        result = annotate({"detect_language": {}}, df)

        # Should still detect via shebang or heuristics
        assert LANGUAGE_COL_NAME in result.columns


class TestBasicAnnotation:
    """Tests for basic annotation."""

    def test_basic_stats(self):
        """Test basic statistics calculation."""
        df = pd.DataFrame(
            {
                "content": ["line1\nline2\nline3"],
                "representative_filename": ["test.txt"],
            }
        )
        result = annotate({"basic": {}}, df)

        assert "basic_num_bytes" in result.columns
        assert "basic_num_lines" in result.columns
        assert "basic_max_line_length" in result.columns
        assert "basic_valid_utf8" in result.columns
        assert "basic_alpha_percent" in result.columns
        assert "basic_alnum_percent" in result.columns

        assert result["basic_num_bytes"].iloc[0] == 17
        assert result["basic_num_lines"].iloc[0] == 3

    def test_basic_xml_detection(self):
        """Test XML header detection."""
        df = pd.DataFrame(
            {
                "content": ['<?xml version="1.0"?>\n<root></root>'],
                "representative_filename": ["test.xml"],
            }
        )
        result = annotate({"basic": {"xml_header_search_length": 100}}, df)

        assert result["basic_contains_xml_header"].iloc[0] == True  # noqa: E712

    def test_basic_non_xml(self):
        """Test non-XML content."""
        df = pd.DataFrame(
            {
                "content": ["def hello(): pass"],
                "representative_filename": ["test.py"],
            }
        )
        result = annotate({"basic": {}}, df)

        assert result["basic_contains_xml_header"].iloc[0] == False  # noqa: E712

    def test_basic_alpha_percent(self):
        """Test alpha percentage calculation."""
        df = pd.DataFrame(
            {
                "content": ["abc123"],  # 3 alpha out of 6
                "representative_filename": ["test.txt"],
            }
        )
        result = annotate({"basic": {}}, df)

        assert result["basic_alpha_percent"].iloc[0] == pytest.approx(0.5, rel=0.01)


class TestSoftwareMetrics:
    """Tests for software_metrics annotation."""

    def test_software_metrics_python(self):
        """Test software metrics for Python code."""
        df = pd.DataFrame(
            {
                "content": [
                    """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
                ],
                "representative_filename": ["factorial.py"],
            }
        )
        # Need language detection first
        result = annotate({"detect_language": {}, "software_metrics": {}}, df)

        assert "software_metrics_cyclomatic_complexity" in result.columns
        assert "software_metrics_cognitive_complexity" in result.columns
        assert "software_metrics_maintainability_index" in result.columns
        assert "software_metrics_parsed_ok" in result.columns

    def test_software_metrics_rust(self):
        """Test software metrics for Rust code."""
        df = pd.DataFrame(
            {
                "content": [
                    """fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
"""
                ],
                "representative_filename": ["factorial.rs"],
            }
        )
        result = annotate({"detect_language": {}, "software_metrics": {}}, df)

        assert result["software_metrics_parsed_ok"].iloc[0] == True  # noqa: E712


class TestOpenCoderRs:
    """Tests for opencoder_rs annotation."""

    def test_opencoder_comment_fractions(self):
        """Test comment fraction calculation."""
        df = pd.DataFrame(
            {
                "content": [
                    """def hello():
    # This is a comment
    print("hello")
"""
                ],
                "representative_filename": ["test.py"],
            }
        )
        result = annotate({"detect_language": {}, "opencoder_rs": {}}, df)

        assert "ors_comment_lines_frac" in result.columns
        assert "ors_comment_chars_frac" in result.columns
        assert result["ors_comment_lines_frac"].iloc[0] > 0

    def test_opencoder_no_comments(self):
        """Test code with no comments."""
        df = pd.DataFrame(
            {
                "content": ["def hello(): pass"],
                "representative_filename": ["test.py"],
            }
        )
        result = annotate({"detect_language": {}, "opencoder_rs": {}}, df)

        assert result["ors_comment_lines_frac"].iloc[0] == 0


class TestTokenize:
    """Tests for tokenize annotation."""

    def test_tokenize_default(self):
        """Test tokenization with default tokenizer."""
        df = pd.DataFrame(
            {
                "content": ["def hello(): pass"],
                "representative_filename": ["test.py"],
            }
        )
        result = annotate({"tokenize": {}}, df)

        assert "num_tokens_github_o200k_base" in result.columns
        assert result["num_tokens_github_o200k_base"].iloc[0] > 0

    def test_tokenize_tiktoken(self):
        """Test tokenization with tiktoken tokenizer."""
        df = pd.DataFrame(
            {
                "content": ["def hello(): pass"],
                "representative_filename": ["test.py"],
            }
        )
        result = annotate({"tokenize": {"tokenizer_name": "tiktoken_o200k_base"}}, df)

        assert "num_tokens_tiktoken_o200k_base" in result.columns
        assert result["num_tokens_tiktoken_o200k_base"].iloc[0] > 0

    def test_tokenize_longer_code(self):
        """Test tokenization produces more tokens for longer code."""
        df = pd.DataFrame(
            {
                "content": [
                    "x",
                    "def hello(): pass",
                    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                ],
                "representative_filename": ["a.py", "b.py", "c.py"],
            }
        )
        result = annotate({"tokenize": {}}, df)

        tokens = result["num_tokens_github_o200k_base"].tolist()
        assert tokens[0] < tokens[1] < tokens[2]


class TestMultipleAnnotations:
    """Tests for running multiple annotations."""

    def test_chain_all_annotations(self):
        """Test chaining all annotation types."""
        df = pd.DataFrame(
            {
                "content": [
                    """def factorial(n):
    # Calculate factorial
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
                ],
                "representative_filename": ["factorial.py"],
            }
        )
        result = annotate(
            {
                "detect_language": {},
                "basic": {},
                "software_metrics": {},
                "opencoder_rs": {},
                "tokenize": {},
            },
            df,
        )

        # Check all annotation types are present
        assert LANGUAGE_COL_NAME in result.columns
        assert "basic_num_bytes" in result.columns
        assert "software_metrics_cyclomatic_complexity" in result.columns
        assert "ors_comment_lines_frac" in result.columns
        assert "num_tokens_github_o200k_base" in result.columns


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content(self):
        """Test handling of empty content."""
        df = pd.DataFrame(
            {
                "content": [""],
                "representative_filename": ["empty.py"],
            }
        )
        result = annotate({"basic": {}}, df)

        assert result["basic_num_bytes"].iloc[0] == 0

    def test_missing_content_column(self):
        """Test error on missing content column."""
        df = pd.DataFrame(
            {
                "text": ["def hello(): pass"],  # Wrong column name
            }
        )
        with pytest.raises(ValueError, match="content"):
            annotate({"detect_language": {}}, df)

    def test_none_values(self):
        """Test handling of None values in content."""
        df = pd.DataFrame(
            {
                "content": ["def hello(): pass", None, "fn main() {}"],
                "representative_filename": ["test.py", "none.txt", "main.rs"],
            }
        )
        result = annotate({"detect_language": {}}, df)

        assert result[LANGUAGE_COL_NAME].iloc[0] == "Python"
        assert pd.isna(result[LANGUAGE_COL_NAME].iloc[1])
        assert result[LANGUAGE_COL_NAME].iloc[2] == "Rust"

    def test_large_batch(self):
        """Test processing a larger batch."""
        n = 100
        df = pd.DataFrame(
            {
                "content": ["def func_%d(): pass" % i for i in range(n)],
                "representative_filename": ["test_%d.py" % i for i in range(n)],
            }
        )
        result = annotate({"detect_language": {}, "basic": {}}, df)

        assert len(result) == n
        assert all(result[LANGUAGE_COL_NAME] == "Python")
