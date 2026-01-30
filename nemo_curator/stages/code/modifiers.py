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

"""Document modifiers for code annotation using the code_annotation Rust library.

These modifiers wrap the Rust-based annotation functions and apply them to
pandas DataFrames containing code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from code_annotation import annotate

from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier

if TYPE_CHECKING:
    from collections.abc import Callable


class CodeLanguageDetector(DocumentModifier):
    """Detect programming language for code documents.

    Uses hyperpolyglot for language detection based on:
    - Filename extension
    - Shebang lines
    - Content heuristics

    Adds columns:
        - language: Detected programming language (e.g., "Python", "Rust", "Java")
        - language_detector: Detection method used ("Extension", "Shebang", "Heuristics", etc.)

    Example:
        >>> modifier = CodeLanguageDetector()
        >>> df = pd.DataFrame({
        ...     "content": ["def hello(): pass"],
        ...     "representative_filename": ["test.py"]
        ... })
        >>> result = modifier.modify_document(df)
        >>> print(result["language"][0])
        'Python'
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "code_language_detector"

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect programming language for each document.

        Args:
            df: DataFrame with 'content' column and optionally 'representative_filename'.

        Returns:
            DataFrame with 'language' and 'language_detector' columns added.
        """
        return annotate({"detect_language": {}}, df)


class CodeBasicStats(DocumentModifier):
    """Compute basic statistics for code documents.

    Adds columns:
        - basic_num_bytes: Number of bytes in the content
        - basic_valid_utf8: Whether the content is valid UTF-8
        - basic_num_lines: Number of lines
        - basic_max_line_length: Maximum line length
        - basic_average_line_length: Average line length
        - basic_alpha_percent: Fraction of alphabetic characters
        - basic_alnum_percent: Fraction of alphanumeric characters
        - basic_base64_percent: Fraction of content matching base64 patterns
        - basic_hex_percent: Fraction of content matching hex patterns
        - basic_contains_xml_header: Whether content contains XML header

    Args:
        xml_header_search_length: Number of characters to search for XML header.
            Default is 100.

    Example:
        >>> modifier = CodeBasicStats()
        >>> df = pd.DataFrame({"content": ["line1\\nline2\\nline3"]})
        >>> result = modifier.modify_document(df)
        >>> print(result["basic_num_lines"][0])
        3
    """

    def __init__(self, xml_header_search_length: int = 100) -> None:
        super().__init__()
        self._name = "code_basic_stats"
        self._xml_header_search_length = xml_header_search_length

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic statistics for each document.

        Args:
            df: DataFrame with 'content' column.

        Returns:
            DataFrame with basic statistics columns added.
        """
        return annotate(
            {"basic": {"xml_header_search_length": self._xml_header_search_length}},
            df,
        )


class CodeSoftwareMetrics(DocumentModifier):
    """Compute software engineering metrics for code documents.

    Uses rust-code-analysis to compute complexity metrics. Requires the 'language'
    column to be present (run CodeLanguageDetector first).

    Adds columns:
        - software_metrics_cyclomatic_complexity: Cyclomatic complexity
        - software_metrics_cognitive_complexity: Cognitive complexity
        - software_metrics_maintainability_index: Maintainability index
        - software_metrics_halstead_difficulty: Halstead difficulty
        - software_metrics_comment_lines: Number of comment lines
        - software_metrics_blank_lines: Number of blank lines
        - software_metrics_parsed_ok: Whether parsing succeeded

    Example:
        >>> modifier = CodeSoftwareMetrics()
        >>> df = pd.DataFrame({
        ...     "content": ["def factorial(n):\\n    if n <= 1: return 1\\n    return n * factorial(n-1)"],
        ...     "representative_filename": ["fact.py"],
        ...     "language": ["Python"]
        ... })
        >>> result = modifier.modify_document(df)
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "code_software_metrics"

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute software metrics for each document.

        Args:
            df: DataFrame with 'content' and 'language' columns.

        Returns:
            DataFrame with software metrics columns added.
        """
        return annotate({"software_metrics": {}}, df)


class CodeOpenCoderMetrics(DocumentModifier):
    """Compute OpenCoder-style comment metrics for code documents.

    Computes the fraction of lines and characters that are comments using
    a vendored version of the loc crate. Requires the 'language' column
    to be present (run CodeLanguageDetector first).

    Adds columns:
        - ors_comment_lines_frac: Fraction of lines that are comments
        - ors_comment_chars_frac: Fraction of characters in comments

    Example:
        >>> modifier = CodeOpenCoderMetrics()
        >>> df = pd.DataFrame({
        ...     "content": ["# This is a comment\\ndef hello(): pass"],
        ...     "representative_filename": ["test.py"],
        ...     "language": ["Python"]
        ... })
        >>> result = modifier.modify_document(df)
        >>> print(result["ors_comment_lines_frac"][0] > 0)
        True
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "code_opencoder_metrics"

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OpenCoder metrics for each document.

        Args:
            df: DataFrame with 'content' and 'language' columns.

        Returns:
            DataFrame with OpenCoder metrics columns added.
        """
        return annotate({"opencoder_rs": {}}, df)


class CodeTokenizer(DocumentModifier):
    """Tokenize code documents using BPE tokenizers.

    Adds a column with the token count for each document.

    Args:
        tokenizer_name: Name of the tokenizer to use.
            - "github_o200k_base" (default): GitHub's o200k tokenizer
            - "tiktoken_o200k_base": OpenAI's tiktoken o200k tokenizer

    Adds columns:
        - num_tokens_{tokenizer_name}: Number of tokens

    Example:
        >>> modifier = CodeTokenizer()
        >>> df = pd.DataFrame({"content": ["def hello(): pass"]})
        >>> result = modifier.modify_document(df)
        >>> print(result["num_tokens_github_o200k_base"][0] > 0)
        True
    """

    def __init__(self, tokenizer_name: str = "github_o200k_base") -> None:
        super().__init__()
        self._name = "code_tokenizer"
        self._tokenizer_name = tokenizer_name

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tokenize each document.

        Args:
            df: DataFrame with 'content' column.

        Returns:
            DataFrame with token count column added.
        """
        return annotate({"tokenize": {"tokenizer_name": self._tokenizer_name}}, df)


class CodeAnnotator(DocumentModifier):
    """Apply multiple code annotations in a single pass.

    This is a convenience modifier that combines multiple annotation functions
    into a single call for efficiency.

    Args:
        detect_language: Whether to detect programming language. Default True.
        basic_stats: Whether to compute basic statistics. Default True.
        software_metrics: Whether to compute software metrics. Default False.
        opencoder_metrics: Whether to compute OpenCoder metrics. Default False.
        tokenize: Whether to tokenize. Default False.
        tokenizer_name: Tokenizer to use if tokenize=True.
        xml_header_search_length: Characters to search for XML header.

    Example:
        >>> modifier = CodeAnnotator(
        ...     detect_language=True,
        ...     basic_stats=True,
        ...     software_metrics=True
        ... )
        >>> df = pd.DataFrame({
        ...     "content": ["def hello(): pass"],
        ...     "representative_filename": ["test.py"]
        ... })
        >>> result = modifier.modify_document(df)
    """

    def __init__(  # noqa: PLR0913
        self,
        detect_language: bool = True,
        basic_stats: bool = True,
        software_metrics: bool = False,
        opencoder_metrics: bool = False,
        tokenize: bool = False,
        tokenizer_name: str = "github_o200k_base",
        xml_header_search_length: int = 100,
    ) -> None:
        super().__init__()
        self._name = "code_annotator"
        self._detect_language = detect_language
        self._basic_stats = basic_stats
        self._software_metrics = software_metrics
        self._opencoder_metrics = opencoder_metrics
        self._tokenize = tokenize
        self._tokenizer_name = tokenizer_name
        self._xml_header_search_length = xml_header_search_length

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply selected annotations to each document.

        Args:
            df: DataFrame with 'content' column and optionally 'representative_filename'.

        Returns:
            DataFrame with annotation columns added.
        """
        config: dict = {}

        if self._detect_language:
            config["detect_language"] = {}

        if self._basic_stats:
            config["basic"] = {"xml_header_search_length": self._xml_header_search_length}

        if self._software_metrics:
            config["software_metrics"] = {}

        if self._opencoder_metrics:
            config["opencoder_rs"] = {}

        if self._tokenize:
            config["tokenize"] = {"tokenizer_name": self._tokenizer_name}

        if not config:
            return df

        return annotate(config, df)


class CodeLicenseDetector(DocumentModifier):
    """Detect software licenses in code documents using scancode-toolkit.

    Uses scancode-toolkit's license detection to identify SPDX license keys
    in code content. This is useful for filtering code based on license
    compatibility or identifying permissively licensed code.

    Adds columns:
        - licenses: List of detected SPDX license keys
        - has_license: Boolean indicating if any license was detected
        - license_count: Number of licenses detected

    Args:
        detection_timeout: Timeout in seconds for license detection per file.
            Default is 100 seconds.
        content_column: Name of the column containing code content.
            Default is "content".

    Example:
        >>> modifier = CodeLicenseDetector()
        >>> df = pd.DataFrame({
        ...     "content": ["# MIT License\\n# Copyright 2024\\ndef hello(): pass"],
        ...     "representative_filename": ["test.py"]
        ... })
        >>> result = modifier.modify_document(df)
        >>> print(result["has_license"][0])
        True

    Note:
        Requires scancode-toolkit to be installed:
        ``pip install scancode-toolkit``
    """

    def __init__(
        self,
        detection_timeout: int = 100,
        content_column: str = "content",
    ) -> None:
        super().__init__()
        self._name = "code_license_detector"
        self._detection_timeout = detection_timeout
        self._content_column = content_column
        self._license_db: dict | None = None
        self._licensing: Callable | None = None

    def _lazy_load_scancode(self) -> None:
        """Lazily load scancode-toolkit components."""
        if self._license_db is None:
            try:
                from licensedcode.cache import get_licenses_db, get_licensing

                self._license_db = get_licenses_db()
                self._licensing = get_licensing()
            except ImportError as e:
                msg = "scancode-toolkit is required for license detection. Install with: pip install scancode-toolkit"
                raise ImportError(msg) from e

    def _detect_licenses(self, content: str) -> list[str]:
        """Detect licenses in the given content string.

        Args:
            content: The code content to analyze.

        Returns:
            List of detected SPDX license keys.
        """
        from licensedcode.detection import detect_licenses

        self._lazy_load_scancode()

        detected_licenses: set[str] = set()
        try:
            for lic in detect_licenses(
                query_string=content,
                deadline=self._detection_timeout,
            ):
                symbols = self._licensing.license_symbols(lic.license_expression)
                for sym in symbols:
                    spdx_key = self._license_db[sym.key].spdx_license_key
                    if spdx_key:
                        detected_licenses.add(spdx_key)
        except Exception:  # noqa: BLE001, S110
            # License detection can fail on malformed content
            pass

        return list(detected_licenses)

    def modify_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect licenses for each document.

        Args:
            df: DataFrame with content column.

        Returns:
            DataFrame with license detection columns added.
        """
        licenses_list = []
        has_license_list = []
        license_count_list = []

        for content in df[self._content_column]:
            if pd.isna(content) or not isinstance(content, str):
                licenses_list.append([])
                has_license_list.append(False)
                license_count_list.append(0)
                continue

            licenses = self._detect_licenses(content)
            licenses_list.append(licenses)
            has_license_list.append(len(licenses) > 0)
            license_count_list.append(len(licenses))

        df = df.copy()
        df["licenses"] = licenses_list
        df["has_license"] = has_license_list
        df["license_count"] = license_count_list

        return df
