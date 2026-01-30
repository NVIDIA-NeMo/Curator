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

"""Minimal code annotation library for NeMo Curator.

This module provides Rust-based annotation functions:
- detect_language: Programming language detection via hyperpolyglot
- basic: Basic statistics (byte count, UTF-8 validation, line stats, patterns)
- tokenize: BPE tokenization
- software_metrics: Code complexity metrics via rust-code-analysis
- opencoder_rs: Comment line/character fractions

Usage:
    import pandas as pd
    from code_annotation import annotate

    df = pd.DataFrame({
        "content": ["def hello(): pass", "fn main() {}"],
        "representative_filename": ["test.py", "main.rs"],
    })

    # Run language detection
    result = annotate({"detect_language": {}}, df)
"""

from typing import Any

import pandas as pd
import pyarrow as pa

from code_annotation._code_annotation import annotate as rust_annotate

CODE_COL_NAME = "content"
FILENAME_COL_NAME = "representative_filename"
LANGUAGE_COL_NAME = "language"


def annotate(config_dict: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Annotate the DataFrame based on the provided config.

    Args:
        config_dict: Annotation configs. Keys are function names, values are kwargs.
            Supported functions:
            - "detect_language": {} - Detects programming language
            - "basic": {"xml_header_search_length": 100, "max_decompressed_byte_size": None}
            - "tokenize": {"tokenizer_name": "github_o200k_base", "vocab": None, "pretokenizer_patterns": None}
            - "software_metrics": {} - Requires "language" column from detect_language
            - "opencoder_rs": {} - Requires "language" column from detect_language
        df: A pandas DataFrame with "content" column (strings) and optionally
            "representative_filename" column for language detection.

    Returns:
        A pandas DataFrame with annotations applied.

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> df = pd.DataFrame({"content": ["print('hello')"], "representative_filename": ["test.py"]})
        >>> result = annotate({"detect_language": {}}, df)
        >>> print(result["language"][0])
        'Python'
    """
    if CODE_COL_NAME not in df.columns:
        msg = f"DataFrame must have '{CODE_COL_NAME}' column with code strings"
        raise ValueError(msg)

    # Convert pandas to PyArrow Table for Rust processing
    arrow_table = pa.Table.from_pandas(df)

    # Call Rust function with PyArrow table
    result_table = rust_annotate(config_dict, arrow_table)

    # Convert back to pandas
    return result_table.to_pandas()


__all__ = ["CODE_COL_NAME", "FILENAME_COL_NAME", "LANGUAGE_COL_NAME", "annotate"]
