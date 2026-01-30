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

"""Processing stages for code annotation pipelines.

These stages wrap the code annotation modifiers and filters for use in
NeMo Curator pipelines.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.code.modifiers import (
    CodeAnnotator,
    CodeLanguageDetector,
    CodeLicenseDetector,
)
from nemo_curator.stages.text.filters.code import (
    AlphaPercentFilter,
    AverageLineLengthFilter,
    Base64ContentFilter,
    CommentFractionFilter,
    CyclomaticComplexityFilter,
    HexContentFilter,
    MaxLineLengthFilter,
    TokenCountFilter,
)
from nemo_curator.tasks import DocumentBatch


@dataclass
class CodeAnnotation(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that applies code annotations to a DocumentBatch.

    This stage wraps the CodeAnnotator modifier to add language detection,
    basic statistics, software metrics, OpenCoder metrics, and tokenization
    columns to the data.

    Args:
        content_column: Column name containing code content. Default "content".
        filename_column: Column name containing filename. Default "rel_path".
            Will be used to create "representative_filename" if needed.
        detect_language: Whether to detect programming language. Default True.
        basic_stats: Whether to compute basic statistics. Default True.
        software_metrics: Whether to compute software metrics. Default False.
        opencoder_metrics: Whether to compute OpenCoder metrics. Default True.
        tokenize: Whether to tokenize. Default True.
        tokenizer_name: Tokenizer to use if tokenize=True.
        xml_header_search_length: Characters to search for XML header.
    """

    content_column: str = "content"
    filename_column: str = "rel_path"
    detect_language: bool = True
    basic_stats: bool = True
    software_metrics: bool = False
    opencoder_metrics: bool = True
    tokenize: bool = True
    tokenizer_name: str = "github_o200k_base"
    xml_header_search_length: int = 100
    name: str = "code_annotation"

    def __post_init__(self) -> None:
        self._annotator = CodeAnnotator(
            detect_language=self.detect_language,
            basic_stats=self.basic_stats,
            software_metrics=self.software_metrics,
            opencoder_metrics=self.opencoder_metrics,
            tokenize=self.tokenize,
            tokenizer_name=self.tokenizer_name,
            xml_header_search_length=self.xml_header_search_length,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.content_column, self.filename_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_cols = [self.content_column, "representative_filename"]
        if self.detect_language:
            output_cols.extend(["language", "language_detector"])
        if self.basic_stats:
            output_cols.extend(
                [
                    "basic_num_bytes",
                    "basic_valid_utf8",
                    "basic_max_line_length",
                    "basic_num_lines",
                    "basic_average_line_length",
                    "basic_alpha_percent",
                    "basic_alnum_percent",
                    "basic_base64_percent",
                    "basic_hex_percent",
                    "basic_contains_xml_header",
                ]
            )
        if self.opencoder_metrics:
            output_cols.extend(["ors_comment_lines_frac", "ors_comment_chars_frac"])
        if self.tokenize:
            output_cols.append(f"num_tokens_{self.tokenizer_name}")
        return ["data"], output_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        # Create representative_filename from filename_column if needed
        if "representative_filename" not in df.columns:
            df["representative_filename"] = df[self.filename_column].apply(
                lambda x: Path(x).name if x else "unknown.txt"
            )

        result_df = self._annotator.modify_document(df)
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class CodeLanguageDetection(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that detects programming language for code files.

    Uses hyperpolyglot for language detection based on filename extension,
    shebang lines, and content heuristics.

    Args:
        content_column: Column name containing code content. Default "content".
        filename_column: Column name containing filename. Default "rel_path".
    """

    content_column: str = "content"
    filename_column: str = "rel_path"
    name: str = "code_language_detection"

    def __post_init__(self) -> None:
        self._detector = CodeLanguageDetector()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.content_column, self.filename_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.content_column, "representative_filename", "language", "language_detector"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        # Create representative_filename from filename_column if needed
        if "representative_filename" not in df.columns:
            df["representative_filename"] = df[self.filename_column].apply(
                lambda x: Path(x).name if x else "unknown.txt"
            )

        result_df = self._detector.modify_document(df)
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class CodeLicenseDetection(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that detects software licenses in code files.

    Uses scancode-toolkit for license detection and adds license information
    columns to the data.

    Args:
        content_column: Column name containing code content. Default "content".
        detection_timeout: Timeout in seconds for license detection per file.
    """

    content_column: str = "content"
    detection_timeout: int = 100
    name: str = "code_license_detection"

    def __post_init__(self) -> None:
        self._detector = CodeLicenseDetector(
            detection_timeout=self.detection_timeout,
            content_column=self.content_column,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.content_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.content_column, "licenses", "has_license", "license_count"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        result_df = self._detector.modify_document(df)
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class CodeQualityFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that filters code based on quality metrics.

    Applies multiple quality filters based on annotation columns:
    - Comment fraction filter
    - Alpha percent filter
    - Max line length filter
    - Average line length filter
    - Token count filter
    - Hex content filter
    - Base64 content filter
    - Cyclomatic complexity filter (if software metrics available)

    Args:
        min_comment_ratio: Minimum comment line fraction.
        max_comment_ratio: Maximum comment line fraction.
        min_alpha_percent: Minimum alphabetic character percentage.
        max_line_length: Maximum allowed line length.
        min_avg_line_length: Minimum average line length.
        max_avg_line_length: Maximum average line length.
        min_tokens: Minimum token count.
        max_tokens: Maximum token count.
        max_hex_percent: Maximum hex pattern percentage.
        max_base64_percent: Maximum base64 pattern percentage.
        max_complexity: Maximum cyclomatic complexity.
        tokenizer_name: Tokenizer name for token count filter.
        add_filter_columns: Whether to add filter result columns.
    """

    min_comment_ratio: float = 0.00
    max_comment_ratio: float = 0.80
    min_alpha_percent: float = 0.25
    max_line_length: int = 1000
    min_avg_line_length: float = 5.0
    max_avg_line_length: float = 100.0
    min_tokens: int = 10
    max_tokens: int = 100000
    max_hex_percent: float = 0.40
    max_base64_percent: float = 0.40
    max_complexity: float = 50.0
    tokenizer_name: str = "github_o200k_base"
    add_filter_columns: bool = True
    name: str = "code_quality_filter"

    def __post_init__(self) -> None:
        self._filters = {
            "comment_fraction": CommentFractionFilter(self.min_comment_ratio, self.max_comment_ratio),
            "alpha_percent": AlphaPercentFilter(self.min_alpha_percent),
            "max_line_length": MaxLineLengthFilter(self.max_line_length),
            "avg_line_length": AverageLineLengthFilter(self.min_avg_line_length, self.max_avg_line_length),
            "token_count": TokenCountFilter(self.min_tokens, self.max_tokens, self.tokenizer_name),
            "hex_content": HexContentFilter(self.max_hex_percent),
            "base64_content": Base64ContentFilter(self.max_base64_percent),
            "complexity": CyclomaticComplexityFilter(self.max_complexity),
        }
        self._column_map = {
            "comment_fraction": "ors_comment_lines_frac",
            "alpha_percent": "basic_alpha_percent",
            "max_line_length": "basic_max_line_length",
            "avg_line_length": "basic_average_line_length",
            "token_count": f"num_tokens_{self.tokenizer_name}",
            "hex_content": "basic_hex_percent",
            "base64_content": "basic_base64_percent",
            "complexity": "software_metrics_cyclomatic_complexity",
        }

    def inputs(self) -> tuple[list[str], list[str]]:
        # Only require columns that we actually filter on (excluding optional complexity)
        required_cols = [
            "ors_comment_lines_frac",
            "basic_alpha_percent",
            "basic_max_line_length",
            "basic_average_line_length",
            f"num_tokens_{self.tokenizer_name}",
            "basic_hex_percent",
            "basic_base64_percent",
        ]
        return ["data"], required_cols

    def outputs(self) -> tuple[list[str], list[str]]:
        output_cols = list(self.inputs()[1])
        if self.add_filter_columns:
            output_cols.extend([f"filter_{name}" for name in self._filters if name != "complexity"])
            output_cols.append("passes_all_filters")
        return ["data"], output_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        # Apply each filter
        filter_results = {}
        for filter_name, filter_obj in self._filters.items():
            col_name = self._column_map[filter_name]
            if col_name not in df.columns:
                msg = (
                    f"Filter '{filter_name}' requires column '{col_name}' which is missing from the input data. "
                    f"Available columns: {list(df.columns)}"
                )
                raise ValueError(msg)

            if filter_name == "token_count":
                scores = df[col_name].apply(lambda x, f=filter_obj, c=col_name: f.score_document(**{c: x}))
            else:
                scores = df[col_name].apply(filter_obj.score_document)
            filter_results[f"filter_{filter_name}"] = scores.apply(filter_obj.keep_document)

        # Compute combined result
        if filter_results:
            filter_df = pd.DataFrame(filter_results)
            passes_all = filter_df.all(axis=1)

            if self.add_filter_columns:
                for col, values in filter_results.items():
                    df[col] = values
                df["passes_all_filters"] = passes_all

            # Filter the dataframe
            df = df[passes_all]

        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
