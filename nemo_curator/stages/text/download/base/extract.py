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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.column_utils import resolve_filename_column


class DocumentExtractor(ABC):
    """Abstract base class for document extractors.

    Takes a record dict and returns processed record dict or None to skip.
    Can transform any fields in the input dict.
    """

    @abstractmethod
    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        """Extract/transform a record dict into final record dict."""
        ...

    @abstractmethod
    def input_columns(self) -> list[str]:
        """Define input columns - produces DocumentBatch with records."""
        ...

    @abstractmethod
    def output_columns(self) -> list[str]:
        """Define output columns - produces DocumentBatch with records."""
        ...


@dataclass
class DocumentExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that extracts structured content from raw records.

    Takes DocumentBatch with raw content and produces DocumentBatch with extracted content.
    This is for cases where iteration and extraction are separate steps.
    """

    extractor: DocumentExtractor
    add_filename_column: bool | str = True

    def __post_init__(self):
        """Initialize the stage."""
        self.filename_col = resolve_filename_column(self.add_filename_column)
        self._name = f"extract_{self.extractor.__class__.__name__.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects DocumentBatch with dict records."""
        return (["data"], self.extractor.input_columns() + ([self.filename_col] if self.add_filename_column else []))  # type: ignore[reportReturnType]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces DocumentBatch with processed records."""
        return (["data"], self.extractor.output_columns() + ([self.filename_col] if self.add_filename_column else []))  # type: ignore[reportReturnType]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Extract structured content from raw records.

        Args:
            task (DocumentBatch): Batch containing records

        Returns:
            DocumentBatch: Batch containing extracted records
        """
        extracted_records = []

        for _, row in task.data.iterrows():
            # Convert pandas Series to dict
            record_dict = row.to_dict()

            # Extract structured content
            extracted = self.extractor.extract(record_dict)
            if extracted is not None:
                if self.add_filename_column:
                    if self.filename_col in extracted:
                        msg = f"Since add_filename_col is specified, we'll overwrite ({self.filename_col}) from the input data."
                        logger.warning(msg)

                    extracted[self.filename_col] = record_dict[self.filename_col]  # type: ignore[reportReturnType]
                extracted_records.append(extracted)

        # Convert to DataFrame
        df = pd.DataFrame(extracted_records)

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata={
                **task._metadata,
            },
            _stage_perf=task._stage_perf,
        )
