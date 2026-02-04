# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.column_utils import resolve_filename_column

from .extract import DocumentExtractor


class DocumentIterator(ABC):
    """Abstract base class for document iterators.

    Always yields dict[str, str] records. For raw content that needs extraction,
    the iterator can put it in any field (e.g., "raw_content", "html", "content", etc.)
    """

    @abstractmethod
    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Iterate over records in a file, yielding dict records."""
        ...

    @abstractmethod
    def output_columns(self) -> list[str]:
        """Define output columns - produces DocumentBatch with records."""
        ...


@dataclass
class DocumentIterateExtractStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Stage that iterates through downloaded files with DocumentIterator,
    then extracts structured content from raw records with DocumentExtractor.

    Takes local file paths and produces a DocumentBatch with extracted content.
    If DocumentIterator produces the final format, then DocumentExtractor is not needed.
    """

    iterator: DocumentIterator
    extractor: DocumentExtractor | None = None
    record_limit: int | None = None
    add_filename_column: bool | str = True

    def __post_init__(self):
        """Initialize the stage."""
        self.filename_col = resolve_filename_column(self.add_filename_column)
        if self.extractor:
            self.name = f"iterate_extract_{self.iterator.__class__.__name__.lower()}_{self.extractor.__class__.__name__.lower()}"
        else:
            self.name = f"iterate_{self.iterator.__class__.__name__.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects FileGroupTask with local file paths."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces DocumentBatch with processed records."""
        if self.extractor:
            return (["data"], self.extractor.output_columns() + ([self.filename_col] if self.add_filename_column else []))
        else:
            return (["data"], self.iterator.output_columns() + ([self.filename_col] if self.add_filename_column else []))

    def process(self, task: FileGroupTask) -> DocumentBatch:
        """Iterate through files and extract structured content.

        Args:
            task (FileGroupTask): Task containing local file paths

        Returns:
            DocumentBatch: Batch containing extracted records
        """
        records = []

        for file_path in task.data:
            try:
                record_count = 0
                iterator_result = self.iterator.iterate(file_path)

                if iterator_result is None:
                    continue

                for record_dict in iterator_result:
                    if self.record_limit and record_count >= self.record_limit:
                        break

                    # Add filename early
                    if self.add_filename_column:
                        record_dict[self.filename_col] = os.path.basename(file_path)

                    # Extract structured content
                    extracted = self.extractor.extract(record_dict) if self.extractor else record_dict

                    if extracted is None:
                        continue

                    # Ensure filename is preserved
                    if self.add_filename_column:
                        extracted[self.filename_col] = record_dict[self.filename_col]

                    records.append(extracted)
                    record_count += 1

            except Exception as e:  # noqa: BLE001
                logger.error(f"Error iterating {file_path}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(records)

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata={
                **task._metadata,
            },
            _stage_perf=task._stage_perf,
        )
