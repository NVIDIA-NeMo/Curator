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
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask, _EmptyTask
from nemo_curator.utils.column_utils import resolve_filename_column

from .download import DocumentDownloader, DocumentDownloadStage
from .extract import DocumentExtractor
from .iterator import DocumentIterator
from .url_generation import URLGenerationStage, URLGenerator


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
        self.name = (
            f"iterate_extract_{self.iterator.__class__.__name__.lower()}_{self.extractor.__class__.__name__.lower()}"
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects FileGroupTask with local file paths."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces DocumentBatch with processed records."""
        if self.extractor:
            return (["data"], self.extractor.output_columns() + ([self.filename_col] if self.add_filename_column else []))
        else:
            return (["data"], self.iterator.output_columns() + ([self.filename_col] if self.add_filename_column else []))

    def process(self, task: FileGroupTask) -> DocumentBatch:  # noqa: C901
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
                    if self.extractor:
                        extracted = self.extractor.extract(record_dict)
                    else:
                        extracted = record_dict

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

        df = pd.DataFrame(records)

        # Return
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata={
                **task._metadata,
            },
            _stage_perf=task._stage_perf,
        )


@dataclass
class DocumentDownloadExtractStage(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage that combines URL generation, download, and iterate-extract stages.

    This supports the full 3-step pipeline pattern like Common Crawl:
    1. Generate URLs from minimal input
    2. Download files from URLs
    3. Iterate through files to extract structured content

    """

    url_generator: URLGenerator
    downloader: DocumentDownloader
    iterator: DocumentIterator
    extractor: DocumentExtractor | None = None
    url_limit: int | None = None
    record_limit: int | None = None
    add_filename_column: bool | str = True

    def __post_init__(self):
        """Initialize the constituent stages."""
        # URL generation stage
        url_stage = URLGenerationStage(
            url_generator=self.url_generator,
            limit=self.url_limit,
        )

        # Download stage
        download_stage = DocumentDownloadStage(
            downloader=self.downloader,
        )

        # Iterate-extract stage
        iterate_extract_stage = DocumentIterateExtractStage(
            iterator=self.iterator,
            extractor=self.extractor,
            record_limit=self.record_limit,
            add_filename_column=self.add_filename_column,
        )

        stages = [url_stage, download_stage, iterate_extract_stage]
        self.stages = stages

        url_generator_name = self.url_generator.__class__.__name__.lower()
        downloader_name = self.downloader.__class__.__name__.lower()
        self.name = f"document_download_extract_{url_generator_name}_{downloader_name}_composite"
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get description of this composite stage."""
        return f"URL-Download-Iterate-Extract pipeline using {self.url_generator.__class__.__name__} and {self.downloader.__class__.__name__}"
