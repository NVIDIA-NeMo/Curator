import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch, FileGroupTask
from ray_curator.utils.column_utils import resolve_filename_column


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
class DocumentIterateStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Stage that iterates through downloaded files and extracts records.

    Takes local file paths and produces a DocumentBatch with records.
    All iterators yield dict[str, str] records uniformly.
    """

    iterator: DocumentIterator
    record_limit: int | None = None
    add_filename_column: bool | str = True

    def __post_init__(self):
        """Initialize the stage."""
        self.filename_col = resolve_filename_column(self.add_filename_column)

    @property
    def name(self) -> str:
        """Return stage name."""
        iterator_name = self.iterator.__class__.__name__
        return f"iterate_{iterator_name.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects FileGroupTask with local file paths."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces DocumentBatch with records."""
        return (["data"], self.iterator.output_columns() + ([self.filename_col] if self.add_filename_column else []))

    def process(self, task: FileGroupTask) -> DocumentBatch:
        """Iterate through files and extract records.

        Args:
            task (FileGroupTask): Task containing local file paths

        Returns:
            DocumentBatch: Batch containing records
        """
        records = []

        for file_path in task.data:
            try:
                record_count = 0
                iterator_result = self.iterator.iterate(file_path)
                if iterator_result is not None:
                    for record_dict in iterator_result:
                        if self.record_limit and record_count >= self.record_limit:
                            break
                        if self.add_filename_column:
                            # TODO: Support cloud storage https://github.com/NVIDIA-NeMo/Curator/issues/779
                            record_dict[self.filename_col] = os.path.basename(file_path)
                        records.append(record_dict)
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
