"""Test utilities for backend integration tests.

This module provides shared utilities for creating test data, pipelines,
and validating expected outputs across different backend implementations.
"""

import io
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from loguru import logger

from ray_curator.pipeline import Pipeline
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch

# Constants for test configuration
TOTAL_DOCUMENTS = 10
EXPECTED_NUM_STAGES = 4  # JsonlReader -> AddLengthStage -> SplitIntoRowsStage -> JsonlWriter


def create_test_data(output_dir: Path, num_files: int) -> None:
    """Create test JSONL files for integration testing."""
    output_dir.mkdir(exist_ok=True)

    sample_documents = [{"id": f"doc_{i}", "text": f"Test document {i}"} for i in range(TOTAL_DOCUMENTS)]

    docs_per_file = len(sample_documents) // num_files

    for file_idx in range(num_files):
        file_path = output_dir / f"test_data_{file_idx}.jsonl"

        with open(file_path, "w") as f:
            start_idx = file_idx * docs_per_file
            end_idx = start_idx + docs_per_file if file_idx < num_files - 1 else len(sample_documents)

            for doc_idx in range(start_idx, end_idx):
                if doc_idx < len(sample_documents):
                    doc = sample_documents[doc_idx]
                    f.write(json.dumps(doc) + "\n")


class AddLengthStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Add a length field to the document."""

    def process(self, input_data: DocumentBatch) -> DocumentBatch:
        df = input_data.to_pandas()
        df["doc_length"] = df["text"].apply(len)
        return DocumentBatch(
            task_id=input_data.task_id,
            dataset_name=input_data.dataset_name,
            data=df,
            _metadata=input_data._metadata,
            _stage_perf=input_data._stage_perf,
        )

    @property
    def name(self) -> str:
        return "add_length"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text", "doc_length"]


class SplitIntoRowsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split the document into rows."""

    @property
    def resources(self) -> Resources:
        return Resources(cpus=0.5)

    def process(self, input_data: DocumentBatch) -> list[DocumentBatch]:
        df = input_data.to_pandas()
        # Remove source_files from metadata to prevent file collision issues
        # When splitting a document into individual rows, each row would inherit
        # the same source_files metadata. This causes the JsonlWriter to hash
        # all rows to the same output file, resulting in overwrites instead of
        # separate files per row. By removing source_files, we allow the writer
        # to create unique output files for each row based on other metadata.
        input_metadata_without_source_files = input_data._metadata.copy()
        input_metadata_without_source_files.pop("source_files", None)

        # Create a list of tasks, each with a unique task_id and metadata
        tasks = []
        for _, row in df.iterrows():
            # Create a single-row DataFrame instead of a list of dicts
            row_df = pd.DataFrame([row.to_dict()])
            tasks.append(
                DocumentBatch(
                    task_id=f"{input_data.task_id}_row_{row['id']}",
                    dataset_name=input_data.dataset_name,
                    data=row_df,
                    _metadata=input_metadata_without_source_files,
                    _stage_perf=input_data._stage_perf.copy(),
                )
            )
        return tasks

    @property
    def name(self) -> str:
        return "split_into_rows"

    @property
    def ray_stage_spec(self) -> dict[str, bool]:
        return {
            "is_fanout_stage": True,
        }

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []


def create_test_pipeline(input_dir: Path, output_dir: Path) -> Pipeline:
    """Create a test pipeline for integration testing."""
    pipeline = Pipeline(
        name="integration_test_pipeline", description="Integration test pipeline for backend comparison"
    )

    # Add JsonlReader stage
    pipeline.add_stage(
        JsonlReader(
            file_paths=str(input_dir),
            files_per_partition=2,  # Create multiple tasks for better testing
            reader="pandas",
        )
    )

    # Add AddLengthStage stage
    pipeline.add_stage(AddLengthStage())

    # Add SplitIntoRowsStage stage
    pipeline.add_stage(SplitIntoRowsStage())

    # Add JsonlWriter stage
    pipeline.add_stage(JsonlWriter(output_dir=str(output_dir)))

    return pipeline


@contextmanager
def capture_logs() -> Iterator[io.StringIO]:
    """Context manager to capture both Ray Data and loguru logs.
    We don't use pytest's caplog fixture because it doesn't capture Ray Data logs.
    """
    ray_data_loggers = ["ray.data"]

    # Create a string buffer to capture all logs
    log_buffer = io.StringIO()
    log_handler = logging.StreamHandler(log_buffer)
    log_handler.setLevel(logging.INFO)

    # Store original handlers and levels for cleanup
    original_handlers = []
    original_levels = []

    # Add handler to all relevant Ray Data loggers
    for logger_name in ray_data_loggers:
        logger_obj = logging.getLogger(logger_name)
        original_handlers.append(logger_obj.handlers.copy())
        original_levels.append(logger_obj.level)
        logger_obj.setLevel(logging.INFO)
        logger_obj.addHandler(log_handler)

    # Add loguru handler to capture loguru logs
    loguru_handler_id = logger.add(
        log_buffer,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level="INFO",
        enqueue=False,  # Set to 'True' if spawning child processes
    )

    try:
        yield log_buffer
    finally:
        # Clean up Ray Data handlers and restore original levels
        for i, logger_name in enumerate(ray_data_loggers):
            logger_obj = logging.getLogger(logger_name)
            logger_obj.removeHandler(log_handler)
            logger_obj.level = original_levels[i]

        # Clean up loguru handler
        logger.remove(loguru_handler_id)
