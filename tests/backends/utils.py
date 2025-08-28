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

"""Test utilities for backend integration tests.

This module provides shared utilities for creating test data, pipelines,
and validating expected outputs across different backend implementations.
"""

import io
import json
import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import ray
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import DocumentBatch


@ray.remote(num_cpus=0.1)
class StageCallCounter:
    """Ray actor to count how many times each stage is called."""

    def __init__(self, output_dir: Path):
        self.counters = Counter()
        self.output_dir = output_dir  # Store output_dir as instance variable

    def increment(self, stage_name: str) -> int:
        """Increment the counter for a stage and return the new count."""
        self.counters[stage_name] += 1
        # dump the counters to a file (so that even after the actor is killed, the counters are persisted)
        with open(self.output_dir / "call_counters.json", "w") as f:
            json.dump(self.counters, f)
        return self.counters[stage_name]

    def get_count(self, stage_name: str) -> int:
        """Get the current count for a stage."""
        return self.counters.get(stage_name, 0)

    def get_all_counts(self) -> dict[str, int]:
        """Get all stage counts."""
        return self.counters.copy()


# Constants for test configuration
TOTAL_DOCUMENTS = 100
EXPECTED_NUM_STAGES = (
    6  # JsonlReader -> AddLengthStage -> SplitIntoRowsStage -> AddLengthStage -> StageWithSetup -> JsonlWriter
)
FILES_PER_PARTITION = 2


def create_test_data(output_dir: Path, num_files: int) -> None:
    """Create test JSONL files for integration testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

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

    def __init__(self, column_name: str = "doc_length"):
        self._name = "add_length"
        self.column_name = column_name

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """Process a batch of tasks and add length field."""
        import time

        # Get the counter actor by name and record timing for the actor+increment call
        t_actor0 = time.perf_counter()
        counter_actor = ray.get_actor("stage_call_counter", namespace="stage_call_counter")
        stage_identifier = f"{self._name}_{self.column_name}"
        ray.get(counter_actor.increment.remote(stage_identifier))
        t_actor1 = time.perf_counter()

        # Compute len(...) timing across this batch
        t_compute0 = time.perf_counter()
        results = []
        for input_data in tasks:
            df = input_data.to_pandas()
            df[self.column_name] = df["text"].apply(len)
            results.append(
                DocumentBatch(
                    task_id=input_data.task_id,
                    dataset_name=input_data.dataset_name,
                    data=df,
                    _metadata=input_data._metadata,
                    _stage_perf=input_data._stage_perf,
                )
            )
        t_compute1 = time.perf_counter()
        # Record custom timing metrics for this stage batch
        self._log_metrics(
            {
                "counter_actor_increment_s": t_actor1 - t_actor0,
                "compute_len_s": t_compute1 - t_compute0,
            }
        )
        return results

    def process(self, input_data: DocumentBatch) -> DocumentBatch:
        """Dummy process method - we use process_batch instead."""
        msg = f"Stage '{self._name}' should use process_batch, not process"
        raise NotImplementedError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text", self.column_name]

    def ray_stage_spec(self) -> dict[str, bool]:
        return {
            RayStageSpecKeys.IS_ACTOR_STAGE: True,
        }


class SplitIntoRowsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split the document into rows."""

    _name = "split_into_rows"

    def process(self, input_data: DocumentBatch) -> list[DocumentBatch]:
        import time

        t0 = time.perf_counter()
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
        # Record custom timing for row splitting
        t1 = time.perf_counter()
        self._log_metric("split_into_rows_time_s", t1 - t0)
        return tasks

    def ray_stage_spec(self) -> dict[str, bool]:
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
            RayStageSpecKeys.IS_ACTOR_STAGE: True,
        }

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []


class StageWithSetup(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Setup stage that adds a numeric field to the document."""

    def __init__(self, temp_file_path: Path):
        self.TEMP_FILE_PATH = temp_file_path

    _name = "stage_with_setup"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["node_id", "random_string"]

    def setup_on_node(self, node_info: NodeInfo, _: WorkerMetadata) -> None:
        with open(self.TEMP_FILE_PATH, "w") as f:
            f.write(f"{node_info.node_id},random_str")

    def setup(self, _: WorkerMetadata) -> None:
        with open(self.TEMP_FILE_PATH) as f:
            self.node_id, self.random_str = f.read().split(",")

    def process(self, input_data: DocumentBatch) -> DocumentBatch:
        df = input_data.to_pandas()
        df["node_id"] = self.node_id
        df["random_string"] = self.random_str
        return DocumentBatch(
            task_id=input_data.task_id,
            dataset_name=input_data.dataset_name,
            data=df,
            _metadata=input_data._metadata,
            _stage_perf=input_data._stage_perf,
        )


def create_test_pipeline(input_dir: Path, output_dir: Path) -> tuple[Pipeline, Any]:
    """Create a test pipeline for integration testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a named counter actor that can be referenced by name
    # we use detached lifetime so that the actor is not killed until the end of the test
    ray.init(ignore_reinit_error=True)
    StageCallCounter.options(name="stage_call_counter", namespace="stage_call_counter", lifetime="detached").remote(
        output_dir
    )
    ray.shutdown()

    pipeline = Pipeline(
        name="integration_test_pipeline", description="Integration test pipeline for backend comparison"
    )

    # Add JsonlReader stage
    pipeline.add_stage(
        JsonlReader(
            file_paths=str(input_dir),
            files_per_partition=FILES_PER_PARTITION,
        )
    )

    pipeline.add_stage(AddLengthStage("doc_length_1"))

    # Add SplitIntoRowsStage stage
    pipeline.add_stage(SplitIntoRowsStage())

    # Add AddLengthStage stage
    pipeline.add_stage(AddLengthStage("doc_length_2"))

    # Add StageWithSetup stage
    pipeline.add_stage(StageWithSetup(input_dir / "temp_file.txt"))

    # Add JsonlWriter stage
    pipeline.add_stage(JsonlWriter(path=str(output_dir)))

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
