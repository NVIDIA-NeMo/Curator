import os
import time
from typing import TYPE_CHECKING, Any, Optional

import fsspec
from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import FileGroupTask
from ray_curator.utils.file_utils import get_fs

if TYPE_CHECKING:
    from ray_curator.backends.base import WorkerMetadata


class IdentifySemanticDuplicatesStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stage that identifies duplicates based on cosine similarity threshold."""

    def __init__(
        self,
        threshold: float,
        output_path: str,
        verbose: bool = False,
        input_storage_options: dict[str, Any] | None = None,
        output_storage_options: dict[str, Any] | None = None,
    ):
        """Initialize the identify duplicates stage.

        Args:
            threshold: Cosine similarity threshold above which items are considered duplicates
            output_path: The path to the output directory
            verbose: Whether to print verbose output
            input_storage_options: Storage options for reading input files
            output_storage_options: Storage options for writing output files
        """
        self.threshold = threshold
        self.output_path = output_path
        self.verbose = verbose
        self.input_storage_options = input_storage_options
        self.output_storage_options = output_storage_options
        self.fs : None | fsspec.AbstractFileSystem = None
        self._name = "IdentifySemanticDuplicatesStage"

    def setup(self, _: Optional["WorkerMetadata"] = None) -> None:
        """Setup the stage on a node."""
        self.fs = get_fs(self.output_path, storage_options=self.output_storage_options)
        self.fs.makedirs(self.output_path, exist_ok=True)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process a FileGroupTask to identify duplicates based on threshold."""
        if not task.data:
            return FileGroupTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
                data=[],
            )

        t1 = time.perf_counter()

        # Read the file directly
        file_path = task.data[0]  # Expect single file
        logger.info(f"Reading {file_path} with resources {self.resources}")
        backend = "cudf" if self.resources.gpus > 0 else "pandas"

        if backend == "cudf":
            import cudf

            df = cudf.read_parquet(file_path, **self.input_storage_options or {})
        else:
            import pandas as pd

            df = pd.read_parquet(file_path, **self.input_storage_options or {})

        t2 = time.perf_counter()
        if self.verbose:
            logger.debug(f"Read {len(df)} rows using {backend} in {(t2 - t1):.2f} seconds")

        # Filter based on threshold
        filtered_df = df[df["cosine_sim_score"] >= self.threshold]

        t3 = time.perf_counter()
        if self.verbose:
            logger.info(
                f"Filtered to {len(filtered_df)} duplicates (threshold: {self.threshold}) in {(t3 - t2):.2f} seconds"
            )

        # If no duplicates found, return empty task
        if len(filtered_df) == 0:
            logger.info(f"No duplicates found above threshold {self.threshold}")
            return FileGroupTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
                data=[],
            )

        # Write results
        output_filename = f"duplicates_threshold_{self.threshold}_{task.task_id}.parquet"
        output_filepath = os.path.join(self.output_path, output_filename)

        # Write using the same backend

        filtered_df.to_parquet(output_filepath, index=False, **self.output_storage_options or {})

        t4 = time.perf_counter()
        if self.verbose:
            logger.info(f"Write completed in {(t4 - t3):.2f} seconds")

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata={
                **task._metadata,
            },
            _stage_perf=task._stage_perf,
            data=[output_filepath],
        )
