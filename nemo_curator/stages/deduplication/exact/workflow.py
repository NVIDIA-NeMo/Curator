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

import os
import time
from typing import Any, Literal

from loguru import logger

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.exact.identification import ExactDuplicateIdentification
from nemo_curator.stages.deduplication.id_generator import (
    create_id_generator_actor,
    kill_id_generator_actor,
    write_id_generator_to_disk,
)
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import FileGroupTask

ID_GENERATOR_OUTPUT_FILENAME = "exact_id_generator.json"


class ExactDeduplicationWorkflow:
    """
    A pipeline that performs exact deduplication of a dataset.
    It consists of the following stages:
    - FilePartitioningStage
        Groups input files into smaller groups that can be processed in parallel.
    - ExactDuplicateIdentification
        Finds exact duplicates in a given column by hashing the column.
    - Removal (Optional)
        Currently not implemented.
    """

    def __init__(  # noqa: PLR0913
        self,
        # I/O config
        output_path: str,
        input_path: str | list[str] | None = None,
        input_filetype: Literal["jsonl", "parquet"] = "parquet",
        input_blocksize: str | int = "2GiB",
        input_file_extensions: list[str] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        # Deduplication config
        assign_id: bool = True,
        id_field: str | None = None,
        text_field: str = "text",
        perform_removal: bool = False,
        env_vars: dict[str, Any] | None = None,
    ):
        """
        Configuration for exact duplicates detection.
        Parameters
        output_path: str
            Directory to store the duplicate Ids and the id generator mapping for removal pipelines.
            It also stores the deduplicated output files, if `perform_removal` is True.
        input_path: str | list[str] | None
            Directory or list of files containing the input dataset.
            Unused if `initial_tasks` is provided during workflow run.
        input_filetype: Literal["jsonl", "parquet"]
            Format of the input dataset.
        input_blocksize: str | int
            Size of the input blocks to read in.
            If an integer is provided, it will be interpreted as bytes.
            If a string is provided, it will be interpreted as a size with a unit.
            If not provided, the default blocksize of 1GiB will be used.
        input_file_extensions: list[str] | None
            File extensions of the input dataset.
            If not provided, the default extensions for the input_filetype will be used.
            If provided, this will override the default extensions for the input_filetype.
        read_kwargs: dict[str, Any] | None = None
            Additional keyword arguments to pass for reading the input files.
            This could include the storage_options dictionary when reading from remote storage.
        write_kwargs: dict[str, Any] | None = None
            Additional keyword arguments to pass for deduplicated results written to output_dir.
            This could include the storage_options dictionary when writing to remote storage.
        assign_id: bool
            Whether to automatically assign a unique id to each document.
        id_field: str | None
            Existing id field name if not automatically assigning a new id.
        text_field: str
            Field containing the text to deduplicate.
        perform_removal: bool
            Whether to remove the duplicates from the original dataset.
        env_vars: dict[str, Any] | None = None
            Environment variables to pass to the pipeline.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.input_filetype = input_filetype
        self.input_blocksize = input_blocksize
        self.input_file_extensions = input_file_extensions
        self.read_kwargs = read_kwargs
        self.write_kwargs = write_kwargs

        self.text_field = text_field
        self.assign_id = assign_id
        self.id_field = id_field
        self.perform_removal = perform_removal

        self.env_vars = env_vars

        self.executor_config = {"runtime_env": {"env_vars": env_vars}} if env_vars is not None else None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.perform_removal:
            msg = "Removal is not implemented yet"
            raise NotImplementedError(msg)

    def _create_input_filegroups(self) -> Pipeline:
        return Pipeline(
            name="input_filegroups_pipeline",
            stages=[
                FilePartitioningStage(
                    file_paths=self.input_path,
                    file_extensions=self.input_file_extensions,
                    blocksize=self.input_blocksize,
                    storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs is not None else None,
                ),
            ],
        )

    def _create_identification_pipeline(self, num_input_tasks: int) -> Pipeline:
        return Pipeline(
            name="exact_deduplication_pipeline",
            stages=[
                ExactDuplicateIdentification(
                    output_path=self.output_path,
                    text_field=self.text_field,
                    input_filetype=self.input_filetype,
                    read_kwargs=self.read_kwargs,
                    write_kwargs=self.write_kwargs,
                    assign_id=self.assign_id,
                    id_field=self.id_field,
                    # Matches previous implementation to write out to 1/3 the number of input tasks
                    total_nparts=max(1, num_input_tasks // 3),
                ),
            ],
        )

    def _validate_initial_tasks(self, initial_tasks: list[FileGroupTask] | None) -> None:
        if initial_tasks is not None:
            if any(not isinstance(task, FileGroupTask) for task in initial_tasks):
                msg = "All input tasks to the pipeline must be of type FileGroupTask pointing to the dataset to be deduplicated."
                raise ValueError(msg)
            elif self.input_path is not None:
                logger.warning("Ignoring input_path as initial_tasks are provided.")
        elif self.input_path is None:
            msg = "input_path to the dataset must be provided if initial_tasks are not provided manually."
            raise ValueError(msg)

    def run(self, initial_tasks: list[FileGroupTask] | None = None) -> None:
        """Run the deduplication pipeline.

        Args:
            initial_tasks:
            Set of FileGroupTasks generated by a previous stage pointing to the dataset to be deduplicated.
            If not provided, the pipeline will generate the input tasks based on the input_dir and input_file_extensions.
        """
        self._validate_initial_tasks(initial_tasks)
        executor = RayActorPoolExecutor(config=self.executor_config)
        if self.assign_id:
            try:
                create_id_generator_actor()
            except ValueError:
                err_msg = """
                An existing id generator actor was found. Please remove or save the existing id generator with
                `nemo_curator.stages.deduplication.id_generator.write_id_generator_to_disk` (if needed) and remove the actor with
                `nemo_curator.stages.deduplication.id_generator.kill_id_generator_actor` before running the exact deduplication pipeline.
                """
                raise RuntimeError(err_msg) from None

        try:
            start_time = time.time()
            if initial_tasks is None:
                input_filegroups_pipeline = self._create_input_filegroups()
                initial_tasks = input_filegroups_pipeline.run(executor=executor, initial_tasks=initial_tasks)
                initial_filegroups_end_time = time.time()
                logger.info(
                    f"Created input tasks from {self.input_path} in {(initial_filegroups_end_time - start_time):.2f} seconds"
                )

            identification_pipeline = self._create_identification_pipeline(num_input_tasks=len(initial_tasks))
            identification_start_time = time.time()
            removal_id_tasks = identification_pipeline.run(executor=executor, initial_tasks=initial_tasks)
            identification_end_time = time.time()
            logger.info(
                f"Exact duplicate identification pipeline completed in {(identification_end_time - identification_start_time):.2f} seconds"
            )

            num_duplicates = sum(task._metadata.get("num_removal_ids", 0) for task in removal_id_tasks)
            if num_duplicates == 0:
                logger.info("No exact duplicates found in the dataset.")

            if self.assign_id:
                id_generator_path = os.path.join(self.output_path, ID_GENERATOR_OUTPUT_FILENAME)
                write_id_generator_to_disk(
                    id_generator_path,
                    storage_options=self.write_kwargs.get("storage_options")
                    if self.write_kwargs is not None
                    else None,
                )
                logger.info(f"Id generator written to {id_generator_path}")
            end_time = time.time()
            logger.info(f"Exact deduplication pipeline completed in {(end_time - start_time):.2f} seconds")
        finally:
            if self.assign_id:
                kill_id_generator_actor()
