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

import time
from typing import Any, Literal

from loguru import logger

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.fuzzy.buckets_to_edges import BucketsToEdgesStage
from nemo_curator.stages.deduplication.fuzzy.connected_components import ConnectedComponentsStage
from nemo_curator.stages.deduplication.fuzzy.identify_duplicates import IdentifyDuplicatesStage
from nemo_curator.stages.deduplication.fuzzy.lsh.stage import LSHStage
from nemo_curator.stages.deduplication.fuzzy.minhash import MinHashStage
from nemo_curator.stages.deduplication.id_generator import (
    create_id_generator_actor,
    kill_id_generator_actor,
    write_id_generator_to_disk,
)
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_fs

ID_GENERATOR_OUTPUT_FILENAME = "fuzzy_id_generator.json"


class FuzzyDeduplicationWorkflow:
    """
    A pipeline that performs fuzzy deduplication of a dataset.
    It consists of the following stages:
    - FilePartitioningStage
        Groups input files into smaller groups that can be processed in parallel.
    - MinHashStage
        Computes minhashes for the input dataset.
    - LSHStage
        Performs Locality Sensitive Hashing on the minhashes.
        This is a shuffle stage that involves moving data between workers.
    - BucketsToEdgesStage
        This stage converts the resulting LSH mapping of bucket ID to document ID into a graph of edges.
    - ConnectedComponentsStage
        Performs weaklyconnected components clustering on the graph represented by the edgelist.
    - IdentifyDuplicatesStage
        Generates a list of document ids to remove based on the connected components clusters/components.
    - Removal (Optional)
        Currently not implemented.
    """

    def __init__(  # noqa: PLR0913
        self,
        # I/O config
        cache_path: str,
        output_path: str,
        input_path: str | list[str] | None = None,
        input_filetype: Literal["jsonl", "parquet"] = "parquet",
        input_blocksize: str | int = "1GiB",
        input_file_extensions: list[str] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        cache_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        text_field: str = "text",
        perform_removal: bool = False,
        # Minhash + LSH Config
        seed: int = 42,
        char_ngrams: int = 24,
        num_bands: int = 20,
        minhashes_per_band: int = 13,
        use_64_bit_hash: bool = False,
        bands_per_iteration: int = 5,
        env_vars: dict[str, Any] | None = None,
    ):
        """
        Configuration for MinHash based fuzzy duplicates detection.
        Parameters
        cache_path: str
            Directory to store deduplication intermediates such as minhashes/buckets etc.
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
        cache_kwargs: dict[str, Any] | None = None
            Additional keyword arguments to pass for intermediate files written to cache_dir.
            This could include the storage_options dictionary when writing to remote storage.
        write_kwargs: dict[str, Any] | None = None
            Additional keyword arguments to pass for deduplicated results written to output_dir.
            This could include the storage_options dictionary when writing to remote storage.

        text_field: str
            Field containing the text to deduplicate.
        perform_removal: bool
            Whether to remove the duplicates from the original dataset.

        seed: int
            Seed for minhash permutations
        char_ngrams: int
            Size of Char ngram shingles used in minhash computation
        num_buckets: int
            Number of Bands or buckets to use during Locality Sensitive Hashing
        hashes_per_bucket: int
            Number of hashes per bucket/band.
        use_64_bit_hash: bool
            Whether to use a 32bit or 64bit hash function for minhashing.
        bands_per_iteration: int
            Number of bands/buckets to shuffle concurrently.
            Larger values process larger batches by processing multiple bands
            but might lead to memory pressures and related errors.

        env_vars: dict[str, Any] | None = None
            Environment variables to pass to the pipeline.
        """
        self.input_path = input_path
        self.cache_path = cache_path
        self.output_path = output_path
        self.input_filetype = input_filetype
        self.input_blocksize = input_blocksize
        self.input_file_extensions = input_file_extensions
        self.read_kwargs = read_kwargs
        self.cache_kwargs = cache_kwargs
        self.write_kwargs = write_kwargs

        self.text_field = text_field
        self.perform_removal = perform_removal

        self.seed = seed
        self.char_ngrams = char_ngrams
        self.num_bands = num_bands
        self.minhashes_per_band = minhashes_per_band
        self.use_64_bit_hash = use_64_bit_hash
        self.bands_per_iteration = bands_per_iteration

        self.env_vars = env_vars

        self.num_hashes = self.num_bands * self.minhashes_per_band
        self.executor_config = {"runtime_env": {"env_vars": env_vars}} if env_vars is not None else None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.char_ngrams < 20:  # noqa: PLR2004
            logger.warning(
                "Using a small char_ngrams value might lead to a large number (~5%) of false positives during deduplication."
                " Using a value of at least 20 for char_ngrams is recommended.",
            )
        if self.perform_removal:
            msg = "Removal is not implemented yet"
            raise NotImplementedError(msg)
        if self.bands_per_iteration < 1 or self.bands_per_iteration > self.num_bands:
            msg = "bands_per_iteration must be between [1, num_bands]"
            raise ValueError(msg)

    def _create_minhash_pipeline(self, generate_input_filegroups: bool) -> Pipeline:
        stages = []
        if generate_input_filegroups:
            stages.append(
                FilePartitioningStage(
                    file_paths=self.input_path,
                    file_extensions=self.input_file_extensions,
                    blocksize=self.input_blocksize,
                    storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs is not None else None,
                ),
            )
        stages.append(
            MinHashStage(
                output_path=self.cache_path,
                text_field=self.text_field,
                char_ngrams=self.char_ngrams,
                num_hashes=self.num_hashes,
                seed=self.seed,
                use_64bit_hash=self.use_64_bit_hash,
                read_format=self.input_filetype,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.cache_kwargs,
            ),
        )
        return Pipeline(
            name="minhash_pipeline",
            stages=stages,
        )

    def _create_lsh_pipeline(self) -> Pipeline:
        cache_dir_fs = get_fs(self.cache_path, self.cache_kwargs)
        return Pipeline(
            name="lsh_duplicate_identification_pipeline",
            stages=[
                FilePartitioningStage(
                    file_paths=cache_dir_fs.sep.join([self.cache_path, "MinHashStage"]),
                    file_extensions=[".parquet"],
                    blocksize="2GiB",
                    storage_options=self.cache_kwargs.get("storage_options")
                    if self.cache_kwargs is not None
                    else None,
                ),
                LSHStage(
                    num_bands=self.num_bands,
                    minhashes_per_band=self.minhashes_per_band,
                    output_path=self.cache_path,
                    # Reading minhashes from cache_path
                    read_kwargs=self.cache_kwargs,
                    write_kwargs=self.cache_kwargs,
                    bands_per_iteration=self.bands_per_iteration,
                    rmm_pool_size="auto",
                    spill_memory_limit="auto",
                ),
            ],
        )

    def _create_connected_components_pipeline(self) -> Pipeline:
        return Pipeline(
            name="connected_components_pipeline",
            stages=[
                BucketsToEdgesStage(
                    output_path=self.cache_path,
                    read_kwargs=self.cache_kwargs,
                    write_kwargs=self.cache_kwargs,
                ),
                ConnectedComponentsStage(
                    output_path=self.cache_path,
                    read_kwargs=self.cache_kwargs,
                    write_kwargs=self.cache_kwargs,
                ),
                IdentifyDuplicatesStage(
                    output_path=self.output_path,
                    read_kwargs=self.cache_kwargs,
                    write_kwargs=self.write_kwargs,
                    rmm_pool_size="auto",
                    spill_memory_limit="auto",
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
        try:
            create_id_generator_actor()
        except ValueError:
            err_msg = """
            An existing id generator actor was found. Please remove or save the existing id generator with
            `nemo_curator.stages.deduplication.id_generator.write_id_generator_to_disk` (if needed) and remove the actor with
            `nemo_curator.stages.deduplication.id_generator.kill_id_generator_actor` before running the fuzzy deduplication pipeline.
            """
            raise RuntimeError(err_msg) from None

        try:
            minhash_pipeline = self._create_minhash_pipeline(generate_input_filegroups=initial_tasks is None)
            start_time = time.time()
            minhash_pipeline.run(executor=executor, initial_tasks=initial_tasks)
            minhash_end_time = time.time()
            logger.info(f"Minhash pipeline completed in {(minhash_end_time - start_time):.2f} seconds")

            lsh_pipeline = self._create_lsh_pipeline()
            lsh_start_time = time.time()
            # LSH stage generates it's own input tasks from the minhash directory
            lsh_tasks = lsh_pipeline.run(executor=executor, initial_tasks=None)
            lsh_end_time = time.time()
            logger.info(f"LSH pipeline completed in {(lsh_end_time - lsh_start_time):.2f} seconds")

            valid_lsh_tasks = [task for task in lsh_tasks if task._metadata.get("num_docs", 0) > 0]
            if len(valid_lsh_tasks) == 0:
                logger.info("No potential duplicates found in the dataset. Skipping connected components pipeline.")
            else:
                connected_components_pipeline = self._create_connected_components_pipeline()
                connected_components_start_time = time.time()
                connected_components_tasks = connected_components_pipeline.run(
                    executor=executor, initial_tasks=valid_lsh_tasks
                )
                connected_components_end_time = time.time()
                logger.info(
                    f"Connected components pipeline completed in {(connected_components_end_time - connected_components_start_time):.2f} seconds"
                )
                num_removed_documents = sum(
                    task._metadata.get("num_removal_ids", 0) for task in connected_components_tasks
                )
                logger.info(f"Number of documents removed: {num_removed_documents}")
                output_fs = get_fs(
                    self.output_path,
                    self.write_kwargs.get("storage_options") if self.write_kwargs is not None else None,
                )
                id_generator_path = output_fs.sep.join([self.output_path, ID_GENERATOR_OUTPUT_FILENAME])
                write_id_generator_to_disk(
                    id_generator_path,
                    storage_options=self.write_kwargs.get("storage_options")
                    if self.write_kwargs is not None
                    else None,
                )
                logger.info(f"Id generator written to {id_generator_path}")
            end_time = time.time()
            logger.info(f"Fuzzy deduplication pipeline completed in {(end_time - start_time):.2f} seconds")
        finally:
            kill_id_generator_actor()
