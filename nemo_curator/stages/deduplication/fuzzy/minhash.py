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

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import cudf
import numpy as np
import pandas as pd
import pyarrow as pa
import rmm
from loguru import logger

from nemo_curator.stages.base import ProcessingStage, StageInputSpecs
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_DEFAULT_MINHASH_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR, get_id_generator_actor
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.utils.text import normalize_text
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs
from nemo_curator.utils.hash_utils import get_deterministic_hash

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


class MinHash(ABC):
    """
    Base class for computing minhash signatures of a document corpus
    """

    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
    ):
        """
        Parameters
        ----------
        seed: Seed for minhash permutations
        num_hashes: Length of minhash signature (No. of minhash permutations)
        char_ngrams: Width of text window (in characters) while computing minhashes.
        use_64bit_hash: Whether to use a 64 bit hash function.
        """
        self.num_hashes = num_hashes
        self.char_ngram = char_ngrams
        self.seed = seed
        self.use_64bit_hash = use_64bit_hash

    def generate_seeds(self, n_permutations: int = 260, seed: int = 0, bit_width: int = 32) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        This is a placeholder that child classes should implement if needed.
        """
        msg = "Child classes should implement this method if needed"
        raise NotImplementedError(msg)

    @abstractmethod
    def compute_minhashes(self, text_series: Any) -> Any:  # noqa: ANN401
        """
        Compute minhash signatures for the given dataframe text column.
        """


class GPUMinHash(MinHash):
    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
        pool: bool = False,
    ):
        # Initialize parent class
        MinHash.__init__(
            self,
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
        )

        # Initialize memory pool for cuDF
        if pool:
            rmm.reinitialize(pool_allocator=pool)

        # Generate seeds
        self.seeds = self.generate_seeds(
            n_permutations=self.num_hashes,
            seed=self.seed,
            bit_width=64 if self.use_64bit_hash else 32,
        )

    def generate_seeds(self, n_permutations: int = 260, seed: int = 0, bit_width: int = 32) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)

        if bit_width == 32:  # noqa: PLR2004
            MERSENNE_PRIME = np.uint32((1 << 31) - 1)  # noqa: N806
            dtype = np.uint32
        elif bit_width == 64:  # noqa: PLR2004
            # For 64-bit, use a larger prime number suitable for 64-bit operations
            MERSENNE_PRIME = np.uint64((1 << 61) - 1)  # noqa: N806
            dtype = np.uint64
        else:
            msg = "Unsupported bit width. Use either 32 or 64."
            raise ValueError(msg)

        return np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=dtype),
                    gen.randint(0, MERSENNE_PRIME, dtype=dtype),
                )
                for _ in range(n_permutations)
            ],
            dtype=dtype,
        )

    def minhash32(self, ser: cudf.Series) -> cudf.Series:
        """
        Compute 32bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        seeds_a = cudf.Series(self.seeds[:, 0], dtype="uint32")
        seeds_b = cudf.Series(self.seeds[:, 1], dtype="uint32")

        return ser.str.minhash(a=seeds_a, b=seeds_b, seed=self.seeds[0][0], width=self.char_ngram)

    def minhash64(self, ser: cudf.Series) -> cudf.Series:
        """
        Compute 64bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        seeds_a = cudf.Series(self.seeds[:, 0], dtype="uint64")
        seeds_b = cudf.Series(self.seeds[:, 1], dtype="uint64")

        return ser.str.minhash64(a=seeds_a, b=seeds_b, seed=self.seeds[0][0], width=self.char_ngram)

    def compute_minhashes(self, text_series: cudf.Series) -> cudf.Series:
        """
        Compute minhash signatures for the given text series.

        Parameters
        ----------
        text_series: cudf.Series
            Series containing text data to compute minhashes for

        Returns
        -------
        cudf.Series containing minhash signatures
        """
        if not isinstance(text_series, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        # Compute minhashes
        minhash_method = self.minhash64 if self.use_64bit_hash else self.minhash32
        return minhash_method(text_series)


class MinHashStage(ProcessingStage[FileGroupTask | DocumentBatch, FileGroupTask], DeduplicationIO):
    """
    ProcessingStage for computing MinHash signatures on documents for fuzzy deduplication.

    This stage accepts either a FileGroupTask (paths to input documents) or a DocumentBatch
    (in-memory pandas/pyarrow data already read by an upstream stage) and produces a
    FileGroupTask containing paths to computed minhash signature files. It uses GPU-accelerated
    MinHash computation to generate locality-sensitive hash signatures that can be used
    for approximate duplicate detection.

    The stage automatically handles:
    - Reading input files (JSONL or Parquet format), OR converting a DocumentBatch to cuDF
    - Assigning unique Integer IDs to documents using the IdGenerator actor (file path only;
      a DocumentBatch must already contain the ``_curator_dedup_id`` column)
    - Computing MinHash signatures using GPU acceleration
    - Writing results to Parquet files

    Parameters
    ----------
    output_path : str
        Base path where minhash output files will be written
    text_field : str, default="text"
        Name of the field containing text to compute minhashes from
    minhash_field : str, default="_minhash_signature"
        Name of the field where minhash signatures will be stored
    char_ngrams : int, default=24
        Width of character n-grams for minhashing
    num_hashes : int, default=260
        Number of hash functions (length of minhash signature)
    seed : int, default=42
        Random seed for reproducible minhash generation
    use_64bit_hash : bool, default=False
        Whether to use 64-bit hash functions (vs 32-bit)
    normalize_text : bool, default=False
        Whether to normalize text before computing minhashes
        Current normalization is limited to lowercase and trim whitespace
    read_format : Literal["jsonl", "parquet"] | None, default=None
        Format of input files. Only applies to FileGroupTask inputs; ignored for DocumentBatch
        inputs (which are already in memory). May be None when only DocumentBatch inputs are used.
    read_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for reading input files. Only applies to FileGroupTask inputs;
        ignored for DocumentBatch inputs.
    write_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for writing output files
    batch_size : int, default=1
        Number of input tasks to coalesce into a single minhash computation. Inputs are still
        read one at a time, but the concatenated frame is hashed and written as one block, which
        keeps the GPU saturated when upstream read blocks are small. The default of 1 preserves
        the previous one-task-in, one-file-out behavior. Must be positive. Values above 1 make the
        stage a fan-in, which is not source-attributable, so the stage is then not resumable and
        cannot be used with ``Pipeline.run(checkpoint_path=...)``. Peak device memory is roughly
        twice the combined size of one batch's inputs, so size it against free GPU memory.

    Examples
    --------
    >>> stage = MinHashStage(
    ...     output_path="/path/to/minhash/output",
    ...     text_field="content",
    ...     num_hashes=128,
    ...     char_ngrams=5
    ... )
    >>> # Use in a pipeline to process document batches
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        text_field: str = "text",
        minhash_field: str = CURATOR_DEFAULT_MINHASH_FIELD,
        char_ngrams: int = 24,
        num_hashes: int = 260,
        seed: int = 42,
        use_64bit_hash: bool = False,
        normalize_text: bool = False,
        read_format: Literal["jsonl", "parquet"] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        pool: bool = True,
        batch_size: int = 1,
    ):
        # Set ProcessingStage attributes
        self.name = self.__class__.__name__
        self.resources = Resources(gpus=1.0)  # Requires 1 GPU

        if batch_size < 1:
            msg = f"batch_size must be a positive integer, got {batch_size}"
            raise ValueError(msg)
        self.batch_size = batch_size
        # batch_size > 1 fans several inputs into one output, so the input->output mapping is no
        # longer source-attributable and resumability accounting cannot credit the sources. Same
        # reasoning as ConnectedComponentsStage / LSH. batch_size == 1 stays 1:1, so it stays
        # resumable and existing checkpointed pipelines are unaffected.
        self.is_resumable = batch_size == 1

        self.text_field = text_field
        self.minhash_field = minhash_field
        self.char_ngrams = char_ngrams
        self.num_hashes = num_hashes
        self.seed = seed
        self.use_64bit_hash = use_64bit_hash
        self.normalize_text = normalize_text
        self.read_format = read_format
        self.read_kwargs = read_kwargs or {}
        self.write_kwargs = write_kwargs or {}
        self.pool = pool
        # Initialize the minhash processor in setup
        self.minhash_processor = None
        self.id_generator = None

        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options", {}))
        self.output_path = self.output_fs.sep.join([output_path, self.name])
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        """Initialize the GPU MinHash processor and ID generator."""
        # The ID generator is only required for the FileGroupTask (file-read) path, where IDs
        # are assigned at read time. DocumentBatch inputs must already carry _curator_dedup_id,
        # so a missing actor is tolerated here; the file path surfaces a clear error at
        # process time if it actually needs the ID generator.
        try:
            self.id_generator = get_id_generator_actor()
        except ValueError:
            logger.warning(
                "IdGenerator actor was not found during MinHashStage setup. "
                "FileGroupTask inputs will fail at process time; DocumentBatch inputs are unaffected."
            )
            self.id_generator = None

        # Initialize the GPU minhash processor
        self.minhash_processor = GPUMinHash(
            seed=self.seed,
            num_hashes=self.num_hashes,
            char_ngrams=self.char_ngrams,
            use_64bit_hash=self.use_64bit_hash,
            pool=self.pool,
        )

    def teardown(self) -> None:
        self.minhash_processor = None
        gc.collect()
        if self.pool:
            rmm.reinitialize(pool_allocator=False)

    def inputs(self) -> StageInputSpecs:
        """Define input requirements for each supported input task type."""
        return {
            FileGroupTask: (["data"], []),
            DocumentBatch: (["data"], [CURATOR_DEDUP_ID_STR, self.text_field]),
        }

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs - produces FileGroupTask with minhash files."""
        return (["data"], [])

    def process(self, task: FileGroupTask | DocumentBatch) -> FileGroupTask:
        """
        Process a FileGroupTask or DocumentBatch to compute minhashes.

        Args:
            task: FileGroupTask containing file paths to process, or a DocumentBatch whose data
                already contains the ``_curator_dedup_id`` and text columns.

        Returns:
            FileGroupTask containing paths to minhash output files
        """

        self._check_setup()

        df = self._read_task(task)
        output_file = self.output_fs.sep.join([self.output_path, f"{task.task_id}.parquet"])
        self._minhash_and_write(df, output_file)

        return self._build_output_task(task, output_file, task._stage_perf)

    def process_batch(self, tasks: list[FileGroupTask | DocumentBatch]) -> list[FileGroupTask]:
        """
        Process several tasks as a single minhash computation.

        Inputs are read one at a time (the read is not saturated any more than it is in
        :meth:`process`), but the concatenated frame is hashed and written as one block. This
        lets a pipeline use a small read blocksize — which CPU stages prefer, and which avoids
        host OOM — while still handing the GPU a batch large enough to saturate it.

        Args:
            tasks: FileGroupTasks and/or DocumentBatches to coalesce.

        Returns:
            A single-element list holding the FileGroupTask for the combined minhash file, or an
            empty list when there is nothing to process.
        """
        if not tasks:
            return []

        self._check_setup()

        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)

        # Read one at a time, then concatenate once. cudf.concat needs the inputs and the combined
        # copy resident together, so peak device memory is roughly twice the combined input size --
        # inherent to the concat, and the reason batch_size should be sized against free GPU memory.
        # Concatenating incrementally would not lower that peak and would recopy the accumulator on
        # every step, so a single concat is both the cheaper and the simpler option.
        frames = []
        for task in tasks:
            frames.append(self._read_task(task))
        df = frames[0] if len(frames) == 1 else cudf.concat(frames, ignore_index=True)
        del frames

        output_file = self.output_fs.sep.join(
            [self.output_path, get_deterministic_hash([task.task_id for task in tasks], tasks[0].task_id) + ".parquet"]
        )
        self._minhash_and_write(df, output_file)

        # Keep the shared history from the first task, plus each other input's final record, so
        # this fan-in does not drop the upstream timings. Mirrors IdentifyDuplicatesStage.
        stage_perf = list(tasks[0]._stage_perf)
        stage_perf.extend(task._stage_perf[-1] for task in tasks[1:] if task._stage_perf)

        return [self._build_output_task(tasks[0], output_file, stage_perf)]

    def _check_setup(self) -> None:
        if self.minhash_processor is None:
            msg = "MinHash processor not initialized. Call setup() first."
            raise RuntimeError(msg)

    def _read_task(self, task: FileGroupTask | DocumentBatch) -> "cudf.DataFrame":
        """Read/convert one input into a cuDF DataFrame with the text and ID columns."""
        if isinstance(task, DocumentBatch):
            if self.read_format is not None:
                logger.warning(
                    f"read_format={self.read_format!r} is ignored for DocumentBatch inputs because their data is "
                    "already loaded."
                )
            return self._read_document_batch(task)
        return self._read_file_group(task)

    def _minhash_and_write(self, df: "cudf.DataFrame", output_file: str) -> None:
        """Compute minhash signatures for ``df`` and write them to ``output_file``."""
        result_df = df[[CURATOR_DEDUP_ID_STR]]
        text_for_minhash = df[self.text_field]

        if self.normalize_text:
            with self._time_metric("normalize_text_time"):
                text_for_minhash = normalize_text(text_for_minhash)

        with self._time_metric("minhash_compute_time"):
            result_df[self.minhash_field] = self.minhash_processor.compute_minhashes(text_for_minhash)

        # Write output file
        with self._time_metric("minhash_write_time"):
            self.write_parquet(df=result_df, filepath=output_file, **self.write_kwargs)

    def _build_output_task(
        self,
        base_task: FileGroupTask | DocumentBatch,
        output_file: str,
        stage_perf: list,
    ) -> FileGroupTask:
        """Build the FileGroupTask describing a written minhash file."""
        return FileGroupTask(
            dataset_name=f"{base_task.dataset_name}_minhash",
            data=[output_file],
            _metadata={
                **base_task._metadata,
                "minhash_field": self.minhash_field,
                "num_hashes": self.num_hashes,
                "storage_options": self.write_kwargs.get("storage_options"),
            },
            _stage_perf=stage_perf,
        )

    def _read_file_group(self, task: FileGroupTask) -> "cudf.DataFrame":
        """Read a FileGroupTask's files into cuDF, assigning IDs at read time."""
        with self._time_metric("minhash_file_read_time"):
            if self.id_generator is None:
                msg = (
                    "IdGenerator actor is required for FileGroupTask input but was not found. "
                    "Start it via create_id_generator_actor(), or pass a DocumentBatch whose data "
                    "already contains the _curator_dedup_id column."
                )
                raise RuntimeError(msg)

            read_kwargs = self.read_kwargs.copy()

            # Read input file based on format
            if self.read_format == "jsonl":
                return self.read_jsonl(filepath=task.data, columns=[self.text_field], assign_id=True, **read_kwargs)
            elif self.read_format == "parquet":
                return self.read_parquet(filepath=task.data, columns=[self.text_field], assign_id=True, **read_kwargs)
            else:
                msg = f"read_format must be 'jsonl' or 'parquet' to process a FileGroupTask; got {self.read_format!r}"
                raise ValueError(msg)

    def _read_document_batch(self, task: DocumentBatch) -> "cudf.DataFrame":
        """Convert an in-memory DocumentBatch to cuDF, keeping only the ID and text columns.

        Non-relevant columns are dropped on the host (before the GPU transfer), mirroring how the
        file path only reads ``columns=[text_field]``. The required columns are guaranteed present
        by ``validate_input``; when ``process`` is called directly a missing column will raise here.
        """
        with self._time_metric("minhash_document_batch_to_cudf_time"):
            keep = [CURATOR_DEDUP_ID_STR, self.text_field]
            data = task.data
            if isinstance(data, pa.Table):
                return cudf.DataFrame.from_arrow(data.select(keep))
            elif isinstance(data, pd.DataFrame):
                return cudf.from_pandas(data[keep])
            else:
                msg = f"Unsupported DocumentBatch data type: {type(data)}"
                raise TypeError(msg)
