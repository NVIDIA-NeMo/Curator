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
from typing import TYPE_CHECKING, Any, Literal

import cudf
import numpy as np
import rmm

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_DEFAULT_MINHASH_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR, get_id_generator_actor
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs

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


class MinHashStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """
    ProcessingStage for computing MinHash signatures on documents for fuzzy deduplication.

    This stage takes FileGroupTask containing paths to input documents and produces
    FileGroupTask containing paths to computed minhash signature files. It uses GPU-accelerated
    MinHash computation to generate locality-sensitive hash signatures that can be used
    for approximate duplicate detection.

    The stage automatically handles:
    - Reading input files (JSONL or Parquet format)
    - Assigning unique Integer IDs to documents using the IdGenerator actor
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
    read_format : Literal["jsonl", "parquet"], default="jsonl"
        Format of input files
    read_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for reading input files
    write_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for writing output files

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
        read_format: Literal["jsonl", "parquet"] = "jsonl",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        pool: bool = True,
    ):
        # Set ProcessingStage attributes
        self._name = self.__class__.__name__
        self._resources = Resources(gpus=1.0)  # Requires 1 GPU

        self.text_field = text_field
        self.minhash_field = minhash_field
        self.char_ngrams = char_ngrams
        self.num_hashes = num_hashes
        self.seed = seed
        self.use_64bit_hash = use_64bit_hash
        self.read_format = read_format
        self.read_kwargs = read_kwargs or {}
        self.write_kwargs = write_kwargs or {}
        self.pool = pool
        # Initialize the minhash processor in setup
        self.minhash_processor = None
        self.id_generator = None

        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options", {}))
        self.output_path = self.output_fs.sep.join([output_path, self._name])
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        """Initialize the GPU MinHash processor and ID generator."""
        # Initialize the ID generator (will be shared across workers)

        try:
            self.id_generator = get_id_generator_actor()
        except ValueError as e:
            err_msg = """
            Failed to get ID generator actor. Start an ID generator actor via `create_id_generator_actor` if calling this stage directly.
            If using the FuzzyDedup API this should be started automatically.
            """
            raise ValueError(err_msg) from e

        # Initialize the GPU minhash processor
        self.minhash_processor = GPUMinHash(
            seed=self.seed,
            num_hashes=self.num_hashes,
            char_ngrams=self.char_ngrams,
            use_64bit_hash=self.use_64bit_hash,
            pool=self.pool,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs - produces FileGroupTask with minhash files."""
        return (["data"], [])

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """
        Process a group of files to compute minhashes.

        Args:
            task: FileGroupTask containing file paths to process

        Returns:
            FileGroupTask containing paths to minhash output files
        """

        if self.minhash_processor is None or self.id_generator is None:
            msg = "MinHash processor or ID generator not initialized. Call setup() first."
            raise RuntimeError(msg)

        output_file = self.output_fs.sep.join([self.output_path, f"{task._uuid}.parquet"])

        read_kwargs = self.read_kwargs.copy()

        # Read input file based on format
        if self.read_format == "jsonl":
            df = self.read_jsonl(filepath=task.data, columns=[self.text_field], assign_id=True, **read_kwargs)
        elif self.read_format == "parquet":
            df = self.read_parquet(filepath=task.data, columns=[self.text_field], assign_id=True, **read_kwargs)
        else:
            msg = f"Unsupported read format: {self.read_format}"
            raise ValueError(msg)

        result_df = df[[CURATOR_DEDUP_ID_STR]]
        result_df[self.minhash_field] = self.minhash_processor.compute_minhashes(df[self.text_field])

        # Write output file
        self.write_parquet(df=result_df, filepath=output_file, **self.write_kwargs)

        # Return FileGroupTask with output file
        return FileGroupTask(
            task_id=f"{task.task_id}",
            dataset_name=f"{task.dataset_name}_minhash",
            data=[output_file],
            _metadata={
                **task._metadata,
                "minhash_field": self.minhash_field,
                "num_hashes": self.num_hashes,
                "storage_options": self.write_kwargs.get("storage_options"),
            },
            _stage_perf=task._stage_perf,
        )
