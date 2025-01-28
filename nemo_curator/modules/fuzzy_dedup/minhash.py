# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Union

import cudf
import dask_cudf
import numpy as np

from nemo_curator._compat import MINHASH_DEPRECATED_API, MINHASH_PERMUTED_AVAILABLE
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix


class MinHash:
    """
    Computes minhash signatures of a document corpus
    """

    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
        logger: Union[logging.LoggerAdapter, str] = "./",
        id_field: str = "id",
        text_field: str = "text",
        profile_dir: str = None,
        cache_dir: str = None,
    ):
        """
        Parameters
        ----------
        seed: Seed for minhash permutations
        num_hashes: Length of minhash signature (No. of minhash permutations)
        char_ngrams: Width of text window (in characters) while computing minhashes.
        use_64bit_hash: Whether to use a 64 bit hash function.
        logger: Existing logger to log to, or a path to a log directory.
        id_field: Column in the Dataset denoting document ID.
        text_field: Column in the Dataset denoting document content.
        profile_dir: str, Default None
          If specified directory to write dask profile
        cache_dir: str, Default None
          If specified, will compute & write id, minhash pairs to directory
        """
        self.num_hashes = num_hashes
        self.char_ngram = char_ngrams

        if MINHASH_DEPRECATED_API:
            self.seeds = self.generate_seeds(n_seeds=self.num_hashes, seed=seed)
        else:
            self.seeds = self.generate_hash_permutation_seeds(
                bit_width=64 if use_64bit_hash else 32,
                n_permutations=self.num_hashes,
                seed=seed,
            )

        self.minhash_method = self.minhash64 if use_64bit_hash else self.minhash32
        self.id_field = id_field
        self.text_field = text_field

        if cache_dir is None and profile_dir is not None:
            warnings.warn(
                "cache_dir for intermediate outputs is required to generate profiles"
            )
        self.cache_dir = cache_dir
        self.profile_dir = profile_dir

        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "Minhash.log"),
                name="Minhash",
            )
        else:
            self._logger = logger

    def generate_seeds(self, n_seeds: int = 260, seed: int = 0) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)
        return gen.randint(0, 1e6, size=n_seeds)

    def generate_hash_permutation_seeds(
        self, bit_width: int, n_permutations: int = 260, seed: int = 0
    ) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)

        if bit_width == 32:
            MERSENNE_PRIME = np.uint32((1 << 31) - 1)
            dtype = np.uint32
        elif bit_width == 64:
            # For 64-bit, use a larger prime number suitable for 64-bit operations
            MERSENNE_PRIME = np.uint64((1 << 61) - 1)
            dtype = np.uint64
        else:
            raise ValueError("Unsupported bit width. Use either 32 or 64.")

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

    def minhash32(
        self, ser: cudf.Series, seeds: np.ndarray, char_ngram: int
    ) -> cudf.Series:
        """
        Compute 32bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            raise TypeError("Expected data of type cudf.Series")

        if MINHASH_DEPRECATED_API:
            warnings.warn(
                "Using an outdated minhash implementation, please update to cuDF version 24.12 "
                "or later for improved performance. "
                "Install the latest version of cuDF using `pip install curator[cuda12x_nightly]`",
                category=FutureWarning,
            )
            seeds = cudf.Series(seeds, dtype="uint32")
            return ser.str.minhash(seeds=seeds, width=char_ngram)
        else:
            seeds_a = cudf.Series(seeds[:, 0], dtype="uint32")
            seeds_b = cudf.Series(seeds[:, 1], dtype="uint32")

            if MINHASH_PERMUTED_AVAILABLE:
                return ser.str.minhash_permuted(
                    a=seeds_a, b=seeds_b, seed=seeds[0][0], width=char_ngram
                )
            else:
                return ser.str.minhash(
                    a=seeds_a, b=seeds_b, seed=seeds[0][0], width=char_ngram
                )

    def minhash64(
        self, ser: cudf.Series, seeds: np.ndarray, char_ngram: int
    ) -> cudf.Series:
        """
        Compute 64bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            raise TypeError("Expected data of type cudf.Series")
        if MINHASH_DEPRECATED_API:
            warnings.warn(
                "Using an outdated minhash implementation, please update to cuDF version 24.12 "
                "or later for improved performance. "
                "Install the latest version of cuDF using `pip install curator[cuda12x_nightly]`",
                category=FutureWarning,
            )
            seeds = cudf.Series(seeds, dtype="uint64")
            return ser.str.minhash64(seeds=seeds, width=char_ngram)
        else:
            seeds_a = cudf.Series(seeds[:, 0], dtype="uint64")
            seeds_b = cudf.Series(seeds[:, 1], dtype="uint64")

            if MINHASH_PERMUTED_AVAILABLE:
                return ser.str.minhash64_permuted(
                    a=seeds_a, b=seeds_b, seed=seeds[0][0], width=char_ngram
                )
            else:
                return ser.str.minhash64(
                    a=seeds_a, b=seeds_b, seed=seeds[0][0], width=char_ngram
                )

    def __call__(self, dataset: DocumentDataset) -> Union[str, DocumentDataset]:
        """
        Computes the MinHash Signatures for a given dataset.
        Parameters
        ----------
        dataset: DocumentDataset
        The input datset to compute MinHashes.
        Returns
        -------
        DocumentDataset containing IDs of all documents and the corresponding MinHash Signature
        """
        result = dataset.df[[self.id_field]]
        result["_minhash_signature"] = dataset.df[self.text_field].map_partitions(
            self.minhash_method,
            seeds=self.seeds,
            char_ngram=self.char_ngram,
        )

        if self.cache_dir is None:
            return DocumentDataset(result)

        t0 = time.time()
        self._logger.info("Starting execution for Minhashes")
        write_path = os.path.join(self.cache_dir, "_minhashes.parquet")
        if os.path.exists(write_path):
            warnings.warn(
                f"Output path {write_path} already exists and will be overwritten"
            )
        with performance_report_if_with_ts_suffix(self.profile_dir, "minhash-profile"):
            result.to_parquet(write_path, write_index=False, overwrite=True)
        self._logger.info(
            f"Time taken for Minhash signature computation = {time.time() - t0}s and output written at {write_path}"
        )
        return DocumentDataset(
            dask_cudf.read_parquet(write_path, blocksize="2GB", aggregate_files=True)
        )
