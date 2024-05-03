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

import warnings
from dataclasses import dataclass

import yaml


@dataclass
class BaseConfig:
    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            yaml_dict = yaml.safe_load(file)
        return cls(**yaml_dict)


@dataclass
class FuzzyDuplicatesConfig(BaseConfig):
    """
    Configuration for MinHash based fuzzy duplicates detection.
    Parameters
    ----------
    seed: Seed for minhash permutations
    char_ngrams: Size of Char ngram shingles used in minhash computation
    num_buckets: Number of Bands or buckets to use during Locality Sensitive Hashing
    hashes_per_bucket: Number of hashes per bucket/band.
    use_64_bit_hash: Whether to use a 32bit or 64bit hash function for minhashing.
    buckets_per_shuffle: Number of bands/buckets to shuffle concurrently.
        Larger values process larger batches by processing multiple bands
        but might lead to memory pressures and related errors.
    id_field: Column in the Dataset denoting document ID.
    text_field: Column in the Dataset denoting document content.
    profile_dir: str, Default None
        If specified directory to write dask profile
    cache_dir: str, Default None
        Location to store deduplcation intermediates such as minhashes/buckets etc.
    false_positive_check: bool,
        Whether to run a check to look for false positives within buckets.
        Note: This is a computationally expensive step.
    num_anchors: int
        Number of documents per bucket to use as reference for computing jaccard
        pairs within that bucket to identify false positives.
    jaccard_threshold: float
        The Jaccard similariy threshold to consider a document a near duplicate
        during false positive evaluations.
    """

    # General config
    cache_dir: str
    profile_dir: str = None
    id_field: str = "id"
    text_field: str = "text"

    # Minhash + LSH Config
    seed: int = 42
    char_ngrams: int = 5
    num_buckets: int = 20
    hashes_per_bucket: int = 13
    use_64_bit_hash: bool = False
    buckets_per_shuffle: int = 1

    false_positive_check: bool = True
    # Only required for fp check
    num_anchors: int = 2
    jaccard_threshold: float = 0.8

    def __post_init__(self):
        self.num_hashes = self.num_buckets * self.hashes_per_bucket
        if self.cache_dir is None:
            raise ValueError(
                "Finding fuzzy duplicates requires a cache directory accessible via all workers to store intermediates"
            )
        if not self.false_positive_check:
            raise NotImplementedError(
                "Skipping false positive checks is not supported at the moment"
            )
        if self.num_anchors <= 0:
            raise ValueError("Number of anchors must be greater than 0")
        if self.num_anchors > 2:
            warnings.warn(
                "Using a higher number of anchor docs might lead to higher memory footprint and might impact performance",
                category=UserWarning,
            )
        if not 0 <= self.jaccard_threshold <= 1:
            raise ValueError("Jaccard Threshold must be between [0,1]")
        if self.buckets_per_shuffle <= 0:
            raise ValueError("Buckets per shuffle must be greater than 0")
