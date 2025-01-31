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
from dataclasses import dataclass, field
from typing import List, Optional

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
    profile_dir: Optional[str] = None
    id_field: str = "id"
    text_field: str = "text"

    # Minhash + LSH Config
    seed: int = 42
    char_ngrams: int = 24
    num_buckets: int = 20
    hashes_per_bucket: int = 13
    use_64_bit_hash: bool = False
    buckets_per_shuffle: int = 1

    false_positive_check: bool = False
    # Only required for false positive check
    num_anchors: Optional[int] = None
    jaccard_threshold: Optional[float] = None
    bucket_mapping_blocksize: Optional[int] = None
    parts_per_worker: Optional[int] = None
    bucket_parts_per_worker: Optional[int] = None

    def __post_init__(self):
        self.num_hashes = self.num_buckets * self.hashes_per_bucket
        false_positive_defaults = {
            "num_anchors": 2,
            "jaccard_threshold": 0.8,
            "bucket_mapping_blocksize": 256,
            "parts_per_worker": 1,
            "bucket_parts_per_worker": 8,
        }
        if self.false_positive_check:
            warnings.warn(
                "Identifying false positives during the Minhash deduplication is computationally expensive."
                " For improved performance consider setting this to False"
            )
            for arg, default in false_positive_defaults.items():
                if getattr(self, arg) is None:
                    setattr(self, arg, default)
            if self.num_anchors <= 0:
                raise ValueError("Number of anchors must be greater than 0")
            if self.num_anchors > 2:
                warnings.warn(
                    "Using a higher number of anchor docs might lead to higher memory footprint and might impact performance",
                    category=UserWarning,
                )
            if not 0 <= self.jaccard_threshold <= 1:
                raise ValueError("Jaccard Threshold must be between [0,1]")
        else:
            if self.char_ngrams < 20:
                warnings.warn(
                    "Using a small char_ngrams value might lead to a large number (~5%) of false positives during deduplication."
                    " Using a value of at least 20 for char_ngrams is recommended."
                )
            unused_false_positive_args = [
                arg
                for arg in false_positive_defaults.keys()
                if getattr(self, arg) is not None
            ]
            if unused_false_positive_args:
                warnings.warn(
                    f"False positive check is disabled. Unused arguments {unused_false_positive_args} will be ignored",
                    category=UserWarning,
                )

        if self.cache_dir is None:
            raise ValueError(
                "Finding fuzzy duplicates requires a cache directory accessible via all workers to store intermediates"
            )
        if not 1 <= self.buckets_per_shuffle <= self.num_buckets:
            raise ValueError("Buckets per shuffle must be between [1, num_buckets]")


@dataclass
class SemDedupConfig(BaseConfig):
    """
    Configuration for Semantic Deduplication.

    Attributes:
        cache_dir (str): Directory to store cache.
        profile_dir (Optional[str]): If specified directory to write dask profile. Default is None.
        cache_dir (str): Directory to store cache.
        num_files (int): Number of files. Default is -1, meaning all files.
        embeddings_save_loc (str): Location to save embeddings.
        embedding_model_name_or_path (str): Model name or path for embeddings.
        embedding_batch_size (int): Inital Batch size for processing embeddings.
        write_embeddings_to_disk (bool): If True, saves the embeddings to disk, defaults to True.
            We recommend setting this to False when you have a delayed pipeline.
            Setting it to False can lead to more memory overhead.
        clustering_save_loc (str): Location to save clustering results.
        n_clusters (int): Number of clusters.
        seed (int): Seed for clustering.
        max_iter (int): Maximum iterations for clustering.
        kmeans_with_cos_dist (bool): Use KMeans with cosine distance.
        which_to_keep (str): Which duplicates to keep.
        largest_cluster_size_to_process (int): Largest cluster size to process.
        sim_metric (str): Similarity metric for deduplication.
        eps_thresholds (List[float]): Epsilon thresholds to calculate if semantically similar or not.
        eps_to_extract (float): Epsilon value to extract deduplicated data.
    """

    cache_dir: str
    profile_dir: Optional[str] = None
    num_files: int = -1

    # Embeddings
    embeddings_save_loc: str = "embeddings"
    embedding_model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 128
    write_embeddings_to_disk: bool = True

    # Clustering config
    clustering_save_loc: str = "clustering_results"
    n_clusters: int = 1000
    seed: int = 1234
    max_iter: int = 100
    kmeans_with_cos_dist: bool = False

    # Semdedup config
    which_to_keep: str = "hard"
    largest_cluster_size_to_process: int = 100000
    sim_metric: str = "cosine"

    # Extract dedup config
    eps_thresholds: List[float] = field(default_factory=lambda: [0.01, 0.001])
    eps_to_extract: float = 0.01

    def __post_init__(self):
        if self.cache_dir is None:
            raise ValueError(
                "Finding sem-dedup requires a cache directory accessible via all workers to store intermediates"
            )

        if self.eps_to_extract not in self.eps_thresholds:
            raise ValueError(
                f"Epsilon to extract {self.eps_to_extract} must be in eps_thresholds {self.eps_thresholds}"
            )
