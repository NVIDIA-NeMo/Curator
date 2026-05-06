# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cupy as cp
import numpy as np

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar
from nemo_curator.tasks import FileGroupTask, _EmptyTask
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS, check_disallowed_kwargs

from .utils import break_parquet_partition_into_groups, get_array_from_df

if TYPE_CHECKING:
    import cudf

import gc
import os
import time

import torch
from loguru import logger

# Column names
L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"


class KMeansReadFitWriteStage(ProcessingStage[FileGroupTask, _EmptyTask], DeduplicationIO):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    def __init__(  # noqa: PLR0913
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        filetype: Literal["parquet", "jsonl"],
        # KMeans args
        n_clusters: int,
        metadata_fields: list[str] | None = None,
        embedding_dim: int | None = None,
        verbose: bool = False,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        init: Literal["k-means||", "random"] | np.ndarray = "k-means||",
        n_init: int | Literal["auto"] = 1,
        oversampling_factor: float = 2.0,
        max_samples_per_batch: int = 1 << 15,
        max_rows_for_fitting: int | None = None,
        # I/O args
        cache_path: str | None = None,
        read_kwargs: dict[dict] | None = None,
        write_kwargs: dict[dict] | None = None,
    ):
        """KMeans clustering stage that requires RAFT for distributed processing.

        Args:
            id_field (str): The column name of the id column.
            embedding_field (str): The column name of the embedding column.
            output_path (str): The path to the output directory.
            n_clusters (int): The number of clusters to create.
            metadata_fields (list[str] | None): The columns to keep in the output. These columns can be used later to prioritize deduplication.
            embedding_dim (int | None): The dimension of the embedding. This helps us read data into smaller chunks.
            verbose (bool): Whether to print verbose output.
            max_iter (int): The maximum number of iterations to run.
            tol (float): Tolerance for stopping criteria of the kmeans algorithm.
            random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
            init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
            oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
            max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
            max_rows_for_fitting (int | None): Maximum number of rows (per actor) to use when fitting the KMeans model. When set, uses a two-pass approach: Pass 1 samples this many rows from all groups to fit the model (freeing each group's GPU memory immediately after sampling), then Pass 2 loads each group one at a time to predict labels and write results. This dramatically reduces peak GPU memory for large datasets. If None, all rows are loaded simultaneously (original behavior).
            cache_path (str | None): The path to save the centroids to. If None, the centroids will not be saved.
            read_kwargs (dict[dict]): Keyword arguments for the read stage.
            write_kwargs (dict[dict]): Keyword arguments for the write stage.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        self.filetype = filetype
        self.n_clusters = n_clusters
        self.metadata_fields = metadata_fields if metadata_fields is not None else []
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch
        self.max_rows_for_fitting = max_rows_for_fitting

        self.cache_path = cache_path
        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}

        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["partition_file_name", "partition_cols", "index"])

        self.input_storage_options = self.read_kwargs.pop("storage_options", None)
        self.output_storage_options = self.write_kwargs.pop("storage_options", None)

        self.name = "KMeansStage"
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> _EmptyTask:
        msg = "KMeansReadFitWriteStage does not support single-task processing"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[_EmptyTask]:
        """Process a batch of FileGroupTasks using distributed RAFT KMeans.

        In RAFT mode, each actor processes its assigned tasks, but the KMeans model
        is trained cooperatively across all actors using RAFT communication.

        When max_rows_for_fitting is set, uses a memory-efficient two-pass approach:
          Pass 1: reads only the embedding column, samples rows, frees GPU memory per group
          Pass 2: loads each group one at a time for prediction and writing
        Otherwise, loads all groups simultaneously (original behavior).
        """

        if not tasks:
            return []

        # Collect all files from all tasks
        all_files = [file for task in tasks for file in task.data]

        # Break files into subgroups to avoid cudf row limits
        if self.filetype == "parquet":
            groups = break_parquet_partition_into_groups(all_files, embedding_dim=self.embedding_dim)
        elif self.filetype == "jsonl":
            # For JSONL files, just group all files together since we can't easily estimate size
            groups = [all_files]
        else:
            msg = f"Unsupported filetype: {self.filetype}. Only jsonl and parquet are supported."
            raise ValueError(msg)

        if self.max_rows_for_fitting is not None:
            return self._process_batch_two_pass(tasks, groups)
        return self._process_batch_single_pass(tasks, groups)

    def _read_group(self, group: list[str], columns: list[str]) -> "cudf.DataFrame":
        """Read a group of files into a cudf DataFrame."""
        if self.filetype == "parquet":
            return self.read_parquet(
                group,
                columns=columns,
                storage_options=self.input_storage_options,
                assign_id=False,
                **self.read_kwargs,
            )
        if self.filetype == "jsonl":
            return self.read_jsonl(
                group,
                columns=columns,
                storage_options=self.input_storage_options,
                assign_id=False,
                **self.read_kwargs,
            )
        msg = f"Unsupported data type: {self.filetype}"
        raise ValueError(msg)

    def _process_batch_single_pass(self, tasks: list[FileGroupTask], groups: list[list[str]]) -> list["_EmptyTask"]:
        """Original single-pass approach: loads all groups simultaneously.

        Requires peak GPU memory = sum(all groups' data). Only suitable when the
        total dataset fits in GPU memory.
        """
        t0 = time.perf_counter()
        # Maintain a list of DataFrames and embeddings arrays for later use
        all_dfs, embeddings_arrays = [], []

        for group in groups:
            df = self._read_group(group, [self.id_field, self.embedding_field, *self.metadata_fields])
            # Normalize the embeddings
            df = self.normalize_embeddings_col_in_df(df, self.embedding_field)
            all_dfs.append(df)
            # Convert embeddings to cupy array to avoid cudf row limits
            embeddings_arrays.append(get_array_from_df(df, self.embedding_field))

        t1 = time.perf_counter()
        self._log_metrics({"kmeans_read_time": t1 - t0, "num_rows": sum(len(df) for df in all_dfs)})
        logger.debug(f"Read time: {(t1 - t0):.2f} seconds")

        # Fit the model cooperatively across actors, then predict on local data
        concatenated_embeddings = cp.concatenate(embeddings_arrays, axis=0)
        self.kmeans._fit(concatenated_embeddings, sample_weight=None, convert_dtype=False, multigpu=True)

        if self.cache_path is not None and getattr(self, "_actor_index", 0) == 0:
            os.makedirs(self.cache_path, exist_ok=True)
            cp.save(f"{self.cache_path}/kmeans_centroids.npy", self.kmeans.cluster_centers_)
            logger.info(f"Saved {self.n_clusters} KMeans centroids to {self.cache_path}/kmeans_centroids.npy")

        labels = self.kmeans.predict(concatenated_embeddings, convert_dtype=False).astype(cp.int32)

        t2 = time.perf_counter()
        self._log_metric("kmeans_fit_predict_time", t2 - t1)
        logger.info(f"KMeans fit+predict time: {(t2 - t1):.2f} seconds")

        results = []
        num_rows_seen = 0

        # Assign labels back to DataFrame and write results
        for i, df in enumerate(all_dfs):
            end_idx = num_rows_seen + len(df)
            df["centroid"] = labels[num_rows_seen:end_idx]
            num_rows_seen = end_idx
            # Assign distances using the fitted cluster centers
            df = self._assign_distances(df, self.embedding_field, self.kmeans.cluster_centers_)  # noqa: PLW2901

            output_filename = f"{tasks[0]._uuid}_{i}"
            # Write results for this subgroup
            self.write_parquet(
                df,
                self.output_path,
                partition_file_name=f"{output_filename}.parquet",
                partition_cols=["centroid"],
                index=False,
                storage_options=self.output_storage_options,
                **self.write_kwargs,
            )

            # Create result task for this subgroup
            results.append(
                _EmptyTask(
                    task_id=output_filename,
                    dataset_name=f"kmeans_group_{i}",
                    _metadata=None,
                    _stage_perf=[],
                    data=None,
                )
            )

        t3 = time.perf_counter()
        self._log_metric("kmeans_write_time", t3 - t2)
        logger.info(f"Write time: {(t3 - t2):.2f} seconds")

        return results

    def _process_batch_two_pass(self, tasks: list[FileGroupTask], groups: list[list[str]]) -> list["_EmptyTask"]:
        """Memory-efficient two-pass approach for large datasets.

        Pass 1: reads only the embedding column, samples up to max_rows_for_fitting
                rows per actor (with a floor of 1 row per group), frees each group's
                GPU memory immediately after sampling.
        Pass 2: loads each group one at a time (full columns), predicts labels, writes,
                then frees GPU memory before loading the next group.

        Peak GPU memory ≈ max(max_rows_for_fitting, one_group_size) x embedding_dim x 4 bytes,
        instead of total_data x embedding_dim x 4 bytes.
        """
        sample_rows_per_group = max(1, self.max_rows_for_fitting // len(groups))

        # --- Pass 1: sample embeddings for fitting ---
        t0 = time.perf_counter()
        sample_embeddings_list = []
        total_rows = 0

        for group_index, group in enumerate(groups):
            df = self._read_group(group, [self.embedding_field])
            total_rows += len(df)

            if len(df) > sample_rows_per_group:
                df = df.sample(sample_rows_per_group, random_state=self.random_state + group_index)

            df = self.normalize_embeddings_col_in_df(df, self.embedding_field)
            # .copy() detaches from the df so we can delete the df and free its memory
            sample_embeddings_list.append(get_array_from_df(df, self.embedding_field).copy())
            del df
            gc.collect()

        t1 = time.perf_counter()
        pass1_read_time = t1 - t0
        logger.debug(f"Pass 1 (sampling) time: {pass1_read_time:.2f} seconds, {total_rows} total rows")

        # Fit on sampled data cooperatively across all actors
        concatenated_samples = cp.concatenate(sample_embeddings_list, axis=0)
        del sample_embeddings_list
        gc.collect()
        logger.info(
            f"Fitting KMeans on {len(concatenated_samples)} sampled rows "
            f"(up to {sample_rows_per_group} per group)"
        )

        self.kmeans._fit(concatenated_samples, sample_weight=None, convert_dtype=False, multigpu=True)
        del concatenated_samples
        gc.collect()

        if self.cache_path is not None and getattr(self, "_actor_index", 0) == 0:
            os.makedirs(self.cache_path, exist_ok=True)
            cp.save(f"{self.cache_path}/kmeans_centroids.npy", self.kmeans.cluster_centers_)
            logger.info(f"Saved {self.n_clusters} KMeans centroids to {self.cache_path}/kmeans_centroids.npy")

        t2 = time.perf_counter()
        self._log_metric("kmeans_fit_time", t2 - t1)
        logger.info(f"KMeans fit time: {(t2 - t1):.2f} seconds")

        # --- Pass 2: predict and write, one group at a time ---
        results = []
        pass2_read_time = 0.0

        for i, group in enumerate(groups):
            t_read_start = time.perf_counter()
            df = self._read_group(group, [self.id_field, self.embedding_field, *self.metadata_fields])
            df = self.normalize_embeddings_col_in_df(df, self.embedding_field)
            embeddings_array = get_array_from_df(df, self.embedding_field)
            pass2_read_time += time.perf_counter() - t_read_start

            labels = self.kmeans.predict(embeddings_array, convert_dtype=False).astype(cp.int32)
            df["centroid"] = labels
            df = self._assign_distances(df, self.embedding_field, self.kmeans.cluster_centers_)

            output_filename = f"{tasks[0]._uuid}_{i}"
            self.write_parquet(
                df,
                self.output_path,
                partition_file_name=f"{output_filename}.parquet",
                partition_cols=["centroid"],
                index=False,
                storage_options=self.output_storage_options,
                **self.write_kwargs,
            )
            results.append(
                _EmptyTask(
                    task_id=output_filename,
                    dataset_name=f"kmeans_group_{i}",
                    _metadata=None,
                    _stage_perf=[],
                    data=None,
                )
            )

            del df, embeddings_array, labels
            gc.collect()

        t3 = time.perf_counter()
        self._log_metrics({
            "kmeans_read_time": pass1_read_time + pass2_read_time,
            "kmeans_predict_write_time": (t3 - t2) - pass2_read_time,
            "num_rows": total_rows,
        })
        logger.info(
            f"Pass 2 total time: {(t3 - t2):.2f} seconds "
            f"(read: {pass2_read_time:.2f}s, predict+write: {(t3 - t2) - pass2_read_time:.2f}s)"
        )

        return results

    def setup(self, _: WorkerMetadata | None = None) -> None:
        from cuml.cluster.kmeans import KMeans as cumlKMeans

        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self.kmeans = cumlKMeans(
            handle=self._raft_handle,
            output_type="cupy",
            init=self.init,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            n_init=self.n_init,
            oversampling_factor=self.oversampling_factor,
            max_samples_per_batch=self.max_samples_per_batch,
        )

    @staticmethod
    def normalize_embeddings_col_in_df(df: "cudf.DataFrame", embedding_col: str) -> "cudf.DataFrame":
        tensor = torch.Tensor(get_array_from_df(df, embedding_col))
        normalized_tensor = tensor / torch.norm(tensor, dim=1, keepdim=True)
        df[embedding_col] = create_list_series_from_1d_or_2d_ar(cp.asarray(normalized_tensor), index=df.index)
        return df

    @staticmethod
    def _assign_distances(df: "cudf.DataFrame", embedding_col: str, centroids: "cp.ndarray") -> "cudf.DataFrame":
        """
        Computes the L2 distance to nearest centroid to each embedding in the DataFrame.
        Embeddings are normalized. For cosine we'll need to normalize the centroids as well.
        """
        normalized_embeddings = get_array_from_df(df, embedding_col)
        # We normalize the centroids as well for cosine distance
        normalized_centroids = centroids / cp.linalg.norm(centroids, axis=1, keepdims=True)

        df[L2_DIST_TO_CENT_COL] = cp.sqrt(
            cp.sum((normalized_embeddings - centroids[df["centroid"].values]) ** 2, axis=1)
        )
        df[COSINE_DIST_TO_CENT_COL] = 1 - (
            cp.sum(
                normalized_embeddings * normalized_centroids[df["centroid"].values],
                axis=1,
            )
        )
        return df

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            "is_raft_actor": True,
        }


@dataclass
class KMeansStage(CompositeStage[_EmptyTask, _EmptyTask]):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    n_clusters: int
    id_field: str
    embedding_field: str
    input_path: str | list[str]
    output_path: str
    metadata_fields: list[str] | None = None
    verbose: bool = False
    embedding_dim: int | None = None
    # I/O args
    input_filetype: Literal["jsonl", "parquet"] = "parquet"
    input_file_extensions: list[str] | None = None
    read_kwargs: dict[dict] | None = None
    write_kwargs: dict[dict] | None = None
    # KMeans args
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    init: Literal["k-means||", "random"] | np.ndarray = "k-means||"
    n_init: int | Literal["auto"] = 1
    oversampling_factor: float = 2.0
    max_samples_per_batch: int = 1 << 15
    max_rows_for_fitting: int | None = None
    cache_path: str | None = None
    """KMeans clustering stage that requires RAFT for distributed processing.

    Args:
        n_clusters (int): The number of clusters to create.
        id_field (str): The column name of the id column.
        embedding_field (str): The column name of the embedding column.
        input_path (str | list[str]): The path to the input directory.
        output_path (str): The path to the output directory.
        metadata_fields (list[str] | None): The columns to keep in the output. These columns can be used later to prioritize deduplication.
        verbose (bool): Whether to print verbose output.
        embedding_dim (int | None): The dimension of the embedding. This helps us read data into smaller chunks.
        input_filetype (Literal["jsonl", "parquet"]): The type of the input file
        read_kwargs (dict[dict]): Keyword arguments for the read stage.
        write_kwargs (dict[dict]): Keyword arguments for the write stage.
        max_iter (int): The maximum number of iterations to run.
        tol (float): Tolerance for stopping criteria of the kmeans algorithm.
        random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
        init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
        max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
        max_rows_for_fitting (int | None): Maximum number of rows (per actor) to use when fitting the KMeans model. When set, uses a two-pass approach: Pass 1 samples this many rows from all groups to fit the model (freeing each group's GPU memory immediately after sampling), then Pass 2 loads each group one at a time to predict labels and write results. This dramatically reduces peak GPU memory for large datasets. If None, all rows are loaded simultaneously (original behavior).
        cache_path (str | None): The path to save the centroids to. If None, the centroids will not be saved.
    """

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        # Set default file extensions based on input_filetype if not provided
        file_extensions = self.input_file_extensions or FILETYPE_TO_DEFAULT_EXTENSIONS.get(self.input_filetype, [])
        if not file_extensions:
            msg = f"Unsupported filetype: {self.input_filetype}"
            raise ValueError(msg)

        return [
            FilePartitioningStage(
                file_paths=self.input_path,
                file_extensions=file_extensions,
                files_per_partition=1,  # We set this to one, and then the RaftActor will break it up into smaller groups
                storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs is not None else None,
            ),
            KMeansReadFitWriteStage(
                id_field=self.id_field,
                embedding_field=self.embedding_field,
                output_path=self.output_path,
                filetype=self.input_filetype,
                n_clusters=self.n_clusters,
                metadata_fields=self.metadata_fields,
                verbose=self.verbose,
                embedding_dim=self.embedding_dim,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                init=self.init,
                n_init=self.n_init,
                oversampling_factor=self.oversampling_factor,
                max_samples_per_batch=self.max_samples_per_batch,
                max_rows_for_fitting=self.max_rows_for_fitting,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
                cache_path=self.cache_path,
            ),
        ]
