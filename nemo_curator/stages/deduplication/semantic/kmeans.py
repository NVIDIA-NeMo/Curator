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
        max_workers: int | None = None,
        # I/O args
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
            max_workers (int | None): Maximum number of actors. None lets the scheduler use all available GPUs.
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

        self._max_workers = max_workers

        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}

        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["partition_file_name", "partition_cols", "index"])

        self.input_storage_options = self.read_kwargs.pop("storage_options", None)
        self.output_storage_options = self.write_kwargs.pop("storage_options", None)

        self.name = "KMeansStage"
        self.resources = Resources(cpus=1.0, gpus=1.0, host_memory_gb=20.0)

    def num_workers(self) -> int | None:
        return self._max_workers

    def process(self, task: FileGroupTask) -> _EmptyTask:
        msg = "KMeansReadFitWriteStage does not support single-task processing"
        raise NotImplementedError(msg)

    # Max files to read in a single cudf.read_parquet call.  libcudf allocates
    # large temporary buffers during Parquet decompression + list-column
    # construction, and the GPU may already be partly occupied by NCCL/cuML
    # contexts or lingering memory from the preceding pipeline stage.
    _MAX_FILES_PER_READ = 25

    # Maximum rows to keep per actor for the KMeans _fit phase.
    # With 263k-dim float32 embeddings each row is ~1 MB, so 10 000 rows
    # ≈ 10.5 GB — leaves ~13 GB on a 24 GB GPU for cuML working memory
    # and NCCL communication buffers during the RAFT collective.
    _MAX_FIT_ROWS_PER_ACTOR = 10_000

    _FILE_READ_TIMEOUT_S = 120

    @staticmethod
    def _read_single_file_with_timeout(
        path: str, columns: list[str], timeout: int
    ) -> "pyarrow.Table":
        """Read a single parquet file with a timeout to avoid NFS hangs."""
        import concurrent.futures

        import pyarrow.parquet as pq

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pq.read_table, path, columns=columns)
            return future.result(timeout=timeout)

    def _read_chunk_resilient(
        self,
        files: list[str],
        columns: list[str],
        embedding_col: str | None = None,
    ) -> "list[tuple[cudf.DataFrame, np.ndarray]] | list[cudf.DataFrame]":
        """Read files via pyarrow (CPU) to avoid kvikio SIGSEGV on corrupt data.

        When *embedding_col* is given, embeddings are extracted as numpy
        arrays **on CPU** and only the remaining metadata columns are
        placed on GPU as cudf DataFrames.  This keeps GPU memory bounded
        to small metadata during the read loop — the bulk embedding data
        stays in host RAM until the caller explicitly moves it to GPU.

        Each file read is subject to ``_FILE_READ_TIMEOUT_S`` to prevent
        a single hanging NFS read from stalling the entire actor.

        Returns a list of ``(cudf_metadata_df, numpy_embedding)`` tuples
        when *embedding_col* is set, or a list of full cudf DataFrames
        otherwise.
        """
        import cudf

        results: list = []
        for f in files:
            try:
                table = self._read_single_file_with_timeout(
                    f, columns, self._FILE_READ_TIMEOUT_S
                )

                if embedding_col is not None:
                    emb_col = table.column(embedding_col)
                    flat = emb_col.combine_chunks().values.to_numpy(
                        zero_copy_only=False
                    )
                    n_rows = len(table)
                    if flat.size == 0 or n_rows == 0:
                        raise ValueError(f"Empty embedding column ({flat.size} elements, {n_rows} rows)")
                    emb_np = flat.reshape(n_rows, -1).astype(np.float32)
                    meta_table = table.drop(embedding_col)
                    meta_df = cudf.DataFrame.from_arrow(meta_table)
                    del table, emb_col, flat, meta_table
                    results.append((meta_df, emb_np))
                else:
                    results.append(cudf.DataFrame.from_arrow(table))
                    del table
            except Exception:
                logger.error(f"Skipping corrupted/hung file: {f}")
                continue

        return results

    def process_batch(self, tasks: list[FileGroupTask]) -> list[_EmptyTask]:
        """Process a batch of FileGroupTasks using distributed RAFT KMeans.

        Uses a two-phase approach so memory stays bounded regardless of
        dataset size:

        Phase 1 — Fit: read a random *subsample* of files capped at
        ``_MAX_FIT_ROWS_PER_ACTOR`` rows, transfer to GPU, and call
        ``_fit`` collectively across all RAFT actors.

        Phase 2 — Predict + Write: stream through *all* files in small
        chunks.  For each chunk: read on CPU, normalize, transfer to GPU,
        predict cluster labels, compute distances, write results, free.
        Peak memory is bounded to one chunk at a time.
        """
        import gc
        import random as stdlib_random

        if not tasks:
            return []

        gc.collect()
        try:
            import torch.cuda
            torch.cuda.empty_cache()
        except Exception:
            pass
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()

        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        logger.info(
            f"GPU memory after cleanup: {free_mem / 1e9:.2f} GB free / "
            f"{total_mem / 1e9:.2f} GB total"
        )

        all_files = [file for task in tasks for file in task.data]

        if self.filetype == "parquet":
            groups = break_parquet_partition_into_groups(all_files, embedding_dim=self.embedding_dim)
        elif self.filetype == "jsonl":
            groups = [all_files]
        else:
            msg = f"Unsupported filetype: {self.filetype}. Only jsonl and parquet are supported."
            raise ValueError(msg)

        chunked_groups: list[list[str]] = []
        for group in groups:
            for i in range(0, len(group), self._MAX_FILES_PER_READ):
                chunked_groups.append(group[i : i + self._MAX_FILES_PER_READ])

        logger.info(
            f"Total {len(all_files)} files in {len(chunked_groups)} chunks "
            f"(max {self._MAX_FILES_PER_READ} files per read)"
        )

        columns = [self.id_field, self.embedding_field, *self.metadata_fields]

        # ── Phase 1: Subsample → Fit ──────────────────────────────
        shuffled_indices = list(range(len(chunked_groups)))
        stdlib_random.Random(self.random_state).shuffle(shuffled_indices)

        t0 = time.perf_counter()
        fit_embeddings: list[np.ndarray] = []
        fit_rows = 0

        for idx in shuffled_indices:
            if fit_rows >= self._MAX_FIT_ROWS_PER_ACTOR:
                break
            chunk_results = self._read_chunk_resilient(
                chunked_groups[idx], columns=columns, embedding_col=self.embedding_field,
            )
            for _meta_df, emb_np in chunk_results:
                if len(emb_np) == 0:
                    continue
                norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
                np.maximum(norms, 1e-8, out=norms)
                emb_np /= norms
                del norms, _meta_df
                fit_embeddings.append(emb_np)
                fit_rows += emb_np.shape[0]

        if not fit_embeddings:
            msg = "No embedding rows loaded for KMeans fit"
            raise ValueError(msg)

        # Strictly trim to _MAX_FIT_ROWS_PER_ACTOR because the last
        # chunk can overshoot the target.
        if fit_rows > self._MAX_FIT_ROWS_PER_ACTOR:
            trimmed: list[np.ndarray] = []
            kept = 0
            for arr in fit_embeddings:
                if kept >= self._MAX_FIT_ROWS_PER_ACTOR:
                    break
                need = self._MAX_FIT_ROWS_PER_ACTOR - kept
                if arr.shape[0] > need:
                    arr = arr[:need]
                trimmed.append(arr)
                kept += arr.shape[0]
            fit_embeddings = trimmed
            fit_rows = kept
            del trimmed

        emb_dim = fit_embeddings[0].shape[1]
        fit_np = np.empty((fit_rows, emb_dim), dtype=np.float32)
        offset = 0
        for i, arr in enumerate(fit_embeddings):
            n = arr.shape[0]
            fit_np[offset : offset + n] = arr
            fit_embeddings[i] = None
            offset += n
        del fit_embeddings
        gc.collect()

        fit_gpu = cp.asarray(fit_np)
        del fit_np
        gc.collect()

        t1 = time.perf_counter()
        logger.info(
            f"Fit sample: {fit_rows} rows "
            f"(~{fit_rows * emb_dim * 4 / 1e9:.2f} GB) "
            f"read in {t1 - t0:.1f}s"
        )

        self.kmeans._fit(fit_gpu, sample_weight=None, convert_dtype=False, multigpu=True)
        t2 = time.perf_counter()
        logger.info(f"KMeans fit time: {t2 - t1:.2f}s")

        del fit_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        # ── Phase 2: Stream predict + write ───────────────────────
        results = []
        chunk_idx = 0
        total_chunks = len(chunked_groups)
        last_log_time = time.perf_counter()
        for chunk in chunked_groups:
            chunk_results = self._read_chunk_resilient(
                chunk, columns=columns, embedding_col=self.embedding_field,
            )
            for meta_df, emb_np in chunk_results:
                if len(meta_df) == 0:
                    continue

                norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
                np.maximum(norms, 1e-8, out=norms)
                emb_np /= norms
                del norms

                emb_gpu = cp.asarray(emb_np)
                del emb_np

                labels = self.kmeans.predict(emb_gpu, convert_dtype=False).astype(cp.int32)
                meta_df["centroid"] = labels
                del labels

                meta_df = self._assign_distances(  # noqa: PLW2901
                    meta_df, self.embedding_field, self.kmeans.cluster_centers_,
                    embeddings_array=emb_gpu,
                )

                meta_df[self.embedding_field] = create_list_series_from_1d_or_2d_ar(
                    emb_gpu, index=meta_df.index
                )
                del emb_gpu

                output_filename = f"{tasks[0]._uuid}_{chunk_idx}"
                self.write_parquet(
                    meta_df,
                    self.output_path,
                    partition_file_name=f"{output_filename}.parquet",
                    partition_cols=["centroid"],
                    index=False,
                    storage_options=self.output_storage_options,
                    **self.write_kwargs,
                )
                del meta_df

                results.append(
                    _EmptyTask(
                        task_id=output_filename,
                        dataset_name=f"kmeans_group_{chunk_idx}",
                        _metadata=None,
                        _stage_perf=[],
                        data=None,
                    )
                )
                chunk_idx += 1

                now = time.perf_counter()
                if now - last_log_time >= 120:
                    elapsed = now - t2
                    logger.info(
                        f"Predict+write progress: {chunk_idx}/{total_chunks} chunks "
                        f"({elapsed:.0f}s elapsed)"
                    )
                    last_log_time = now

            cp.get_default_memory_pool().free_all_blocks()

        t3 = time.perf_counter()
        logger.info(
            f"Predict+write time: {t3 - t2:.2f}s, "
            f"wrote {chunk_idx} chunks"
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
    def _assign_distances(
        df: "cudf.DataFrame",
        embedding_col: str,
        centroids: "cp.ndarray",
        *,
        embeddings_array: "cp.ndarray | None" = None,
    ) -> "cudf.DataFrame":
        """
        Computes L2 and cosine distances from each embedding to its nearest centroid.

        If *embeddings_array* is provided, it is used directly instead of
        extracting from the DataFrame.  This avoids requiring the heavy
        list-column to be present in *df*.
        """
        normalized_embeddings = (
            embeddings_array if embeddings_array is not None else get_array_from_df(df, embedding_col)
        )
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
    max_workers: int | None = None
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
                max_workers=self.max_workers,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
            ),
        ]
