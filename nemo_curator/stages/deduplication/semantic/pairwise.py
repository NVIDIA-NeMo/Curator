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
import os
import time
import traceback
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal

import cudf
import cupy as cp
import torch
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs

from .pairwise_io import ClusterWiseFilePartitioningStage
from .ranking import RankingStrategy
from .utils import (
    break_parquet_partition_into_groups,
    get_array_from_df,
    read_parquet_file_info,
)

PairwiseComputeDtype = Literal["auto", "float16", "float32"]


def _release_cached_memory() -> None:
    """Best-effort cleanup of caches owned directly by Pairwise operations."""
    cleanup_actions = (
        ("Python", gc.collect),
        ("Torch", torch.cuda.empty_cache),
        ("CuPy device", lambda: cp.get_default_memory_pool().free_all_blocks()),
        ("CuPy pinned", lambda: cp.get_default_pinned_memory_pool().free_all_blocks()),
    )
    for allocator, action in cleanup_actions:
        try:
            action()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to release {allocator} allocator cache: {exc}")


def _decode_embedding_array(df: "cudf.DataFrame", embedding_col: str) -> "cp.ndarray":
    """Decode the embedding storage representation independently of compute precision."""
    embeddings = get_array_from_df(df, embedding_col)
    if embeddings.dtype == cp.uint16:
        return embeddings.view(cp.float16)
    if embeddings.dtype == cp.float32:
        return embeddings
    if embeddings.dtype == cp.float64:
        # Python-float list columns historically arrive as float64. Pairwise
        # normalizes them to the previously used float32 precision.
        return embeddings.astype(cp.float32)
    msg = f"Expected uint16 FP16 bits or float32 embedding storage, got {embeddings.dtype}"
    raise TypeError(msg)


def validate_pairwise_batch_size(batch_size: object) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        msg = f"pairwise_batch_size must be a positive integer, got {batch_size!r}"
        raise ValueError(msg)


def validate_pairwise_compute_dtype(compute_dtype: object) -> None:
    if compute_dtype not in {"auto", "float16", "float32"}:
        msg = f"Unsupported compute_dtype: {compute_dtype}"
        raise ValueError(msg)


def _resolve_compute_dtype(cluster_reps: "torch.Tensor", compute_dtype: PairwiseComputeDtype) -> "torch.dtype":
    validate_pairwise_compute_dtype(compute_dtype)
    if cluster_reps.dtype not in {torch.float16, torch.float32}:
        msg = f"Pairwise requires float16 or float32 embeddings, got {cluster_reps.dtype}"
        raise TypeError(msg)
    if cluster_reps.dtype == torch.float16 and compute_dtype == "float32":
        msg = "float16 embeddings must use compute_dtype='auto' or 'float16'"
        raise ValueError(msg)
    if compute_dtype == "float16":
        return torch.float16
    if compute_dtype == "float32":
        return torch.float32
    return cluster_reps.dtype


def pairwise_cosine_similarity_batched(
    cluster_reps: "torch.Tensor",
    batch_size: int = 1024,
    compute_dtype: PairwiseComputeDtype = "float32",
) -> tuple["cp.ndarray", "cp.ndarray"]:
    """Return each ranked row's most similar earlier row and its similarity.

    The input order is a preservation preference: row ``A`` is preferred over
    ``B``, ``B`` over ``C``, and so on. For example, consider these normalized
    four-dimensional embeddings::

        X = A [1.00, 0.00, 0.00, 0.00]
            B [0.96, 0.28, 0.00, 0.00]
            C [0.00, 1.00, 0.00, 0.00]
            D [0.00, 0.00, 1.00, 0.00]
            E [0.00, 0.96, 0.00, 0.28]

    Because the rows have unit length, ``S = X @ X.T`` is their full cosine-
    similarity matrix::

        query \\ candidate    A       B       C       D       E
        A                  1.0000  0.9600  0.0000  0.0000  0.0000
        B                  0.9600  1.0000  0.2800  0.0000  0.2688
        C                  0.0000  0.2800  1.0000  0.0000  0.9600
        D                  0.0000  0.0000  0.0000  1.0000  0.0000
        E                  0.0000  0.2688  0.9600  0.0000  1.0000

    We use only the strict lower triangle of this matrix::

        query \\ candidate    A       B       C       D       E
        A                     -       x       x       x       x
        B                  0.9600     -       x       x       x
        C                  0.0000  0.2800     -       x       x
        D                  0.0000  0.0000  0.0000     -       x
        E                  0.0000  0.2688  0.9600  0.0000     -

    Here ``x`` is a later-ranked candidate and ``-`` is self-similarity; neither
    is eligible. Query row ``i`` may select candidate row ``j`` only when ``j < i``.
    Each query selects the maximum eligible similarity, giving ``B -> A (0.96)``,
    ``C -> B (0.28)``, ``D -> A (0.00)``, and ``E -> C (0.96)``. ``D`` demonstrates
    deterministic ties: ``torch.max`` chooses the earliest-ranked candidate ``A``.
    The first row has no eligible candidate, so it maps to itself with score zero.

    The downstream ``IdentifyDuplicatesStage`` removes the query row's ``id`` when
    its score is at least ``1 - eps``; ``max_id`` is the earlier row it matched, not
    the row to remove. With ``eps=0.1`` in the example, ``B`` and ``E`` are removed
    because their scores are at least ``0.9``, while ``A``, ``C``, and ``D`` remain.
    This is a fixed ranked comparison rather than greedy survivor tracking: an
    earlier candidate remains eligible here even if that candidate's own score later
    causes it to be removed.

    Materializing all of ``S`` requires ``O(N^2)`` memory. This implementation
    computes query columns in batches of width ``B`` and retains only an ``N x B``
    workspace, without changing which neighbor the full masked matrix would select.
    Multiplication uses ``compute_dtype`` and returns scores in that same dtype.
    """
    validate_pairwise_batch_size(batch_size)
    resolved_compute_dtype = _resolve_compute_dtype(cluster_reps, compute_dtype)
    cluster_reps = cluster_reps.to(device="cuda", dtype=resolved_compute_dtype)
    batch_size = min(batch_size, len(cluster_reps))

    num_rows = len(cluster_reps)
    max_similarity = torch.zeros(num_rows, dtype=cluster_reps.dtype, device=cluster_reps.device)
    max_indices = torch.zeros(num_rows, dtype=torch.int64, device=cluster_reps.device)
    pairwise_sim_workspace = torch.empty(
        (num_rows, batch_size),
        dtype=cluster_reps.dtype,
        device=cluster_reps.device,
    )
    invalid_batch_triangle = torch.ones(
        (batch_size, batch_size),
        dtype=torch.bool,
        device=cluster_reps.device,
    ).tril_()

    for start_idx in range(0, num_rows, batch_size):
        end_idx = min(start_idx + batch_size, num_rows)
        batch_width = end_idx - start_idx

        # In the conceptual query-by-candidate matrix above, only the strict lower
        # triangle is valid. torch.mm below stores its transpose (candidate rows by
        # query columns), so only the candidate prefix through this query block can
        # contain valid entries.
        candidate_reps = cluster_reps[:end_idx]
        pairwise_sim_matrix = pairwise_sim_workspace[:end_idx, :batch_width]
        torch.mm(candidate_reps, cluster_reps[start_idx:end_idx].T, out=pairwise_sim_matrix)

        # Candidate rows from earlier blocks remain valid. Within the current block,
        # candidate row >= query column represents self or a later-ranked candidate;
        # that is the diagonal/lower triangle in this transposed workspace.
        pairwise_sim_matrix[start_idx:end_idx].masked_fill_(
            invalid_batch_triangle[:batch_width, :batch_width],
            -torch.inf,
        )

        max_values, batch_max_indices = torch.max(pairwise_sim_matrix, dim=0)
        if start_idx == 0:
            max_values[0] = 0.0
            batch_max_indices[0] = 0
        max_similarity[start_idx:end_idx] = max_values
        max_indices[start_idx:end_idx] = batch_max_indices

    return cp.asarray(max_similarity), cp.asarray(max_indices)


class PairwiseCosineSimilarityStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """Pairwise cosine similarity stage that computes similarity within clusters."""

    def __init__(  # noqa: PLR0913
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        ranking_strategy: RankingStrategy,
        pairwise_batch_size: int = 1024,
        verbose: bool = False,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        compute_dtype: PairwiseComputeDtype = "float32",
    ):
        """Initialize the pairwise cosine similarity stage.

        Args:
            id_field: The column name of the id column.
            embedding_field: The column name of the embedding column.
            output_path: The path to the output directory.
            ranking_strategy: Strategy for ranking/sorting clusters before similarity computation.
            pairwise_batch_size: Positive batch size for the bounded similarity workspace.
            verbose: Whether to print verbose output.
            compute_dtype: Multiplication dtype. ``"auto"`` retains the decoded embedding precision.
            read_kwargs: Kwargs for reading parquet files.
            write_kwargs: Kwargs for writing parquet files.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        validate_pairwise_batch_size(pairwise_batch_size)
        self.pairwise_batch_size = pairwise_batch_size
        validate_pairwise_compute_dtype(compute_dtype)
        self.compute_dtype = compute_dtype
        self.ranking_strategy = ranking_strategy
        self.verbose = verbose
        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}
        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["index"])
        self.input_storage_options = self.read_kwargs.pop("storage_options", None) if self.read_kwargs else None
        self.output_storage_options = self.write_kwargs.pop("storage_options", None) if self.write_kwargs else None
        self.name = "PairwiseCosineSimilarityStage"
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process one cluster, releasing cached allocations if it fails."""
        try:
            return self._process(task)
        except BaseException as exc:
            # An exception's traceback otherwise keeps the unwound _process
            # frame—and its large GPU objects—alive until after this finalizer.
            traceback.clear_frames(exc.__traceback__)
            _release_cached_memory()
            raise

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        """Process clusters while aggregating their phase timings for the adapter batch."""
        results = []
        batch_metrics: dict[str, float] = {}
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            results.append(self.process(task))
            for name, value in self._consume_custom_metrics().items():
                batch_metrics[name] = batch_metrics.get(name, 0.0) + value
        self._log_metrics(batch_metrics)
        return results

    def teardown(self) -> None:
        """Release worker-local caches once after all clusters are processed."""
        _release_cached_memory()

    def _process(self, task: FileGroupTask) -> FileGroupTask:  # noqa: PLR0915
        """Process a PairwiseFileGroupTask to compute pairwise similarities."""
        if task._metadata.get("filetype") != "parquet":
            msg = f"PairwiseCosineSimilarityStage only supports parquet files, got {task._metadata.get('filetype')}"
            raise ValueError(msg)

        cluster_id = task._metadata.get("centroid_id")
        output_path = os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")
        if cluster_id is None:
            msg = "centroid_id not found in task metadata"
            raise ValueError(msg)

        process_started = time.perf_counter()

        footer_start = time.perf_counter()
        additional_cols = self.ranking_strategy.metadata_cols if self.ranking_strategy.strategy == "sort" else []
        metadata_cols = list(dict.fromkeys([self.id_field, *additional_cols]))
        columns = [*metadata_cols, self.embedding_field]
        file_info = read_parquet_file_info(
            task.data,
            retained_columns=metadata_cols,
            embedding_column=self.embedding_field,
            storage_options=self.input_storage_options,
        )
        footer_time = time.perf_counter() - footer_start

        read_start = time.perf_counter()
        frames = [
            self.read_parquet(
                group,
                columns=columns,
                assign_id=False,
                storage_options=self.input_storage_options,
                **self.read_kwargs,
            )
            for group in break_parquet_partition_into_groups(file_info)
        ]
        read_time = time.perf_counter() - read_start
        if not frames or not any(len(frame) for frame in frames):
            logger.warning(f"No data found for cluster {cluster_id}")
            return FileGroupTask(
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
                data=[],
            )

        metadata_cluster_df = cudf.concat([frame[metadata_cols] for frame in frames], ignore_index=True).reset_index(
            drop=True
        )
        num_rows = len(metadata_cluster_df)
        if num_rows == 1:
            result_df = cudf.DataFrame(
                {
                    "id": metadata_cluster_df[self.id_field],
                    "max_id": metadata_cluster_df[self.id_field],
                    "cosine_sim_score": cudf.Series([0], dtype="float32"),
                }
            )
            write_start = time.perf_counter()
            self.write_parquet(
                result_df, output_path, storage_options=self.output_storage_options, index=False, **self.write_kwargs
            )
            self._log_metrics(
                {
                    "pairwise_footer_scan_time": footer_time,
                    "pairwise_read_time": read_time,
                    "pairwise_rank_time": 0.0,
                    "pairwise_conversion_time": 0.0,
                    "pairwise_compute_time": 0.0,
                    "pairwise_write_time": time.perf_counter() - write_start,
                }
            )
            return FileGroupTask(
                dataset_name=task.dataset_name,
                _metadata={
                    **task._metadata,
                    "centroid_id": cluster_id,
                },
                _stage_perf=task._stage_perf,
                data=[os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")],
            )

        rank_start = time.perf_counter()
        metadata_cluster_df["_original_idx"] = metadata_cluster_df.index
        ranked_metadata_df = self.ranking_strategy.rank_cluster(metadata_cluster_df)
        reorder_indices = ranked_metadata_df["_original_idx"].values
        ranked_metadata_df = ranked_metadata_df.drop(columns=["_original_idx"])
        destination_indices = cp.empty(num_rows, dtype=cp.int64)
        destination_indices[reorder_indices] = cp.arange(num_rows, dtype=cp.int64)

        first_embeddings = _decode_embedding_array(frames[0], self.embedding_field)
        ranked_embeddings = cp.empty((num_rows, first_embeddings.shape[1]), dtype=first_embeddings.dtype)
        source_offset = 0
        embedding_arrays = (_decode_embedding_array(frame, self.embedding_field) for frame in frames[1:])
        for embeddings in chain([first_embeddings], embedding_arrays):
            if embeddings.shape[1:] != ranked_embeddings.shape[1:] or embeddings.dtype != ranked_embeddings.dtype:
                msg = "All Pairwise embedding files must have the same width and storage dtype"
                raise TypeError(msg)
            source_stop = source_offset + len(embeddings)
            ranked_embeddings[destination_indices[source_offset:source_stop]] = embeddings
            source_offset = source_stop
        if source_offset != num_rows:
            msg = f"Pairwise metadata contained {num_rows} rows but embedding reads returned {source_offset}"
            raise RuntimeError(msg)
        del embeddings, first_embeddings, embedding_arrays, frames
        cluster_embeddings = torch.as_tensor(ranked_embeddings, device="cuda")
        # Finish queued reordering before closing the rank timing interval.
        torch.cuda.synchronize()
        rank_time = time.perf_counter() - rank_start

        ids = ranked_metadata_df[self.id_field]

        conversion_start = time.perf_counter()
        resolved_compute_dtype = _resolve_compute_dtype(cluster_embeddings, self.compute_dtype)
        storage_was_converted = cluster_embeddings.dtype != resolved_compute_dtype
        cluster_embeddings = cluster_embeddings.to(dtype=resolved_compute_dtype)
        torch.cuda.synchronize()
        if storage_was_converted:
            ranked_embeddings = None
        conversion_time = time.perf_counter() - conversion_start

        # Compute pairwise similarities after any requested precision conversion.
        compute_start = time.perf_counter()
        resolved_batch_size = min(self.pairwise_batch_size, num_rows)
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            cluster_embeddings, resolved_batch_size, compute_dtype=self.compute_dtype
        )
        # Finish the matrix multiplications before recording compute time and
        # returning their now-unused Torch workspace to the allocator.
        torch.cuda.synchronize()
        del cluster_embeddings, ranked_embeddings
        torch.cuda.empty_cache()
        compute_time = time.perf_counter() - compute_start

        # Convert indices back to IDs
        max_indices_id = ids.iloc[max_indices].reset_index(drop=True)

        # Create result dataframe
        points_to_remove_df = cudf.DataFrame(
            {
                "id": ids,
                "max_id": max_indices_id,
                # cuDF has no numeric FP16 column, so retain FP16 through the
                # reduction and promote only when materializing the output.
                "cosine_sim_score": max_similarity.astype(cp.float32, copy=False),
            }
        )
        # Write results
        write_start = time.perf_counter()
        self.write_parquet(
            points_to_remove_df,
            output_path,
            storage_options=self.output_storage_options,
            index=False,
            **self.write_kwargs,
        )
        write_time = time.perf_counter() - write_start
        self._log_metrics(
            {
                "pairwise_footer_scan_time": footer_time,
                "pairwise_read_time": read_time,
                "pairwise_rank_time": rank_time,
                "pairwise_conversion_time": conversion_time,
                "pairwise_compute_time": compute_time,
                "pairwise_write_time": write_time,
            }
        )

        if self.verbose:
            logger.debug(
                f"Pairwise computation for cluster {cluster_id} with {num_rows} rows done in "
                f"{time.perf_counter() - process_started:.2f} seconds"
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            _metadata={**task._metadata, "centroid_id": cluster_id},
            _stage_perf=task._stage_perf,
            data=[output_path],
        )


@dataclass
class PairwiseStage(CompositeStage[EmptyTask, FileGroupTask]):
    """Pairwise similarity stage for semantic deduplication."""

    # Required parameters
    id_field: str
    embedding_field: str
    input_path: str  # Path to kmeans output
    output_path: str
    # Ranking strategy
    ranking_strategy: RankingStrategy | None = None

    # Optional parameters
    pairwise_batch_size: int = 1024
    verbose: bool = False
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Ranking (for backward compatibility)
    which_to_keep: Literal["hard", "easy", "random"] = "hard"
    sim_metric: Literal["cosine", "l2"] = "cosine"
    random_seed: int = 42
    compute_dtype: PairwiseComputeDtype = "float32"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        validate_pairwise_batch_size(self.pairwise_batch_size)
        validate_pairwise_compute_dtype(self.compute_dtype)
        if self.ranking_strategy is None:
            if self.which_to_keep == "random":
                self.ranking_strategy = RankingStrategy(
                    metadata_cols=[], strategy="random", random_seed=self.random_seed
                )
            else:
                if self.sim_metric not in {"cosine", "l2"}:
                    msg = f"Invalid similarity metric: {self.sim_metric}. Only 'cosine' and 'l2' are supported."
                    raise ValueError(msg)
                if self.which_to_keep not in {"hard", "easy"}:
                    msg = f"Invalid which_to_keep value: {self.which_to_keep}. Supported: 'hard', 'easy', 'random'"
                    raise ValueError(msg)
                distance_col = "cosine_dist_to_cent" if self.sim_metric == "cosine" else "l2_dist_to_cent"
                # Determine sort order for ranking within cluster:
                # - "hard": Keep outliers farthest from centroid (descending distance, i.e., ascending=False)
                # - "easy": Keep representatives closest to centroid (ascending distance, i.e., ascending=True)
                # - "random": Handled above, not used here
                ascending = False if self.which_to_keep == "hard" else True  # noqa: SIM211

                # For distance-based ranking, explicitly add ID column as tie-breaker to maintain
                # compatibility with original semantic deduplication behavior
                self.ranking_strategy = RankingStrategy(
                    metadata_cols=[distance_col, self.id_field],
                    ascending=[ascending, ascending],  # Same sort order for both distance and ID
                )

    def decompose(self) -> list[ProcessingStage]:
        return [
            ClusterWiseFilePartitioningStage(
                input_path=self.input_path,
                storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs else None,
            ),
            PairwiseCosineSimilarityStage(
                id_field=self.id_field,
                embedding_field=self.embedding_field,
                output_path=self.output_path,
                pairwise_batch_size=self.pairwise_batch_size,
                verbose=self.verbose,
                ranking_strategy=self.ranking_strategy,
                compute_dtype=self.compute_dtype,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
            ),
        ]
