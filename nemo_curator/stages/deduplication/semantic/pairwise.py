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
import traceback
from dataclasses import dataclass
from typing import Any, Literal

import cudf
import cupy as cp
import numpy as np
import torch
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.gpu_utils import release_cached_gpu_memory
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs

from .pairwise_io import ClusterWiseFilePartitioningStage
from .ranking import RankingStrategy
from .utils import EmbeddingStorageDtype, break_parquet_partition_into_groups, decode_embedding_array

PairwiseBatchSize = int | Literal["auto"]
PairwiseComputeDtype = Literal["auto", "float16", "float32"]

_AUTO_BATCH_MEMORY_FRACTION = 0.8
_AUTO_BATCH_OVERHEAD_FACTOR = 1.25
_AUTO_BATCH_ALIGNMENT = 256


def validate_pairwise_batch_size(batch_size: object) -> None:
    if batch_size == "auto":
        return
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        msg = f"pairwise_batch_size must be a positive integer or 'auto', got {batch_size!r}"
        raise ValueError(msg)


def validate_pairwise_compute_dtype(compute_dtype: object) -> None:
    if compute_dtype not in {"auto", "float16", "float32"}:
        msg = f"Unsupported compute_dtype: {compute_dtype}"
        raise ValueError(msg)


def _resolve_compute_dtype(cluster_reps: "torch.Tensor", compute_dtype: PairwiseComputeDtype) -> "torch.dtype":
    validate_pairwise_compute_dtype(compute_dtype)
    if compute_dtype == "float16":
        return torch.float16
    if compute_dtype == "float32":
        return torch.float32
    if cluster_reps.dtype not in {torch.float16, torch.float32}:
        msg = f"compute_dtype='auto' requires float16 or float32 embeddings, got {cluster_reps.dtype}"
        raise TypeError(msg)
    return cluster_reps.dtype


def _estimate_pairwise_workspace_bytes(cluster_reps: "torch.Tensor", batch_size: int) -> int:
    """Estimate retained Pairwise tensors and per-batch reduction outputs."""
    num_rows, embedding_width = cluster_reps.shape
    compute_element_size = cluster_reps.element_size()
    output_bytes = num_rows * (compute_element_size + torch.int64.itemsize)
    similarity_bytes = num_rows * batch_size * compute_element_size
    query_bytes = batch_size * embedding_width * compute_element_size
    invalid_mask_bytes = batch_size * batch_size * torch.bool.itemsize
    reduction_bytes = batch_size * (compute_element_size + torch.int64.itemsize)
    return output_bytes + similarity_bytes + query_bytes + invalid_mask_bytes + reduction_bytes


def _resolve_pairwise_batch_size(cluster_reps: "torch.Tensor", batch_size: PairwiseBatchSize) -> int:
    """Resolve a fixed or memory-derived workspace width for ``cluster_reps``."""
    validate_pairwise_batch_size(batch_size)
    num_rows = cluster_reps.shape[0]
    if batch_size != "auto":
        return min(batch_size, num_rows)

    free_memory, _ = torch.cuda.mem_get_info(cluster_reps.device)
    memory_budget = int(free_memory * _AUTO_BATCH_MEMORY_FRACTION)
    if _estimate_pairwise_workspace_bytes(cluster_reps, 1) * _AUTO_BATCH_OVERHEAD_FACTOR > memory_budget:
        msg = (
            "Insufficient free GPU memory for one Pairwise similarity column: "
            f"free_memory={free_memory}, num_rows={num_rows}, embedding_width={cluster_reps.shape[1]}, "
            f"compute_dtype={cluster_reps.dtype}"
        )
        raise MemoryError(msg)

    low, high = 1, num_rows
    while low < high:
        candidate = (low + high + 1) // 2
        estimated_bytes = _estimate_pairwise_workspace_bytes(cluster_reps, candidate)
        if estimated_bytes * _AUTO_BATCH_OVERHEAD_FACTOR <= memory_budget:
            low = candidate
        else:
            high = candidate - 1
    max_batch_size = low
    if max_batch_size >= _AUTO_BATCH_ALIGNMENT:
        max_batch_size = max_batch_size // _AUTO_BATCH_ALIGNMENT * _AUTO_BATCH_ALIGNMENT
    return min(max_batch_size, num_rows)


def pairwise_cosine_similarity_batched(
    cluster_reps: "torch.Tensor",
    batch_size: PairwiseBatchSize = 1024,
    compute_dtype: PairwiseComputeDtype = "auto",
) -> tuple["cp.ndarray", "cp.ndarray"] | tuple[np.ndarray, np.ndarray]:
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
    resolved_batch_size = _resolve_pairwise_batch_size(cluster_reps, batch_size)

    num_rows, embedding_width = cluster_reps.shape
    max_similarity = torch.zeros(num_rows, dtype=cluster_reps.dtype, device=cluster_reps.device)
    max_indices = torch.zeros(num_rows, dtype=torch.int64, device=cluster_reps.device)
    pairwise_sim_workspace = torch.empty(
        (num_rows, resolved_batch_size),
        dtype=cluster_reps.dtype,
        device=cluster_reps.device,
    )
    batch_workspace = torch.empty(
        (resolved_batch_size, embedding_width),
        dtype=cluster_reps.dtype,
        device=cluster_reps.device,
    )
    invalid_batch_triangle = torch.ones(
        (resolved_batch_size, resolved_batch_size),
        dtype=torch.bool,
        device=cluster_reps.device,
    ).tril_()

    for start_idx in range(0, num_rows, resolved_batch_size):
        end_idx = min(start_idx + resolved_batch_size, num_rows)
        batch_width = end_idx - start_idx

        # In the conceptual query-by-candidate matrix above, only the strict lower
        # triangle is valid. torch.mm below stores its transpose (candidate rows by
        # query columns), so only the candidate prefix through this query block can
        # contain valid entries.
        candidate_reps = cluster_reps[:end_idx]
        batch_workspace[:batch_width].copy_(cluster_reps[start_idx:end_idx])
        if batch_width < resolved_batch_size:
            batch_workspace[batch_width:].zero_()
        torch.mm(candidate_reps, batch_workspace.T, out=pairwise_sim_workspace[:end_idx])
        pairwise_sim_matrix = pairwise_sim_workspace[:end_idx, :batch_width]

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
        pairwise_batch_size: PairwiseBatchSize = 1024,
        verbose: bool = False,
        embedding_dim: int | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        input_embedding_dtype: EmbeddingStorageDtype = "auto",
        compute_dtype: PairwiseComputeDtype = "auto",
        profile: bool = False,
    ):
        """Initialize the pairwise cosine similarity stage.

        Args:
            id_field: The column name of the id column.
            embedding_field: The column name of the embedding column.
            output_path: The path to the output directory.
            ranking_strategy: Strategy for ranking/sorting clusters before similarity computation.
            pairwise_batch_size: Positive fixed batch size or ``"auto"`` for memory-derived sizing.
            verbose: Whether to print verbose output.
            embedding_dim: Embedding dimension for memory estimation.
            input_embedding_dtype: Storage representation of embedding list leaves.
            compute_dtype: Multiplication dtype. ``"auto"`` retains the decoded embedding precision.
            profile: Whether to synchronize and record granular phase timings.
            read_kwargs: Kwargs for reading parquet files.
            write_kwargs: Kwargs for writing parquet files.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        validate_pairwise_batch_size(pairwise_batch_size)
        self.pairwise_batch_size = pairwise_batch_size
        self.embedding_dim = embedding_dim
        if input_embedding_dtype not in {"auto", "float16", "float32"}:
            msg = f"Unsupported input_embedding_dtype: {input_embedding_dtype}"
            raise ValueError(msg)
        self.input_embedding_dtype = input_embedding_dtype
        validate_pairwise_compute_dtype(compute_dtype)
        self.compute_dtype = compute_dtype
        self.profile = profile
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
        """Process one cluster and release cached allocations on every exit path."""
        try:
            return self._process(task)
        except BaseException as exc:
            # An exception's traceback otherwise keeps the unwound _process
            # frame—and its large GPU objects—alive until after this finalizer.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            release_cached_gpu_memory()

    def _process(self, task: FileGroupTask) -> FileGroupTask:  # noqa: C901, PLR0912, PLR0915
        """Process a PairwiseFileGroupTask to compute pairwise similarities."""
        if task._metadata.get("filetype") != "parquet":
            msg = f"PairwiseCosineSimilarityStage only supports parquet files, got {task._metadata.get('filetype')}"
            raise ValueError(msg)

        cluster_id = task._metadata.get("centroid_id")
        output_path = os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")
        if cluster_id is None:
            msg = "centroid_id not found in task metadata"
            raise ValueError(msg)

        if self.profile:
            torch.cuda.synchronize()
        process_started = time.perf_counter()
        read_started = process_started
        metrics: dict[str, float | int] = {}

        # Read all file groups and concatenate
        dfs = []
        num_rows = 0

        # Break input files into groups to avoid 2bn row limit
        file_groups = break_parquet_partition_into_groups(
            task.data, embedding_dim=self.embedding_dim, storage_options=self.input_storage_options
        )

        # Determine which columns to read based on ranking strategy
        additional_cols = self.ranking_strategy.metadata_cols if self.ranking_strategy.strategy == "sort" else []

        # We do the list(dict.fromkeys(...)) to remove duplicates from the list of columns to read, in case additional_cols contains self.id_field
        metadata_cols = list(dict.fromkeys([self.id_field, *additional_cols]))
        for file_group in file_groups:
            # Read required columns including metadata columns for ranking
            df = self.read_parquet(
                file_group,
                columns=[*metadata_cols, self.embedding_field],
                assign_id=False,
                storage_options=self.input_storage_options,
                **self.read_kwargs,
            )
            dfs.append(df)
            num_rows += len(df)
        if self.profile:
            torch.cuda.synchronize()
            prepare_started = time.perf_counter()
            metrics["pairwise_read_time"] = prepare_started - read_started

        if not dfs:
            logger.warning(f"No data found for cluster {cluster_id}")
            return FileGroupTask(
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
                data=[],
            )

        num_rows = sum(len(df) for df in dfs)

        # Cannot concatenate dataframes with embeddings due to cudf 2bn row limit
        # Instead, concatenate metadata columns and handle embeddings separately
        metadata_dfs, embedding_arrays = [], []
        for df in dfs:
            metadata_dfs.append(df[metadata_cols])
            embedding_arrays.append(decode_embedding_array(df, self.embedding_field, self.input_embedding_dtype))

        metadata_cluster_df = cudf.concat(metadata_dfs, ignore_index=True).reset_index(drop=True)

        # Add original index to track reordering
        metadata_cluster_df["_original_idx"] = metadata_cluster_df.index

        ranked_metadata_df = self.ranking_strategy.rank_cluster(metadata_cluster_df)
        # Get reorder indices from the ranked dataframe (TODO: we get it to CPU, but maybe we can do it on GPU todo)
        reorder_indices = ranked_metadata_df["_original_idx"].to_arrow().to_pylist()
        # Remove the helper column
        ranked_metadata_df = ranked_metadata_df.drop(columns=["_original_idx"])

        # Convert numpy arrays to torch tensors before concatenating
        concatenated_embeddings = torch.cat([torch.as_tensor(arr, device="cuda") for arr in embedding_arrays], dim=0)
        cluster_embeddings = concatenated_embeddings[reorder_indices]

        ids = ranked_metadata_df[self.id_field]

        # Compute pairwise similarities
        resolved_compute_dtype = _resolve_compute_dtype(cluster_embeddings, self.compute_dtype)
        cluster_embeddings = cluster_embeddings.to(device="cuda", dtype=resolved_compute_dtype)
        resolved_batch_size = _resolve_pairwise_batch_size(cluster_embeddings, self.pairwise_batch_size)
        if self.pairwise_batch_size == "auto":
            logger.info(
                f"Resolved Pairwise batch size to {resolved_batch_size} for cluster {cluster_id} "
                f"({num_rows} rows, {resolved_compute_dtype})"
            )
        if self.profile:
            torch.cuda.synchronize()
            similarity_started = time.perf_counter()
            metrics["pairwise_prepare_time"] = similarity_started - prepare_started
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            cluster_embeddings,
            resolved_batch_size,
        )
        if self.profile:
            torch.cuda.synchronize()
            output_started = time.perf_counter()
            metrics["pairwise_similarity_time"] = output_started - similarity_started
        # The O(N * B) workspace has died with the helper frame. Return its
        # cached blocks before cuDF materializes IDs and writes Parquet.
        torch.cuda.empty_cache()

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
        if self.profile:
            torch.cuda.synchronize()
            write_started = time.perf_counter()
            metrics["pairwise_output_time"] = write_started - output_started
        self.write_parquet(
            points_to_remove_df,
            output_path,
            storage_options=self.output_storage_options,
            index=False,
            **self.write_kwargs,
        )
        if self.profile:
            torch.cuda.synchronize()
            finished = time.perf_counter()
            metrics["pairwise_write_time"] = finished - write_started
        metrics.update(
            pairwise_num_rows=num_rows,
            pairwise_resolved_batch_size=resolved_batch_size,
        )
        self._log_metrics(metrics)
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
    embedding_dim: int | None = None
    pairwise_batch_size: PairwiseBatchSize = 1024
    verbose: bool = False
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Ranking (for backward compatibility)
    which_to_keep: Literal["hard", "easy", "random"] = "hard"
    sim_metric: Literal["cosine", "l2"] = "cosine"
    random_seed: int = 42
    input_embedding_dtype: EmbeddingStorageDtype = "auto"
    compute_dtype: PairwiseComputeDtype = "auto"
    profile: bool = False

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        validate_pairwise_batch_size(self.pairwise_batch_size)
        if self.input_embedding_dtype not in {"auto", "float16", "float32"}:
            msg = f"Unsupported input_embedding_dtype: {self.input_embedding_dtype}"
            raise ValueError(msg)
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
                embedding_dim=self.embedding_dim,
                input_embedding_dtype=self.input_embedding_dtype,
                compute_dtype=self.compute_dtype,
                profile=self.profile,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
            ),
        ]
