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
from dataclasses import dataclass
from typing import Any, Literal

import cudf
import cupy as cp
import numpy as np
import torch
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, _EmptyTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs

from .pairwise_io import ClusterWiseFilePartitioningStage
from .ranking import RankingStrategy
from .utils import break_parquet_partition_into_groups, get_array_from_df


def pairwise_cosine_similarity_batched(
    cluster_reps: "torch.Tensor",
    batch_size: int = 1024,
) -> tuple["cp.ndarray", "cp.ndarray"] | tuple[np.ndarray, np.ndarray]:
    """
    Computes pairwise cosine similarity between cluster items,
    then replace to diagonal with zeros to ignore self similarity.
    This function is useful for large clusters where the pairwise similarity matrix
    does not fit into memory.
    We use a batched approach to compute the pairwise similarity matrix in batches.
    Memory requirements are O(N*B) where N is the number of items in the cluster and B is the batch size
    instead of O(N^2) for the full matrix.

    TODO: In future we can estimate memory requirement and calculate batch size dynamically.
    """
    device = "cuda"

    cluster_reps = cluster_reps.to(device)
    max_similarity = torch.zeros(cluster_reps.shape[0], dtype=torch.float32, device=device)
    max_indices = torch.zeros(cluster_reps.shape[0], dtype=torch.int64, device=device)
    for start_idx in range(0, cluster_reps.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, cluster_reps.shape[0])
        batch = cluster_reps[start_idx:end_idx]
        pairwise_sim_matrix = torch.mm(cluster_reps, batch.T)
        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1 - start_idx)
        del batch, pairwise_sim_matrix
        max_values_and_indices = torch.max(triu_sim_matrix, dim=0)
        max_similarity[start_idx:end_idx] = max_values_and_indices[0]
        max_indices[start_idx:end_idx] = max_values_and_indices[1]

    if device == "cuda":
        return cp.asarray(max_similarity), cp.asarray(max_indices)
    else:
        # convert to numpy arrays
        return max_similarity.numpy(), max_indices.numpy()


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
        embedding_dim: int | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the pairwise cosine similarity stage.

        Args:
            id_field: The column name of the id column.
            embedding_field: The column name of the embedding column.
            output_path: The path to the output directory.
            ranking_strategy: Strategy for ranking/sorting clusters before similarity computation.
            pairwise_batch_size: Batch size for pairwise similarity computation.
            verbose: Whether to print verbose output.
            embedding_dim: Embedding dimension for memory estimation.
            read_kwargs: Kwargs for reading parquet files.
            write_kwargs: Kwargs for writing parquet files.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        self.pairwise_batch_size = pairwise_batch_size
        self.embedding_dim = embedding_dim
        self.ranking_strategy = ranking_strategy
        self.verbose = verbose
        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}
        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["index"])
        self.input_storage_options = self.read_kwargs.pop("storage_options", None) if self.read_kwargs else None
        self.output_storage_options = self.write_kwargs.pop("storage_options", None) if self.write_kwargs else None
        self._name = "PairwiseCosineSimilarityStage"
        self._resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process a PairwiseFileGroupTask to compute pairwise similarities."""
        if task._metadata.get("filetype") != "parquet":
            msg = f"PairwiseCosineSimilarityStage only supports parquet files, got {task._metadata.get('filetype')}"
            raise ValueError(msg)

        cluster_id = task._metadata.get("centroid_id")
        output_path = os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")
        if cluster_id is None:
            msg = "centroid_id not found in task metadata"
            raise ValueError(msg)

        t1 = time.perf_counter()

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

        if not dfs:
            logger.warning(f"No data found for cluster {cluster_id}")
            return FileGroupTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
                data=[],
            )

        num_rows = sum(len(df) for df in dfs)

        # Handle single item clusters
        if num_rows == 1:
            result_df = cudf.DataFrame(
                {
                    "id": dfs[0][self.id_field],
                    "max_id": dfs[0][self.id_field],
                    "cosine_sim_score": cudf.Series([0], dtype="float32"),
                }
            )
            self.write_parquet(
                result_df, output_path, storage_options=self.output_storage_options, index=False, **self.write_kwargs
            )
            return FileGroupTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata={
                    **task._metadata,
                    "centroid_id": cluster_id,
                },
                _stage_perf=task._stage_perf,
                data=[os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")],
            )

        # Cannot concatenate dataframes with embeddings due to cudf 2bn row limit
        # Instead, concatenate metadata columns and handle embeddings separately
        metadata_dfs, embedding_arrays = [], []
        for df in dfs:
            metadata_dfs.append(df[metadata_cols])
            embedding_arrays.append(get_array_from_df(df, self.embedding_field))

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
        max_similarity, max_indices = pairwise_cosine_similarity_batched(cluster_embeddings, self.pairwise_batch_size)

        # Convert indices back to IDs
        max_indices_id = ids.iloc[max_indices].reset_index(drop=True)

        # Create result dataframe
        points_to_remove_df = cudf.DataFrame(
            {
                "id": ids,
                "max_id": max_indices_id,
                "cosine_sim_score": max_similarity,
            }
        )

        # Write results
        self.write_parquet(
            points_to_remove_df,
            output_path,
            storage_options=self.output_storage_options,
            index=False,
            **self.write_kwargs,
        )

        t2 = time.perf_counter()
        if self.verbose:
            logger.debug(
                f"Pairwise computation for cluster {cluster_id} with {num_rows} rows done in {(t2 - t1):.2f} seconds"
            )

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata={**task._metadata, "centroid_id": cluster_id},
            _stage_perf=task._stage_perf,
            data=[output_path],
        )


@dataclass
class PairwiseStage(CompositeStage[_EmptyTask, FileGroupTask]):
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
    pairwise_batch_size: int = 1024
    verbose: bool = False
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Ranking (for backward compatibility)
    which_to_keep: Literal["hard", "easy", "random"] = "hard"
    sim_metric: Literal["cosine", "l2"] = "cosine"
    random_seed: int = 42

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
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
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
            ),
        ]
