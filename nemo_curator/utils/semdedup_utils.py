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


import logging
import os
from typing import Literal, Tuple

import cudf
import cupy as cp
import dask.dataframe as dd
import pandas as pd
import torch
from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar

L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"


def normalize_embeddings_col_in_df(
    df: cudf.DataFrame, embedding_col: str
) -> cudf.DataFrame:
    tensor = torch.Tensor(get_array_from_df(df, embedding_col))
    normalized_tensor = tensor / torch.norm(tensor, dim=1, keepdim=True)
    df[embedding_col] = create_list_series_from_1d_or_2d_ar(
        cp.asarray(normalized_tensor), index=df.index
    )
    return df


def get_array_from_df(df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def add_l2_cosine_dist_to_centroid(
    df: cudf.DataFrame, embedding_col: str, centroids: cp.ndarray
) -> cudf.DataFrame:
    """
    Computes the L2 distance to nearest centroid to each embedding in the DataFrame.
    Embeddings are normalized. For cosine we'll need to normalize the centroids as well.
    """
    normalized_embeddings = get_array_from_df(df, embedding_col)
    centroids_ar = centroids[df["nearest_cent"].values]
    dist_to_cents = cp.sqrt(cp.sum((normalized_embeddings - centroids_ar) ** 2, axis=1))
    df[L2_DIST_TO_CENT_COL] = dist_to_cents
    del centroids_ar

    centroids_norm = centroids / cp.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_ar = centroids_norm[df["nearest_cent"].values]
    # We normalize the centroids as well
    cosine_similarities = cp.sum(normalized_embeddings * centroids_ar, axis=1)
    df[COSINE_DIST_TO_CENT_COL] = 1 - cosine_similarities
    return df


def pairwise_cosine_similarity(
    cluster_reps: torch.Tensor,
    device: Literal["cuda", "cpu"],
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute pairwise cosine similarity between cluster items,
    then replace to diagonal with zeros to ignore self similarity
    """
    # Move to device
    cluster_reps = cluster_reps.to(device)
    # Compute pairwise cosine similarity
    pairwise_sim_matrix = torch.mm(cluster_reps, cluster_reps.T)
    del cluster_reps
    # Get upper triangular matrix
    assert pairwise_sim_matrix.shape[0] == pairwise_sim_matrix.shape[1]
    triu_sim_mat = torch.triu(pairwise_sim_matrix, diagonal=1)
    # Get max similarity and indices
    max_values_and_indices = torch.max(triu_sim_mat, dim=0)
    max_similarity = max_values_and_indices[0]
    max_indices = max_values_and_indices[1]
    return cp.asarray(max_similarity, dtype=cp.float32), cp.asarray(max_indices)


def pairwise_cosine_similarity_batched(
    cluster_reps: torch.Tensor,
    device: Literal["cuda", "cpu"],
    batch_size: int = 1024,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Computes pairwise cosine similarity between cluster items,
    then replace to diagonal with zeros to ignore self similarity.
    This function is useful for large clusters where the pairwise similarity matrix
    does not fit into memory.
    We use a batched approach to compute the pairwise similarity matrix in batches.
    Memory requirements are O(N*B) where N is the number of items in the cluster and B is the batch size
    instead of O(N^2) for the full matrix.
    """
    cluster_reps = cluster_reps.to(device)
    max_similarity = torch.zeros(
        cluster_reps.shape[0], dtype=torch.float32, device=device
    )
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

    return cp.asarray(max_similarity), cp.asarray(max_indices)


def get_semantic_matches_per_cluster(
    cluster_id: int,
    emb_by_clust_dir: str,
    id_col: str,
    output_dir: str,
    embedding_col: str,
    which_to_keep: Literal["hard", "easy", "random"],
    sim_metric: Literal["cosine", "l2"],
    batched_cosine_similarity: int = 1024,
) -> None:
    """
    Get the semantic matches for a single cluster.
    Reads the cluster embeddings and then computes pairwise cosine similarity between them.
    """
    if sim_metric == "cosine":
        distance_col = COSINE_DIST_TO_CENT_COL
    elif sim_metric == "l2":
        distance_col = L2_DIST_TO_CENT_COL
    else:
        msg = f"Invalid similarity metric: {sim_metric}. Only cosine and l2 are supported."
        raise ValueError(msg)

    cluster_df = cudf.read_parquet(
        os.path.join(emb_by_clust_dir, f"nearest_cent={cluster_id}"),
        columns=[embedding_col, id_col, distance_col],
    )
    output_df_file_path = os.path.join(output_dir, f"cluster_{cluster_id}.parquet")
    if len(cluster_df) == 1:
        cluster_df["id"] = cluster_df[id_col]
        cluster_df["max_id"] = cluster_df[id_col]
        cluster_df["cosine_sim_score"] = [0]
        cluster_df = cluster_df[["id", "max_id", "cosine_sim_score"]]
        cluster_df.to_parquet(output_df_file_path)
        return

    if which_to_keep == "hard":
        cluster_df = cluster_df.sort_values(
            by=[distance_col, id_col], ascending=False, ignore_index=True
        )
    elif which_to_keep == "easy":
        cluster_df = cluster_df.sort_values(
            by=[distance_col, id_col], ascending=True, ignore_index=True
        )
    elif which_to_keep == "random":
        cluster_df = cluster_df.sample(frac=1, random_state=42, ignore_index=True)

    cluster_embeddings = torch.as_tensor(
        get_array_from_df(cluster_df, embedding_col), device="cuda"
    )
    ids = cluster_df[id_col]
    assert cluster_embeddings.shape[0] == len(ids)

    if batched_cosine_similarity > 0:
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            cluster_embeddings, "cuda", batched_cosine_similarity
        )
    else:
        max_similarity, max_indices = pairwise_cosine_similarity(
            cluster_embeddings, "cuda"
        )
    max_indices_id = ids.iloc[max_indices].reset_index(drop=True)
    points_to_remove_df = cudf.DataFrame(
        {
            "id": ids,
            "max_id": max_indices_id,
            "cosine_sim_score": max_similarity,
        }
    )
    points_to_remove_df.to_parquet(output_df_file_path)


def prune_single_cluster(
    cluster_id: int,
    id_col: str,
    emb_by_clust_dir: str,
    semdedup_pruning_tables_dir: str,
    eps: float,
) -> cudf.DataFrame:
    """
    Processes data for a single cluster, applying pruning based on specified epsilon.

    Args:
        cluster_id (int): The specific cluster ID to process.
        id_col (str): The name of the ID column.
        emb_by_clust_dir (str): Path to where clustered embeddings are stored.
        semdedup_pruning_tables_dir (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.

    Returns:
        cudf.DataFrame: A DataFrame of the pruned cluster data
    """
    cluster_dir = os.path.join(emb_by_clust_dir, f"nearest_cent={cluster_id}")
    # For the output we only return id, cosine_dist_to_cent, and cluster
    df_cluster = cudf.read_parquet(
        cluster_dir, columns=[id_col, COSINE_DIST_TO_CENT_COL]
    ).assign(cluster=cluster_id)

    pruning_table_fname = os.path.join(
        semdedup_pruning_tables_dir, f"cluster_{cluster_id}.parquet"
    )
    # In future we can add more columns to the pruning table like max_id etc.
    pruning_table = cudf.read_parquet(
        pruning_table_fname, columns=["id", "cosine_sim_score"]
    )
    # If the pruning table only has one row, we don't need to remove any records
    if len(pruning_table) == 1:
        # Create empty dataframe with same schema / dtypes as df_cluster
        empty_df = cudf.DataFrame(columns=df_cluster.columns).astype(df_cluster.dtypes)
        return empty_df
    # We keep only records that are very similar i.e cosine_sim_score >= 1 - eps
    pruning_table = pruning_table[pruning_table["cosine_sim_score"] >= 1 - eps][["id"]]
    # In future we can avoid this merge if we add more columns to the pruning table
    # However that might increase memory consumption at that stage, keeping it as is for now
    return df_cluster.merge(
        pruning_table.rename(columns={"id": id_col}), on=id_col, how="inner"
    )


def write_pruned_summary_file(
    eps: float,
    emb_by_clust_dir: str,
    filtered_unique_ids_path: str,
    output_summary_file: str,
    logger: logging.Logger,
):
    """
    Writes a summary file for the pruned data.
    """
    removed = len(dd.read_parquet(filtered_unique_ids_path))
    total = len(dd.read_parquet(emb_by_clust_dir))
    kept = total - removed

    logger.info(
        f"DONE saving {kept} out of {total}. Removed: {removed}. Epsilon: {eps:.4f}"
    )
    result_dict = {
        "eps": [eps],
        "kept": [kept],
        "removed": [removed],
        "total": [total],
    }
    df = pd.DataFrame(result_dict)
    df.to_csv(output_summary_file, index=False)
