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
import random
import shutil
import time
from typing import List, Optional, Tuple

import cudf
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from dask.distributed import progress

from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


def assign_and_sort_clusters(
    id_col: str,
    kmeans_centroids_file: str,
    nearest_cent_dir: str,
    output_sorted_clusters_dir: str,
    cluster_ids: List[int],
    embedding_col: str,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = True,
    logger: Optional[logging.Logger] = None,
    profile_dir: Optional[str] = None,
):
    """
    Args:
        id_col (str): The column name representing the unique identifier for each data point.
        centroids_path (str): The location of the K-means centroids file.
        nearest_cent_dir (str): The location of the nearest center files.
        output_sorted_clusters_dir (str): The location to save the sorted clusters.
        sim_metric (str): The similarity metric to use for clustering. Defaults to "cosine".
        keep_hard (bool): When True, sorts cluster items in descending order by similarity to the cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to True.
        sorted_clusters_file_loc (str): The location to save the sorted clusters file. Defaults to an empty string.
        cluster_ids (list): The range of cluster IDs to sort.
        logger (logging.Logger): A logger object to log messages. Defaults to None.
        profile_dir (str): If specified directory to write dask profile. Default is None.

    Returns:
        None
    """
    # Step 3: Sort each class/cluster
    logger.info("Ranking...")
    if os.path.exists(output_sorted_clusters_dir):
        logger.info(
            f"Removing existing sorted cluster directory: {output_sorted_clusters_dir}"
        )
        shutil.rmtree(output_sorted_clusters_dir)

    expand_outdir_and_mkdir(output_sorted_clusters_dir)

    kmeans_centroids = np.load(kmeans_centroids_file)
    start_time = time.time()

    with performance_report_if_with_ts_suffix(
        profile_dir,
        "ranking-clusters",
    ):
        cluster_ids_bag = db.from_sequence(cluster_ids, npartitions=len(cluster_ids))
        completed_count = cluster_ids_bag.map(
            lambda cluster_c: rank_within_cluster(
                id_col=id_col,
                nearest_cent_dir=nearest_cent_dir,
                output_sorted_clusters_dir=output_sorted_clusters_dir,
                centroids=kmeans_centroids,
                embedding_col=embedding_col,
                sim_metric=sim_metric,
                keep_hard=keep_hard,
                kmeans_with_cos_dist=kmeans_with_cos_dist,
                cluster_ids=[cluster_c],
            )
        ).compute()

        missing = len(cluster_ids) - sum(completed_count)
    logger.info(
        f"Completed {sum(completed_count)} clusters. Missing {missing} clusters."
    )
    logger.info(f"Time taken for Ranking Clusters: {time.time() - start_time}")
    logger.info("DONE!")


def rank_within_cluster(
    id_col: str,
    nearest_cent_dir: str,
    output_sorted_clusters_dir: str,
    centroids: np.ndarray,
    embedding_col: str,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = False,
    cluster_ids: List[int] = range(50000),
):
    """
    Sorts each cluster's items by their distance to the cluster centroid.

    Args:
        id_col (str): The column name representing the unique identifier for each data point.
        nearest_cent_dir (str): The location of the nearest center files.
        output_sorted_clusters_dir (str): The location to save the sorted clusters.
        centroids (np.ndarray): The centroids for each cluster.
        sim_metric (str): The similarity metric used to compute distances. Should be one of ["cosine"]. Defaults to "cosine".
        keep_hard (bool): When True, sorts cluster items in descending order by similarity to the cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to False.
        cluster_ids (List[int]): The list of cluster IDs to process. Defaults to range(50000).

    Returns:
        None
    """
    assert sim_metric in [
        "cosine",
    ], "sim_metric should be in ['cosine']"

    missing_files = 0
    for cluster_c in cluster_ids:
        cluster_c_path = os.path.join(nearest_cent_dir, f"nearest_cent={cluster_c}")
        if not os.path.exists(cluster_c_path):
            missing_files += 1
            continue

        cluster_df = cudf.read_parquet(
            cluster_c_path, columns=[id_col, "dist_to_cent", embedding_col]
        )
        embeds = torch.as_tensor(
            cluster_df[embedding_col].list.leaves.values.reshape(
                cluster_df.shape[0], -1
            ),
            device="cuda",
        )
        cluster_df = cluster_df.to_pandas()

        assert kmeans_with_cos_dist is False

        if sim_metric == "cosine":
            cluster_c_centroid = torch.as_tensor(centroids[cluster_c], device="cuda")
            sim_to_cent = torch.nn.CosineSimilarity(dim=1)(embeds, cluster_c_centroid)
            sim_to_cent = sim_to_cent.cpu().numpy()
            cluster_dists_to_cent = (1 - sim_to_cent).tolist()
        elif sim_metric == "l2":
            # Used when kmeans_with_cos_dist is True
            cluster_dists_to_cent = list(cluster_df["dist_to_cent"])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        example_id = list(cluster_df[id_col])
        sort_descending = keep_hard
        cluster_sorted = sorted(
            zip(example_id, cluster_dists_to_cent, cluster_label),
            key=lambda x: x[1],
            reverse=sort_descending,
        )  # -- sort_descending = True for descending sort

        sorted_cluster_file_path = os.path.join(
            output_sorted_clusters_dir, f"cluster_{cluster_c}.npy"
        )
        np.save(sorted_cluster_file_path, cluster_sorted)

    return len(cluster_ids) - missing_files


def _semdedup(
    cluster_reps: torch.Tensor, device: str
) -> Tuple[torch.Tensor, List[int]]:
    # compute pairwise cos sim between cluster items,
    # then replace to diagonal with zeros to ignore self similarity
    cluster_reps.to(device)
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    M1 = torch.max(triu_sim_mat, dim=0)[1].cpu().numpy().tolist()
    return M, M1


def get_cluster_reps(
    cluster_id: int,
    emb_by_clust_dir: str,
    id_col: str,
    embedding_col: str,
    sorted_ids: np.ndarray,
) -> torch.Tensor:
    cluster_i_path = os.path.join(emb_by_clust_dir, f"nearest_cent={cluster_id}")
    cluster_reps = cudf.read_parquet(
        cluster_i_path, columns=[embedding_col, id_col]
    ).sort_values(by=id_col)
    num = cluster_reps.shape[0]

    df_ = pd.DataFrame(
        {"sorted_ids": sorted_ids, "inverse_sort": list(range(num))}
    ).sort_values(by="sorted_ids")
    cluster_reps["inverse_sort_id"] = df_["inverse_sort"].values
    cluster_reps = cluster_reps.sort_values(by="inverse_sort_id")

    cluster_reps = torch.as_tensor(
        cluster_reps[embedding_col].list.leaves.values.reshape(len(cluster_reps), -1),
        device="cuda",
    )
    return cluster_reps


def get_semantic_matches_per_cluster(
    cluster_id: int,
    emb_by_clust_dir: str,
    sorted_clusters_dir: str,
    id_col: str,
    id_col_type: str,
    eps_list: List[float],
    output_dir: str,
    embedding_col: str,
    which_to_keep: str,
) -> None:

    output_df_file_path = os.path.join(output_dir, f"cluster_{cluster_id}.parquet")

    sorted_file = os.path.join(sorted_clusters_dir, f"cluster_{cluster_id}.npy")
    if not os.path.exists(sorted_file):
        logging.info(f"{sorted_file} does not exist. Continue")
        return

    cluster_i = np.load(sorted_file)
    cluster_size = cluster_i.shape[0]
    logging.info(f"{cluster_id}: cluster_size: {cluster_size}")

    if cluster_size == 1:
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = [0]
        for eps in eps_list:
            points_to_remove_df[f"eps={eps}"] = [False]
        points_to_remove_df.to_parquet(output_df_file_path)
        return

    clutser_items_indices = list(range(cluster_size))

    which_to_keep = which_to_keep.lower()
    if which_to_keep == "random":
        random.shuffle(clutser_items_indices)
        cluster_i = cluster_i[clutser_items_indices]
    elif which_to_keep == "easy":
        clutser_items_indices = clutser_items_indices[::-1]
        cluster_i = cluster_i[clutser_items_indices]

    text_ids = cluster_i[:, 0].astype(id_col_type)

    cluster_reps = get_cluster_reps(
        cluster_id, emb_by_clust_dir, id_col, embedding_col, text_ids
    )
    M, M1 = _semdedup(cluster_reps, "cuda")
    assert cluster_reps.shape[0] == len(text_ids)

    M1_id = [text_ids[m] for m in M1]

    points_to_remove_df = cudf.DataFrame()
    points_to_remove_df["indices"] = clutser_items_indices
    points_to_remove_df["id"] = text_ids
    points_to_remove_df["max_id"] = M1_id
    points_to_remove_df["cosine_sim_score"] = M.numpy().tolist()

    for eps in eps_list:
        eps_points_to_remove = M > 1 - eps
        points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

    points_to_remove_df.to_parquet(output_df_file_path)


def get_num_records(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "rb") as f:
        # Read the header of the npy file
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape[0]


def _get_empty_results_df(id_col, id_col_type):
    meta_df = pd.DataFrame(
        {
            id_col: np.empty(0, dtype="int64"),
            "dist": np.empty(0, dtype="float32"),
            "cluster": np.empty(0, dtype="int32"),
        }
    )
    meta_df[id_col] = meta_df[id_col].astype(id_col_type)
    return meta_df


def prune_single_cluster(
    cluster_id: int,
    id_col: str,
    id_col_type: str,
    sorted_clusters_dir: str,
    semdedup_pruning_tables_dir: str,
    eps: float,
) -> cudf.DataFrame:
    """
    Processes data for a single cluster, applying pruning based on specified epsilon.

    Args:
        cluster_id (int): The specific cluster ID to process.
        id_col (str): The name of the ID column.
        id_col_type (str): The data type of the ID column.
        sorted_clusters_dir (str): Path to the sorted clusters directory.
        semdedup_pruning_tables_dir (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.

    Returns:
        cudf.DataFrame: A DataFrame of the pruned cluster data
    """
    sorted_fname = os.path.join(sorted_clusters_dir, f"cluster_{cluster_id}.npy")
    if not os.path.exists(sorted_fname):
        return _get_empty_results_df(id_col, id_col_type)

    cluster_data = np.load(sorted_fname)
    df_cluster = cudf.DataFrame(
        {
            id_col: cluster_data[:, 0],
            "dist": cluster_data[:, 1],
            "cluster": cluster_data[:, 2],
        }
    )

    df_cluster[id_col] = df_cluster[id_col].astype(id_col_type)
    df_cluster["dist"] = df_cluster["dist"].astype("float32")
    df_cluster["cluster"] = df_cluster["cluster"].astype("int32")

    cluster_df_fname = os.path.join(
        semdedup_pruning_tables_dir, f"cluster_{cluster_id}.parquet"
    )
    pruning_table = cudf.read_parquet(cluster_df_fname)
    if pruning_table.shape[0] == 1:
        return df_cluster

    # TODO: Fix this without going to host
    items_to_keep = (
        pruning_table[pruning_table[f"eps={eps}"] == False]["id"].to_arrow().to_pylist()
    )
    pruned_cluster = df_cluster[df_cluster[id_col].isin(items_to_keep)]
    pruned_cluster[id_col] = pruned_cluster[id_col].astype(id_col_type)
    return pruned_cluster


def extract_pruned_data(
    id_col: str,
    id_col_type: str,
    sorted_clusters_dir: str,
    semdedup_pruning_tables_dir: str,
    eps: float,
    n_clusters: int,
    output_parquet_path: str,
    logger: Optional[logging.Logger] = None,
    profile_dir: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Extracts pruned data from sorted clusters and saves it to a CSV file.

    Args:
        id_col (str): The name of the ID column.
        id_col_type (str): The data type of the ID column.
        sorted_clusters_dir (str): Path to the sorted clusters directory.
        semdedup_pruning_tables_dir (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.
        n_clusters (int): Number of clusters.
        output_csv_path (str): Path to save the output CSV file.
        logger (Optional[logging.Logger]): Logger object or path to store logs, defaults to None.
        profile_dir (str): If specified directory to write dask profile. Default is None.

    Returns:
        Tuple[int, int, int]: Number of kept records, removed records, and total records.
    """

    t0 = time.time()

    with performance_report_if_with_ts_suffix(
        profile_dir,
        "extracting-pruned-from-clusters",
    ):
        results_df = dd.from_map(
            prune_single_cluster,
            range(n_clusters),
            id_col=id_col,
            id_col_type=id_col_type,
            sorted_clusters_dir=sorted_clusters_dir,
            semdedup_pruning_tables_dir=semdedup_pruning_tables_dir,
            eps=eps,
        )
        results_df[id_col] = results_df[id_col].astype(id_col_type)
        results_df = results_df.persist()
        progress(results_df)

        results_df.to_parquet(output_parquet_path)
    if logger:
        logger.info(
            f"Time taken for Extracting Pruned Data : {time.time() - t0} and output written at {output_parquet_path}"
        )

    total_kept = len(results_df)

    np_files = [
        os.path.join(sorted_clusters_dir, f"cluster_{i}.npy") for i in range(n_clusters)
    ]
    total_records = sum(get_num_records(file_path) for file_path in np_files)
    # Aggregate results
    total_removed = total_records - total_kept
    return total_kept, total_removed, total_records


def extract_dedup_data(
    eps,
    n_clusters,
    id_col,
    id_col_type,
    sorted_clusters_dir,
    semdedup_pruning_tables_dir,
    output_summary_file,
    output_parquet_path,
    logger: logging.Logger,
    profile_dir: Optional[str] = None,
) -> dd.DataFrame:
    """
    Extracts deduplicated data based on provided parameters and logs the process.

    Args:

    """

    kept, removed, total = extract_pruned_data(
        id_col=id_col,
        id_col_type=id_col_type,
        sorted_clusters_dir=sorted_clusters_dir,
        semdedup_pruning_tables_dir=semdedup_pruning_tables_dir,
        eps=eps,
        n_clusters=n_clusters,
        output_parquet_path=output_parquet_path,
        logger=logger,
        profile_dir=profile_dir,
    )

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

    fps = [
        os.path.join(output_parquet_path, file_name)
        for file_name in os.listdir(output_parquet_path)
    ]
    ids_to_keep_df = dd.from_map(cudf.read_parquet, fps)
    return ids_to_keep_df
