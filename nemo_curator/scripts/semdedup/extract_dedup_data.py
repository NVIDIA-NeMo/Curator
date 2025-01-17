import logging
import os
from datetime import datetime

from nemo_curator import SemanticClusterLevelDedup
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    semdedup_config = SemDedupConfig.from_yaml(args.config_file)
    client = get_client(**ArgumentHelper.parse_client_args(args))

    root = semdedup_config.cache_dir
    save_loc = semdedup_config.clustering_save_loc
    client = get_client(**ArgumentHelper.parse_client_args(args))

    logger = create_logger(
        rank=0,
        log_file=os.path.join(root, save_loc, "extract_dedup_data.log"),
        name="logger-extract-dedup-data",
        log_level=logging.INFO,
        stdout=True,
    )

    dt1 = datetime.now()
    logger.info(f"Start: {dt1}")
    cache_dir = semdedup_config.cache_dir
    semantic_dedup = SemanticClusterLevelDedup(
        n_clusters=semdedup_config.n_clusters,
        emb_by_clust_dir=os.path.join(
            cache_dir, semdedup_config.clustering_save_loc, "embs_by_nearest_center"
        ),
        sorted_clusters_dir=os.path.join(
            cache_dir, semdedup_config.clustering_save_loc, "sorted"
        ),
        id_column=args.id_column,
        id_column_type=args.id_column_type,
        which_to_keep=semdedup_config.which_to_keep,
        output_dir=os.path.join(
            semdedup_config.cache_dir, semdedup_config.clustering_save_loc
        ),
        logger=logger,
    )

    semantic_dedup.compute_semantic_match_dfs(semdedup_config.eps_thresholds)
    for eps in semdedup_config.eps_thresholds:
        dedup_id_dataset = semantic_dedup.extract_dedup_data(eps_to_extract=eps)
        print(dedup_id_dataset.df.head(10))

    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")

    client.cancel(client.futures, force=True)
    client.close()


def attach_args():
    parser = ArgumentHelper.parse_semdedup_args(
        description=(
            "Extracts deduplicated data from the clustered embeddings of a collection of documents. "
            "This script requires that embeddings and clustering have been performed "
            "earlier using semdedup_extract_embeddings and semdedup_cluster_embeddings."
            "Input arguments include: "
            "--id-column for the the identifier in the dataset, "
            "--id-column-type for the data type of ID column, "
            "--config-file for the path to the semantic deduplication configuration file. "
            "Important configuration parameters include:"
            " cache_dir for the directory to store cache"
            " which_to_keep for specifying which duplicates to keep,"
            " largest_cluster_size_to_process for the largest cluster size to process,"
            " sim_metric for the similarity metric for deduplication,"
            " eps_thresholds for epsilon thresholds to calculate if semantically similar or not"
            " and eps_to_extract for the epsilon value to extract deduplicated data."
        ),
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
