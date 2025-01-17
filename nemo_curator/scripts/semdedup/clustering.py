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
from datetime import datetime

import dask_cudf

from nemo_curator import ClusteringModel
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    semdedup_config = SemDedupConfig.from_yaml(args.config_file)
    client = get_client(**ArgumentHelper.parse_client_args(args))
    save_folder = os.path.join(
        semdedup_config.cache_dir, semdedup_config.clustering_save_loc
    )
    expand_outdir_and_mkdir(save_folder)
    # Initialize logger
    log_file = os.path.join(save_folder, "compute_centroids.log")

    logger = create_logger(
        rank=0,
        log_file=log_file,
        log_level=logging.INFO,
        name="logger-compute-centroids",
        stdout=True,
    )

    client = get_client(**ArgumentHelper.parse_client_args(args))
    dt1 = datetime.now()
    print("Start time:", dt1)

    embedding_fp = os.path.join(
        semdedup_config.cache_dir, semdedup_config.embeddings_save_loc
    )
    clustering_output_dir = os.path.join(
        semdedup_config.cache_dir, semdedup_config.clustering_save_loc
    )

    # Switch to https://github.com/NVIDIA/NeMo-Curator/issues/50
    # When we fix that
    embedding_df = dask_cudf.read_parquet(embedding_fp, blocksize="2GB")
    embedding_dataset = DocumentDataset(embedding_df)

    clustering_model = ClusteringModel(
        id_column=args.id_column,
        max_iter=semdedup_config.max_iter,
        n_clusters=semdedup_config.n_clusters,
        clustering_output_dir=clustering_output_dir,
        logger=logger,
    )

    clustered_embeddings = clustering_model(embedding_dataset)
    clustered_embeddings.df.head(10)
    dt2 = datetime.now()
    elapse = dt2 - dt1
    print("End time:", dt2)
    print("elapse:", elapse)

    client.cancel(client.futures, force=True)
    client.close()


def attach_args():
    parser = ArgumentHelper.parse_semdedup_args(
        description=(
            "Performs clustering on the computed embeddings of a collection of documents. "
            "This script requires that the embeddings have been created beforehand using "
            "semdedup_extract_embeddings"
            "Input arguments include: "
            "--id-column for the identifier in the dataset, "
            "--config-file for the path to the semantic deduplication configuration file. "
            "Important configuration parameters include: "
            " cache_dir for the directory to store cache,"
            " clustering_save_loc for the location to save clustering results,"
            " n_clusters for the number of clusters,"
            " seed for the seed for clustering,"
            " max_iter for the maximum iterations for clustering,"
            " kmeans_with_cos_dist for using K-Means with cosine distance."
        ),
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
