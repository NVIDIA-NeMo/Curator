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
import time

from nemo_curator import EmbeddingCreator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_remaining_files
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    semdedup_config = SemDedupConfig.from_yaml(args.config_file)
    client = get_client(**ArgumentHelper.parse_client_args(args))
    expand_outdir_and_mkdir(semdedup_config.cache_dir)
    logger = create_logger(
        rank=0,
        name="logger-compute-embeddings",
        log_file=os.path.join(semdedup_config.cache_dir, "compute_embeddings.log"),
        log_level=logging.INFO,
        stdout=True,
    )

    output_data_dir = os.path.join(
        semdedup_config.cache_dir, semdedup_config.embeddings_save_loc
    )

    # Some time jsonl files are stored as .json
    # So to handle that case we can pass the input_file_extension
    if args.input_file_extension is not None:
        input_file_extension = args.input_file_extension
    else:
        input_file_extension = args.input_file_type
    print("input_file_extension", input_file_extension)

    st = time.time()
    input_files = get_remaining_files(
        input_file_path=args.input_data_dir,
        output_file_path=output_data_dir,
        input_file_type=input_file_extension,
        output_file_type="parquet",
        num_files=semdedup_config.num_files,
    )

    logger.info(f"Processing {len(input_files)} files")
    if len(input_files) == 0:
        logger.info("No files to process")
        return

    ddf = read_data(
        input_files=input_files,
        file_type=args.input_file_type,
        add_filename=True,
    )
    ddf = ddf.reset_index(drop=True)
    dataset = DocumentDataset(ddf)

    # Can repartition here if needed
    # ddf = ddf.repartition(partition_size="64MB")
    embedding_creator = EmbeddingCreator(
        embedding_model_name_or_path=semdedup_config.embedding_model_name_or_path,
        embedding_batch_size=semdedup_config.embedding_batch_size,
        embedding_output_dir=os.path.join(
            semdedup_config.cache_dir, semdedup_config.embeddings_save_loc
        ),
        input_column=args.input_text_field,
        write_embeddings_to_disk=semdedup_config.write_embeddings_to_disk,
        logger=logger,
        write_to_filename=True,
    )

    embedding_dataset = embedding_creator(dataset=dataset)
    print(embedding_dataset.df.head())
    logger.info(f"Time taken: {time.time() - st}")
    client.cancel(client.futures, force=True)
    client.close()


def attach_args():
    parser = ArgumentHelper.parse_semdedup_args(
        description=(
            "Computes the embeddings of a collection of documents using the specified model. "
            'The model is specified in the configuration file using embedding_model_name_or_path (e.g. "sentence-transformers/paraphrase-MiniLM-L6-v2"). '
            "The embeddings are saved in the specified cache directory under the embeddings_save_loc directory. "
            "Input arguments include: "
            "--input-data-dir for the directory containing input data files, "
            '--input-file-type for the type of input files (e.g., "json", "csv"), '
            '--input-file-extension for specifying the file extension of input files (e.g., ".jsonl"), '
            "--input-text-field for the field in the input files containing the text data to be embedded, "
            "--config-file for the path to the semantic deduplication configuration file. "
            "Important configuration parameters include: "
            " cache_dir for the directory to store cache"
            " num_files for the number of files to process (default is -1, meaning all files),"
            " embeddings_save_loc for the location to save embeddings,"
            " embedding_model_name_or_path for the model name or path for embeddings,"
            " embedding_batch_size for the batch size for processing embeddings."
        ),
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
