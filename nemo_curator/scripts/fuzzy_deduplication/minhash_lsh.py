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

import os
import time

import cudf
import dask_cudf
import numpy as np

from nemo_curator import LSH
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import convert_str_id_to_int
from nemo_curator.utils.script_utils import parse_gpu_dedup_args


def pre_imports():
    import cudf  # noqa: F401


def main(args):

    logger = create_logger(
        rank=0, log_file=os.path.join(args.log_dir, "rank_000.log"), name="lsh_log"
    )
    logger.info(f"Starting workflow with args:\n {args}")

    assert args.device == "gpu"
    args.set_torch_to_use_rmm = False
    client = get_client(args, cluster_type=args.device)
    logger.info(f"Client Created {client}")
    client.run(pre_imports)
    logger.info("Pre imports complete")

    data_paths = args.input_data_dirs
    id_field = args.input_json_id_field
    minhash_field = args.input_minhash_field

    dfs = []
    for data_path in data_paths:
        dfs.append(
            dask_cudf.read_parquet(data_path, blocksize="2GB", aggregate_files=True)
        )
    df = dask_cudf.concat(dfs, ignore_unknown_divisions=True)
    df = df.map_partitions(
        convert_str_id_to_int,
        id_column=id_field,
        meta=cudf.DataFrame(
            {minhash_field: [[1, 2, 3]], "doc_id": [1], "dataset_id": np.uint32(1)}
        ),
    )
    lsh = LSH(
        cache_dir=args.output_bucket_dir,
        num_hashes=args.minhash_length,
        num_buckets=args.num_bands,
        buckets_per_shuffle=args.buckets_per_shuffle,
        id_fields=["dataset_id", "doc_id"],
        profile_dir=args.profile_path,
        minhash_field=minhash_field,
        logger=logger,
    )
    t1 = time.time()
    _ = lsh(DocumentDataset(df))
    logger.info(f"Computing and writing buckets took {time.time() - t1} s")


def attach_args(parser=None):
    description = """Compute buckets from existing minhashes and writes the output
    to files. Each row corresponding to a document-id followed by the columns
    denoting the bucket id's that document belongs to.
    """
    if not parser:
        parser = parse_gpu_dedup_args(description=description)

    parser.add_argument(
        "--minhash-length",
        type=int,
        default=260,
        help="The minhash signature length of each input document",
    )
    parser.add_argument(
        "--input-minhash-field",
        type=str,
        default="_minhash_signature",
        help="Name of the column containing minhashes",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=20,
        help="The number of minhashes to compute for each document.",
    )
    parser.add_argument(
        "--buckets-per-shuffle",
        type=int,
        required=True,
        help="Number of buckets to shuffle per batch",
    )
    parser.add_argument(
        "--output-bucket-dir",
        type=str,
        required=True,
        help="Output directory where minhashes will be written. "
        "Each file parquet file consiting of document and bucket IDs",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
