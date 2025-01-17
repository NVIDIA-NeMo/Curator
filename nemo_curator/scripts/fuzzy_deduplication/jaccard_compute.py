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

import argparse
import os
import time

from nemo_curator import JaccardSimilarity
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    """Computes the Jaccard similarity between document pairs
    from a partitioned Parquet dataset. Result is a Parquet dataset consiting of
    document and ID pairs, along with their Jaccard similarity scores.
    """
    OUTPUT_PATH = args.output_dir
    shuffled_docs_path = args.shuffled_docs_path
    output_final_results_path = os.path.join(
        OUTPUT_PATH, "jaccard_similarity_results.parquet"
    )
    args.enable_spilling = True
    client = get_client(**ArgumentHelper.parse_client_args(args))

    print(f"Num Workers = {get_num_workers(client)}", flush=True)
    print("Connected to dask cluster", flush=True)
    print("Running jaccard compute script", flush=True)
    st = time.time()
    jaccard = JaccardSimilarity(
        id_field=args.input_json_id_field,
        text_field=args.input_json_text_field,
        anchor_id_fields=[f"anchor_{i}_{args.input_json_id_field}" for i in range(2)],
        ngram_width=args.ngram_size,
    )
    # Run actual computation
    result_df = jaccard.jaccard_compute(shuffled_docs_path)

    result_df.to_parquet(
        output_final_results_path,
        write_index=False,
        write_metadata_file=False,
    )
    print(f"Jaccard Computing+Writing time: {time.time() - st:.1f} seconds")


def attach_args():
    description = "Computes Jaccard similarity scores."
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.parse_gpu_dedup_args()

    argumentHelper.add_arg_output_dir()
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
        help="Size of n-gram to use during Jaccard similarity computation.",
    )
    parser.add_argument(
        "--shuffled-docs-path",
        type=str,
        help="The directory containing the shuffled documents.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
