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

from nemo_curator.modules.fuzzy_dedup import JaccardSimilarity
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.script_utils import parse_gpu_dedup_args


def main(args):
    """Computes the Jaccard similarity between document pairs
    from partitioned parquet dataset. Result is a parquet dataset consiting of
    document id pair along with their Jaccard similarity score.
    """
    OUTPUT_PATH = args.output_dir
    shuffled_docs_path = args.shuffled_docs_path
    output_final_results_path = os.path.join(
        OUTPUT_PATH, "jaccard_similarity_results.parquet"
    )
    args.enable_spilling = True
    client = get_client(args, "gpu")

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


def attach_args(parser=None):
    description = """Computes jaccard similarity"""
    if not parser:
        parser = parse_gpu_dedup_args(description=description)

    parser.add_argument(
        "--shuffled-docs-path",
        type=str,
        help="The directory containing the shuffled documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory to write results to",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
        help="Size of ngram to use during jaccard similarity",
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
