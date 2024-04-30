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

from nemo_curator.modules.fuzzy_dedup import ConnectedComponents
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import parse_gpu_dedup_args


def main(args):
    """
    Takes a dataset consisting of document pairs
    and their corresponding jaccard similarity to compute connected
    components of docuements across pairs to find similar docuemnt
    after applying a given threshold. The result is a dataset
    consisting of all documents that are similar (above the threshold)
    and the component they belong to.
    """
    st = time.time()
    output_path = os.path.join(args.output_dir, "connected_components.parquet")
    args.set_torch_to_use_rmm = False
    args.enable_spilling = True

    client = get_client(args, cluster_type="gpu")

    components_stage = ConnectedComponents(
        cache_dir=args.cache_dir,
        jaccard_pairs_path=args.jaccard_pairs_path,
        id_column=args.input_json_id_field,
        convert_str_ids=True,
        jaccard_threshold=args.jaccard_threshold,
    )
    components_stage.cc_workflow(output_path=output_path)
    print(f"All done in {time.time()-st:.1f} seconds")
    print(f"Results written to {output_path}")


def attach_args(parser=None):
    description = """Computes connected component"""
    if not parser:
        parser = parse_gpu_dedup_args(description=description)

    parser.add_argument(
        "--jaccard-pairs-path",
        type=str,
        help="The directory containing the jaccard results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory to write results to",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="The cache directory to write intermediate results to",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=int,
        default=0.8,
        help="Jaccard threshold below which we don't consider documents"
        " to be duplicate",
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
