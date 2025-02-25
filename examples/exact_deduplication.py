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

import argparse
import time

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client, write_to_disk
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports():
    import cudf  # noqa: F401


def main(args):

    dataset_dir = "/path/to/data"
    log_dir = "./"
    output_dir = "./"
    dataset_id_field = "id"
    dataset_text_field = "text"
    client = get_client(**ArgumentHelper.parse_client_args(args))
    backend = "cudf" if args.device == "gpu" else "pandas"

    if args.device == "gpu":
        client.run(pre_imports)

    t0 = time.time()
    input_dataset = DocumentDataset.read_json(
        dataset_dir, backend=backend, blocksize="1GiB", files_per_partition=None
    )

    exact_dup = ExactDuplicates(
        logger=log_dir,
        id_field=dataset_id_field,
        text_field=dataset_text_field,
        # Decides whether output of the module is deduplicated dataset or duplicates
        # If true, you should set cache_dir for performance improvement
        perform_removal=False,
        # cache_dir=output_dir  # Optionally write the output to disk
    )

    # When perform_removal=False, it will only call .identify_duplicates() and return the list of duplicate IDs.
    # When perform_removal=True, then exact_dup outputs the dataset with the duplicates removed.
    # It will behave by calling .identify_duplicates() and .remove() in sequence.
    duplicates = exact_dup(
        dataset=input_dataset
    )  # or exact_dup.identify_duplicates(input_dataset)

    # If caching, result is a path to the output dataset.
    if isinstance(duplicates, str):
        duplicates = DocumentDataset.read_parquet(duplicates, backend=backend)

    result = exact_dup.remove(input_dataset, duplicates)
    write_to_disk(result, output_dir, output_type="parquet")
    print(time.time() - t0)


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
