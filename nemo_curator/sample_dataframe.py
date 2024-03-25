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
import time

from nemo_curator.distributed_data_classification.arg_utils import (
    add_cluster_args,
    add_input_output_args,
)
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under


def sample_dataframe(df, num_samples):
    """
    This function samples a specified number of rows from a DataFrame.

    Args:
        df: A DataFrame.
        num_samples: The number of rows to randomly sample from the DataFrame.
    Returns:
        The sampled DataFrame.

    """
    len_df = len(df)
    print(f"Total length = {len_df}", flush=True)
    sampled_df = df.sample(frac=(num_samples / len_df), random_state=42)
    return sampled_df


if __name__ == "__main__":
    """
    This script is useful if a user wishes to sample a very large dataset for smaller scale testing,
    for example, a dataset written as a directory containing thousands of jsonl files.
    """
    parser = argparse.ArgumentParser(description="Sample rows and write them to disk")

    parser = add_cluster_args(parser)
    parser = add_input_output_args(parser)
    parser.add_argument(
        "--num_samples",
        type=int,
        help="The number of rows to sample",
        required=True,
    )
    args = parser.parse_args()
    print(f"Arguments parsed = {args}", flush=True)
    client = get_client(args, cluster_type="gpu")

    print("Starting sampling workflow", flush=True)
    st = time.time()
    df = read_data(
        input_files=get_all_files_paths_under(
            args.input_file_path, recurse_subdirecties=False
        ),
        file_type=args.input_file_type,
        add_filename=True,
    )
    input_files = get_all_files_paths_under(
        args.input_file_path, recurse_subdirecties=False
    )
    sampled_df = sample_dataframe(df, num_samples=args.num_samples)
    write_to_disk(
        df=sampled_df,
        output_file_dir=args.output_file_path,
        write_to_filename=True,
    )
    et = time.time()
    print(f"Sampling workflow completed in {et-st}", flush=True)
    client.close()
