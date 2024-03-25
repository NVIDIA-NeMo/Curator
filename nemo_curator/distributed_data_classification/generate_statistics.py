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
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import get_all_files_paths_under


def value_counts(df, column_name):
    """
    This function groups a DataFrame by the specified column and counts the occurrences of each group.
    It is essentially the same as pandas.Series.value_counts, except it returns a DataFrame.

    Args:
        df: A DataFrame.
        column_name: The column by which to group the DataFrame.
    Returns:
        A DataFrame with two columns: column_name and a second column containing the counts per group.

    """
    return df.groupby(column_name).size().reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Generate label statistics and write them to disk"
    )

    parser = add_cluster_args(parser)
    parser = add_input_output_args(parser)
    parser.add_argument(
        "--label",
        type=str,
        help="The label column on which to generate statistics",
        required=True,
    )
    args = parser.parse_args()
    print(f"Arguments parsed = {args}", flush=True)
    client = get_client(args, cluster_type="gpu")

    print("Starting statistics workflow", flush=True)
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

    result = value_counts(df, column_name=args.label)
    result = result.rename(columns={0: "count"})
    result.to_json(args.output_file_path)

    et = time.time()
    print(f"Statistics workflow completed in {et-st}", flush=True)
    client.close()


def console_script():
    main()
