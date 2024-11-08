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
import ast
import os
from typing import Union

import pandas as pd

from nemo_curator.utils.script_utils import ArgumentHelper


def parse_args():
    """
    This function adds command line arguments related to the two files we wish to compare.

    Returns:
        An argparse Namespace object.

    """
    description = "Run verification."
    parser = argparse.ArgumentParser(description=description)

    ArgumentHelper(parser).add_arg_input_meta()
    parser.add_argument(
        "--expected_pred_column",
        type=str,
        default="pred",
        help="The prediction column name for the expected_result file.",
    )

    parser.add_argument(
        "--expected_results_file_path",
        type=str,
        required=True,
        help="The path of the expected_result file.",
    )
    parser.add_argument(
        "--results_file_path",
        type=str,
        required=True,
        help="The path of the input files.",
    )
    parser.add_argument(
        "--results_pred_column",
        type=str,
        default="pred",
        help="The prediction column name for the input files.",
    )

    return parser.parse_args()


def verify_same_counts(got_counts, expected_counts):
    """
    This function compares the results of `value_counts` on two Series.
    If the value counts are not the same, it prints information about the percent difference between them.

    Args:
        got_counts: A dictionary of value counts for the input Series.
        expected_counts: A dictionary of the expected value counts.

    """
    exact_same_counts = got_counts == expected_counts
    if exact_same_counts:
        print("Results are exactly the same")
    else:
        print("Results are not exactly the same, see below")
        total_count = sum(expected_counts.values())
        total_diff = 0
        for key in expected_counts:
            if got_counts[key] != expected_counts[key]:
                diff = expected_counts[key] - got_counts[key]
                print(
                    f"Expected doc count for label {key} {expected_counts[key]} but got {got_counts[key]}"
                )
                total_diff = total_diff + abs(diff)

        diff_percent = (total_diff / total_count) * 100
        print(
            f"Total difference: {total_diff} out of {total_count}, diff percentage: {diff_percent}%"
        )
    print("---" * 30)


def verify_same_dataframe(
    got_df, expected_df, results_pred_column, expected_pred_column
):
    """
    This function verifies whether a column from one DataFrame is identical to another column in another DataFrame.
    It prints information about the differences between the two columns.

    Args:
        got_df: The input DataFrame.
        expected_df: The expected DataFrame.
        results_pred_column: The column of interest in the input DataFrame.
        expected_pred_column: The column of interest in the expected DataFrame.

    """
    # Compare the values in the two DataFrames element-wise
    matches_df = got_df.merge(
        expected_df,
        left_on=["text", results_pred_column],
        right_on=["text", expected_pred_column],
        indicator="Matched",
        how="outer",
    )

    matches_df = matches_df[matches_df["Matched"] == "both"]
    # Calculate the match ratio
    total_values = len(expected_df)
    match_ratio = len(matches_df) / total_values

    different_count = abs(total_values - len(matches_df))

    print(f"DataFrame Match Ratio: {match_ratio:.4f}")
    print(f"Out of {len(expected_df)} rows {different_count} are different", flush=True)
    print("---" * 30)


def verify_results(
    results_file_path: str,
    expected_results_file_path: str,
    results_pred_column: str,
    expected_pred_column: str,
    input_meta: Union[str, dict] = None,
):
    """
    This function compares an input file with its expected result file.
    See `verify_same_counts` and `verify_same_dataframe`.

    Args:
        results_file_path: The path of the input files.
        expected_results_file_path: The path of the expected_result file.
        results_pred_column: The prediction column name for the input files.
        expected_pred_column: The prediction column name for the expected_result file.
        input_meta: A dictionary or a string formatted as a dictionary, which outlines
            the field names and their respective data types within the JSONL input file.

    """
    if type(input_meta) == str:
        input_meta = ast.literal_eval(input_meta)

    expected_df = pd.read_json(expected_results_file_path, lines=True, dtype=input_meta)
    expected_df = expected_df.sort_values(by=["text"]).reset_index(drop=True)
    expected_counts = expected_df[expected_pred_column].value_counts().to_dict()

    expected_columns = expected_df.columns
    if results_pred_column != expected_pred_column:
        expected_columns = [
            results_pred_column if item == expected_pred_column else item
            for item in expected_columns
        ]

    got_paths = [p for p in os.scandir(results_file_path)]
    got_df = [
        pd.read_json(path, lines=True, dtype=input_meta)[expected_columns]
        for path in got_paths
    ]
    got_df = pd.concat(got_df, ignore_index=True)
    got_df = got_df.sort_values(by=["text"]).reset_index(drop=True)
    got_counts = got_df[results_pred_column].value_counts().to_dict()

    verify_same_counts(got_counts, expected_counts)
    verify_same_dataframe(
        got_df, expected_df, results_pred_column, expected_pred_column
    )


def main():
    """
    This script is useful for determining whether your predicted classifications match an expected output.
    With this in mind, it is only useful if users have the expected results already computed.
    """
    args = parse_args()
    verify_results(
        args.results_file_path,
        args.expected_results_file_path,
        args.results_pred_column,
        args.expected_pred_column,
        args.input_meta,
    )


def console_script():
    main()
