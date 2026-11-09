# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Print the Slurm arguments needed to retry outstanding array shards."""

from __future__ import annotations

import argparse

from nemo_curator.backends.slurm_array import (
    build_slurm_array_retry_submissions,
    find_slurm_array_retries,
    format_slurm_array_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find retryable NeMo Curator Slurm array shards")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Checkpoint directory used by the original logical array run.",
    )
    parser.add_argument(
        "--format",
        choices=["array", "fields"],
        default="array",
        help=(
            "Output only the Slurm array expression, or four shell fields per submission: "
            "array expression, shard index offset, minimum shard index, and original total shards."
        ),
    )
    parser.add_argument(
        "--max-array-size",
        type=int,
        default=None,
        help="Maximum physical Slurm array size; missing logical shards are split into offset submissions.",
    )
    args = parser.parse_args()

    if args.max_array_size is not None and args.max_array_size <= 0:
        parser.error("--max-array-size must be greater than 0.")
    if args.max_array_size is not None and args.format != "fields":
        parser.error("--max-array-size requires --format fields so each derived offset is included.")

    retry_plan = find_slurm_array_retries(args.checkpoint_path)
    if retry_plan is None:
        parser.error(
            "Slurm array run configuration was not found. Use the same checkpoint path as the original array run."
        )
    submissions = build_slurm_array_retry_submissions(retry_plan, args.max_array_size)
    if not submissions:
        return

    for submission in submissions:
        array_expression = format_slurm_array_indices(submission.array_indices)
        if args.format == "fields":
            print(
                array_expression,
                submission.shard_index_offset,
                retry_plan.minimum_shard_index,
                retry_plan.total_shards,
            )
        else:
            print(array_expression)


if __name__ == "__main__":
    main()
