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

from nemo_curator.backends.slurm_array import find_slurm_array_retries, format_slurm_array_indices


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
            "Output only the Slurm array expression, or three shell fields: "
            "array expression, minimum shard index, and original total shards."
        ),
    )
    args = parser.parse_args()

    retry_plan = find_slurm_array_retries(args.checkpoint_path)
    if retry_plan is None:
        parser.error(
            "Slurm array run configuration was not found. Use the same checkpoint path as the original array run."
        )
    if not retry_plan.shard_indices:
        return

    array_expression = format_slurm_array_indices(retry_plan.shard_indices)
    if args.format == "fields":
        print(array_expression, retry_plan.minimum_shard_index, retry_plan.total_shards)
    else:
        print(array_expression)


if __name__ == "__main__":
    main()
