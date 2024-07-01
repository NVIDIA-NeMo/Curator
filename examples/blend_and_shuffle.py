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

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    # Params
    dataset_paths = ["/path/to/first", "/path/to/second", "/path/to/third"]
    dataset_weights = [5.0, 2.0, 1.0]
    target_size = 1000
    output_path = "/path/to/output"

    # Set up Dask client
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Blend the datasets
    datasets = [DocumentDataset.read_json(path) for path in dataset_paths]
    blended_dataset = nc.blend_datasets(target_size, datasets, dataset_weights)

    shuffle = nc.Shuffle(seed=42)
    blended_dataset = shuffle(blended_dataset)

    # Save the blend
    blended_dataset.to_json(output_path)


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
