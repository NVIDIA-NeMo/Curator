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

from nemo_curator.download import download_arxiv
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    # Params
    output_directory = "/path/to/output"

    # Only download 10 shards as an example
    url_limit = 10

    # Set up Dask client
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Download and sample data
    arxiv = download_arxiv(output_directory, url_limit=url_limit)
    sample = arxiv.df.sample(frac=10 / len(arxiv))

    # Inspect the samples
    print(sample.compute())


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
