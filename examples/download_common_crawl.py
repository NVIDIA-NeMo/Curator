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

from nemo_curator.download import (
    JusTextExtractor,
    ResiliparseExtractor,
    TrafilaturaExtractor,
    download_common_crawl,
)
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args: argparse.Namespace) -> None:
    # Params
    start_snapshot = "2021-04"
    end_snapshot = "2021-10"
    output_directory = "/path/to/output"
    extractor = "justext"

    # Only download 10 shards as an example
    url_limit = 10

    # Set up Dask client
    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841

    # Set up the extractor
    if extractor == "resiliparse":
        extraction_algorithm = ResiliparseExtractor()
    elif extractor == "trafilatura":
        extraction_algorithm = TrafilaturaExtractor()
    else:
        # jusText is the default extractor in NeMo Curator
        extraction_algorithm = JusTextExtractor()

    # Download and sample data
    common_crawl = download_common_crawl(
        output_directory,
        start_snapshot,
        end_snapshot,
        url_limit=url_limit,
        algorithm=extraction_algorithm,
    )
    print(common_crawl.head())

    print(f"Downloaded Common Crawl data to {output_directory}")


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
