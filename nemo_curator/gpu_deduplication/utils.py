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
from time import time


def get_num_workers(client):
    """
    Returns the number of workers in the cluster
    """
    worker_list = list(client.scheduler_info()["workers"].keys())
    return len(worker_list)


def get_list_of_lists(lst, nchunks):
    """
    Splits a list into nchunks lists
    """
    return [lst[i::nchunks] for i in range(nchunks)]


def parse_nc_args(
    description="Default gpu dedup nemo_curator argument parser",
) -> argparse.ArgumentParser:
    """
    Adds default set of arguments that are common to multiple stages
    of the pipeline
    """
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-data-dirs",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Input directories consisting of .jsonl files that are accessible "
        "to all nodes. This path must be accessible by all machines in the cluster",
    )
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=None,
        help="Address to the scheduler of a created dask cluster. If not provided"
        "a single node LocalCUDACluster will be started.",
    )
    parser.add_argument(
        "--scheduler-file",
        type=str,
        default=None,
        help="Path to the scheduler file of a created dask cluster. If not provided"
        " a single node LocalCUDACluster will be started.",
    )
    parser.add_argument(
        "--rmm-pool-size",
        type=str,
        default=None,
        help="Initial pool size to use for the RMM Pool Memory allocator"
        "Note: This only applies to the localCUDACluster. If providing an user created "
        "cluster refer to"
        "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size",  # noqa: E501
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="tcp",
        help="Protcol to use for dask cluster"
        "Note: This only applies to the localCUDACluster. If providing an user created "
        "cluster refer to"
        "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol",  # noqa: E501
    )
    parser.add_argument(
        "--nvlink-only",
        action="store_true",
        help="Start a local cluster with only NVLink enabled."
        "Only applicable when protocol=ucx and no scheduler file/address is specified",
    )
    parser.add_argument(
        "--input-json-text-field",
        type=str,
        default="text",
        help="The name of the field within each json object of the jsonl "
        "file that contains the text from which minhashes will be computed. ",
    )
    parser.add_argument(
        "--input-json-id-field",
        type=str,
        default="adlr_id",
        help="The name of the field within each json object of the jsonl "
        "file that assigns a unqiue ID to each document. "
        "Can be created by running the script "
        "'./prospector/add_id.py' which adds the field 'adlr_id' "
        "to the documents in a distributed fashion",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="The output log directory where node and local",
    )
    parser.add_argument(
        "--files-per-partition",
        type=int,
        default=2,
        help="Number of jsonl files to combine into single partition",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Upper limit on the number of json files to process",
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=500,
        help="The frequency with which to write log messages when "
        "computing MinHashses. By default a log message will "
        "be written every 500 partitions",
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help="Path to save dask profile",
    )
    return parser


def timer(func):

    def wrapper(*args, **kw):
        print(f"function {func.__name__} started...")
        start = time()
        res = func(*args, **kw)
        duration = time() - start
        timing = f"function {func.__name__} finished in {duration:.1f} seconds"
        print(timing)
        return res

    return wrapper
