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
import os
from itertools import islice


def attach_bool_arg(parser, flag_name, default=False, help_str=None):
    attr_name = flag_name.replace("-", "_")
    parser.add_argument(
        "--{}".format(flag_name),
        dest=attr_name,
        action="store_true",
        help=flag_name.replace("-", " ") if help_str is None else help_str,
    )
    parser.add_argument(
        "--no-{}".format(flag_name),
        dest=attr_name,
        action="store_false",
        help=flag_name.replace("-", " ") if help_str is None else help_str,
    )
    parser.set_defaults(**{attr_name: default})


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds default set of arguments that are needed for Dask cluster setup
    """
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
        "--n-workers",
        type=int,
        default=os.cpu_count(),
        help="The number of workers to run in total on the Dask CPU cluster",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="The number of threads ot launch per worker on the Dask CPU cluster. Usually best set at 1 due to the GIL.",
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
        "--files-per-partition",
        type=int,
        default=2,
        help="Number of jsonl files to combine into single partition",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=-1,
        help="Upper limit on the number of json files to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the script on. Either 'cpu' or 'gpu'.",
    )

    return parser


def chunk_list(lst, nchnks):
    nitem = len(lst)
    splits = splitnum(nitem, nchnks)
    beg, end = 0, splits[0]
    for i in range(nchnks):
        if i == nchnks - 1:
            yield lst[beg:]
        else:
            yield lst[beg:end]
            beg = end
            end += splits[i + 1]


def get_ranges(n, nchnks):
    splits = splitnum(n, nchnks)
    beg, end = 0, splits[0]
    for i in range(nchnks):
        if i == nchnks - 1:
            yield beg, n
        else:
            yield beg, end
            beg = end
            end += splits[i + 1]


def chunk_list_lean(lst, nchnks):
    nitem = len(lst)
    splits = splitnum(nitem, nchnks)
    # Outer loop over chunks
    for i in range(nchnks):
        # Slice thie list
        yield lst[0 : splits[i]]
        # Remove the chunk from the total list
        del lst[0 : splits[i]]


def chunk_dict(din, nchnks):
    nitem = len(din)
    splits = splitnum(nitem, nchnks)
    beg, end = 0, splits[0]
    # Outer loop over chunks
    for i in range(nchnks):
        it, out = iter(din), {}
        # Slice the dictionary
        for k in islice(it, beg, end):
            out[k] = din[k]
        if i == nchnks - 1:
            yield out
        else:
            beg = end
            end += splits[i + 1]
            yield out


def chunk_dict_lean(din, nchnks):
    nitem = len(din)
    splits = splitnum(nitem, nchnks)
    # Outer loop over chunks
    for i in range(nchnks):
        it = iter(din)
        out = {}
        # Slice the dictionary
        for k in islice(it, splits[i]):
            out[k] = din[k]
        yield out
        # Clear out chunked entries
        for k in out.keys():
            del din[k]


def splitnum(num, div):
    """Splits a number into nearly even parts"""
    splits = []
    igr, rem = divmod(num, div)
    for i in range(div):
        splits.append(igr)
    for i in range(rem):
        splits[i] += 1

    return splits
