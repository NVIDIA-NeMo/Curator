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
import logging
import os
import socket
from contextlib import nullcontext
from time import time

import cudf
from dask_cuda import LocalCUDACluster
from distributed import Client, performance_report


def create_logger(rank, log_file, name="logger", log_level=logging.INFO):
  # Create the logger
  logger = logging.getLogger(name)
  logger.setLevel(log_level)

  myhost = socket.gethostname()

  extra = {"host": myhost, "rank": rank}
  formatter = logging.Formatter(
      "%(asctime)s | %(host)s | Rank %(rank)s | %(message)s")

  # File handler for output
  file_handler = logging.FileHandler(log_file, mode="a")
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger = logging.LoggerAdapter(logger, extra)

  return logger


#TODO: Remove below to use nemo_curator.distributed_utils.get_client
def get_client(args) -> Client:
  if args.scheduler_address:
    if args.scheduler_file:
      raise ValueError(
          "Only one of scheduler_address or scheduler_file can be provided")
    else:
      return Client(address=args.scheduler_address, timeout="30s")
  elif args.scheduler_file:
    return Client(scheduler_file=args.scheduler_file, timeout="30s")
  else:
    extra_kwargs = ({
        "enable_tcp_over_ucx": True,
        "enable_nvlink": True,
        "enable_infiniband": False,
        "enable_rdmacm": False,
    } if args.nvlink_only and args.protocol == "ucx" else {})

    cluster = LocalCUDACluster(
        rmm_pool_size=args.rmm_pool_size,
        protocol=args.protocol,
        rmm_async=True,
        **extra_kwargs,
    )
    return Client(cluster)


def performance_report_if(path=None, report_name="dask-profile.html"):
  if path is not None:
    return performance_report(os.path.join(path, report_name))
  else:
    return nullcontext()


#TODO: Remove below to use nemo_curator.distributed_utils._enable_spilling
def enable_spilling():
  """
    Enables spilling to host memory for cudf
    """
  cudf.set_option("spill", True)


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
