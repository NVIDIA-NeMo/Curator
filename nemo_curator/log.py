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

import logging
import os
import socket

from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


def create_logger(rank, log_file, name="logger", log_level=logging.INFO):
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    myhost = socket.gethostname()

    extra = {"host": myhost, "rank": rank}
    formatter = logging.Formatter(
        "%(asctime)s | %(host)s | Rank %(rank)s | %(message)s"
    )

    # File handler for output
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger = logging.LoggerAdapter(logger, extra)

    return logger


def create_rank_logger(
    rank,
    log_dir,
    name="node_logger",
    log_level=logging.INFO,
):
    # Make the log directory if it does not exist
    log_dir = expand_outdir_and_mkdir(log_dir)

    # Create the rank subdirectory
    rank_tag = str(rank).rjust(3, "0")
    rank_dir = os.path.join(log_dir, f"rank_{rank_tag}")
    rank_dir = expand_outdir_and_mkdir(rank_dir)

    log_file = os.path.join(rank_dir, f"rank_{rank_tag}.log")
    return create_logger(rank, log_file, name=name, log_level=log_level)


def create_local_logger(
    rank,
    local_id,
    log_dir,
    name="local_logger",
    log_level=logging.INFO,
):
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Make tags
    rank_tag = str(rank).rjust(3, "0")
    local_id_tag = str(local_id).rjust(3, "0")

    myhost = socket.gethostname()
    extra = {"host": myhost, "node": rank_tag, "local": local_id_tag}
    formatter = logging.Formatter(
        "%(asctime)s | %(host)s | Node rank %(node)s "
        "| Local rank %(local)s | %(message)s"
    )

    # Output log file
    rank_dir = os.path.join(log_dir, f"rank_{rank_tag}")
    log_file = os.path.join(rank_dir, f"local_{local_id_tag}.log")

    # File handler for output
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger = logging.LoggerAdapter(logger, extra)

    return logger
