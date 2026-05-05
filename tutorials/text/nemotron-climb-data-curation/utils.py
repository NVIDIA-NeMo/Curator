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

import argparse
import glob
import os

from nemo_curator.core.client import RayClient
from nemo_curator.core.constants import (
    DEFAULT_RAY_CLIENT_SERVER_PORT,
    DEFAULT_RAY_DASHBOARD_HOST,
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_METRICS_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_RAY_TEMP_DIR,
)


def attach_ray_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ray-port", type=int, default=DEFAULT_RAY_PORT)
    parser.add_argument("--ray-dashboard-port", type=int, default=DEFAULT_RAY_DASHBOARD_PORT)
    parser.add_argument("--ray-client-server-port", type=int, default=DEFAULT_RAY_CLIENT_SERVER_PORT)
    parser.add_argument("--ray-temp-dir", type=str, default=DEFAULT_RAY_TEMP_DIR)
    parser.add_argument("--include-dashboard", action="store_true", default=False)
    parser.add_argument("--ray-metrics-port", type=int, default=DEFAULT_RAY_METRICS_PORT)
    parser.add_argument("--ray-dashboard-host", type=str, default=DEFAULT_RAY_DASHBOARD_HOST)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--enable-object-spilling", action="store_true", default=False)
    parser.add_argument("--ray-stdouterr-capture-file", type=str, default=None)
    parser.add_argument("--metrics-dir", type=str, default=None)


def create_ray_client(args: argparse.Namespace) -> RayClient:
    return RayClient(
        ray_port=args.ray_port,
        ray_dashboard_port=args.ray_dashboard_port,
        ray_client_server_port=args.ray_client_server_port,
        ray_temp_dir=args.ray_temp_dir,
        include_dashboard=args.include_dashboard,
        ray_metrics_port=args.ray_metrics_port,
        ray_dashboard_host=args.ray_dashboard_host,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        enable_object_spilling=args.enable_object_spilling,
        ray_stdouterr_capture_file=args.ray_stdouterr_capture_file,
        metrics_dir=args.metrics_dir,
    )


def get_token_distribution(input_path: str) -> dict[str, float]:
    """
    Get the token distribution from the input path of the tokenized files.

    This function is adapted from the RegMix project:
    https://github.com/sail-sg/regmix/blob/main/mixture_config/synthesize_mixture.py

    Args:
    input_path (str): Path to the input directory containing the tokenized files.

    Returns:
    dict: Dictionary of tokenized files and their corresponding weights.
    """

    files = sorted(glob.glob(f"{input_path}/*.bin"))

    sizes = [os.path.getsize(f) for f in files]
    total = sum(sizes)

    weights: list[float] = [s / total for s in sizes]

    return dict(zip(files, weights, strict=True))
