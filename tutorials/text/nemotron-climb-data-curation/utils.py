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
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.core.client import RayClient


def attach_ray_client_args(parser: argparse.ArgumentParser) -> None:
    from nemo_curator.core.constants import (
        DEFAULT_RAY_CLIENT_SERVER_PORT,
        DEFAULT_RAY_DASHBOARD_HOST,
        DEFAULT_RAY_DASHBOARD_PORT,
        DEFAULT_RAY_METRICS_PORT,
        DEFAULT_RAY_PORT,
        DEFAULT_RAY_TEMP_DIR,
    )

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


def create_ray_client(args: argparse.Namespace) -> "RayClient":
    from nemo_curator.core.client import RayClient

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


def centroid_id(path: str) -> int | None:
    """Return the integer N from a `centroid=N` basename, or None if the path doesn't match the convention."""
    base = os.path.basename(path.rstrip("/"))
    if not base.startswith("centroid="):
        return None
    try:
        return int(base.split("=", 1)[1])
    except ValueError:
        return None


def list_centroid_dirs(parent: str) -> list[str]:
    """List subdirectories of `parent` whose basename matches `centroid=<int>`."""
    return sorted(
        os.path.join(parent, name)
        for name in os.listdir(parent)
        if os.path.isdir(os.path.join(parent, name)) and centroid_id(name) is not None
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

    # Normalize via Path so a trailing slash on input_path doesn't produce "//" keys
    files = sorted(str(p) for p in Path(input_path).glob("*.bin"))

    if not files:
        msg = f"No .bin files found under {input_path}. Check that the path points to the tokenized output from step 4."
        raise FileNotFoundError(msg)

    sizes = [os.path.getsize(f) for f in files]
    total = sum(sizes)

    if total == 0:
        msg = f"All .bin files under {input_path} are empty (total size 0 bytes); cannot compute a token distribution."
        raise ValueError(msg)

    weights: list[float] = [s / total for s in sizes]

    return dict(zip(files, weights, strict=True))
