#!/bin/env python
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
import os
import sys
from pathlib import Path

import yaml

this_script_path = Path(__file__).parent.absolute()

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "nemo_curator_benchmarking:latest")
GPUS = os.environ.get("GPUS", '"device=1"')
HOST_CURATOR_DIR = os.environ.get("HOST_CURATOR_DIR", str(this_script_path.parent.parent.absolute()))

CONTAINER_CURATOR_DIR = "/opt/Curator"
CONTAINER_RESULTS_DIR = "/results"
CONTAINER_ARTIFACTS_DIR = "/artifacts"
CONTAINER_DATASETS_DIR = "/datasets"
CONTAINER_HOST_CONFIG_DIR = "/CONFIG"

BASH_ENTRYPOINT_OVERRIDE = ""
ENTRYPOINT_ARGS = []
VOLUME_MOUNTS = []


def print_help(script_name: str) -> None:
    """Print usage and help message for the run script (not this script)to stderr."""
    sys.stderr.write(f"""
    Usage: {script_name} [OPTIONS] [ARGS ...]

    Options:
        --use-host-curator       Mount $HOST_CURATOR_DIR into the container for benchmarking/debugging curator sources without rebuilding the image.
        --shell                  Start an interactive bash shell instead of running benchmarks. ARGS, if specified, will be passed to 'bash -c'.
                                 For example: '--shell uv pip list | grep cugraph' will run 'uv pip list | grep cugraph' to display the version of cugraph installed in the container.
        -h, --help               Show this help message and exit.

        ARGS, if specified, are passed to the container entrypoint, either the default benchmarking entrypoint or the --shell bash entrypoint.

    Optional environment variables to override config and defaults:
        GPUS                      Value for --gpus option to docker run (using: {GPUS}).
        DOCKER_IMAGE              Docker image to use (using: {DOCKER_IMAGE}).
        HOST_CURATOR_DIR          Curator repo path used with --use-host-curator (see above) (using: {HOST_CURATOR_DIR}).
    """)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    global BASH_ENTRYPOINT_OVERRIDE  # noqa: PLW0603

    # This script must be called with the calling bash script as the first arg.
    # It will be removed before parsing the rest of the args.
    if len(sys.argv) > 1:
        script_name = Path(sys.argv[1]).name
        # Show help and exit if --help is passed as first arg. All other options including are passed to the
        # container entrypoint including -h and --help if other args are present.
        if len(sys.argv) > 2 and sys.argv[2] in ("-h", "--help"):  # noqa: PLR2004
            print_help(script_name)
            sys.exit(1)
        sys.argv.pop(1)
    else:
        msg = "Internal error: script name not provided"
        raise ValueError(msg)

    parser = argparse.ArgumentParser(
        description="Parse benchmarking tool options and output env variables for bash integration.",
        add_help=False,
    )
    parser.add_argument("--use-host-curator", action="store_true")
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--config", action="append", type=Path, default=[])

    args, unknown = parser.parse_known_args()

    # Set volume mount for host curator directory.
    if args.use_host_curator:
        VOLUME_MOUNTS.append(f"--volume {HOST_CURATOR_DIR}:{CONTAINER_CURATOR_DIR}")

    # Set entrypoint to bash if --shell is passed.
    if args.shell:
        BASH_ENTRYPOINT_OVERRIDE = "--entrypoint=bash"
        if len(unknown) > 0:
            ENTRYPOINT_ARGS.extend(["-c", " ".join(unknown)])
    else:
        ENTRYPOINT_ARGS.extend(unknown)

    # Parse config files and set volume mounts for results, artifacts, and datasets.
    if args.config:
        config_data = {}
        for config_file in args.config:
            if not config_file.exists():
                msg = f"Config file not found: {config_file}."
                raise FileNotFoundError(msg)
            with open(config_file) as f:
                new_config = yaml.safe_load(f)
            if not isinstance(new_config, dict):
                continue
            config_data.update(new_config)

        for path_type, container_dir in [
            ("results_path", CONTAINER_RESULTS_DIR),
            ("artifacts_path", CONTAINER_ARTIFACTS_DIR),
            ("datasets_path", CONTAINER_DATASETS_DIR),
        ]:
            if path_type in config_data:
                path_value = config_data[path_type]
                if path_value.startswith("/"):
                    VOLUME_MOUNTS.append(f"--volume {path_value}:{container_dir}")
                else:
                    msg = f"Path value {path_value} for {path_type} must be an absolute path."
                    raise ValueError(msg)
            else:
                msg = f"Path value {path_type} not found in config file(s)."
                raise ValueError(msg)

    # Add volume mounts for each config file so the script in the container can read each one and add each to ENTRYPOINT_ARGS.
    container_config_path = Path(CONTAINER_HOST_CONFIG_DIR)

    for config_file in args.config:
        config_file_host = config_file.absolute().expanduser().resolve()
        # Use the full abs path of the host file to ensure uniqueness.
        container_config_file_path = Path(f"{container_config_path}/{config_file_host}")
        VOLUME_MOUNTS.append(f"--volume {config_file_host}:{container_config_file_path}")
        # Only add modified --config args if running the benchmark tool entrypoint, not the shell entrypoint.
        if not args.shell:
            ENTRYPOINT_ARGS.append(f"--config {container_config_file_path}")

    # Print final vars for eval in bash
    print(f"BASH_ENTRYPOINT_OVERRIDE={BASH_ENTRYPOINT_OVERRIDE}")
    print(f"DOCKER_IMAGE={DOCKER_IMAGE}")
    print(f"GPUS={GPUS}")
    print(f"HOST_CURATOR_DIR={HOST_CURATOR_DIR}")
    vms = f'"{" ".join(VOLUME_MOUNTS)}"' if VOLUME_MOUNTS else ""
    print(f"VOLUME_MOUNTS={vms}")
    # Print ENTRYPOINT_ARGS as a bash array suitable for eval
    array_str = "ENTRYPOINT_ARGS=(" + " ".join([f'"{arg}"' for arg in ENTRYPOINT_ARGS]) + ")"
    print(array_str)


if __name__ == "__main__":
    main()
