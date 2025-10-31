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
# Add the parent directory to PYTHONPATH to import the runner modules
sys.path.insert(0, str(this_script_path.parent))
from runner.path_resolver import (  # noqa: E402
    CONTAINER_ARTIFACTS_DIR_ROOT,
    CONTAINER_CONFIG_DIR_ROOT,
    CONTAINER_CURATOR_DIR,
    CONTAINER_DATASETS_DIR_ROOT,
    CONTAINER_RESULTS_DIR_ROOT,
)

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "nemo_curator_benchmarking:latest")
GPUS = os.environ.get("GPUS", '"device=1"')
HOST_CURATOR_DIR = os.environ.get("HOST_CURATOR_DIR", str(this_script_path.parent.parent.absolute()))
CURATOR_BENCHMARKING_DEBUG = os.environ.get("CURATOR_BENCHMARKING_DEBUG", "0")

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
      GPUS                          Value for --gpus option to docker run (using: {GPUS}).
      DOCKER_IMAGE                  Docker image to use (using: {DOCKER_IMAGE}).
      HOST_CURATOR_DIR              Curator repo path used with --use-host-curator (see above) (using: {HOST_CURATOR_DIR}).
      CURATOR_BENCHMARKING_DEBUG    Set to 1 for debug mode (regular output, possibly more in the future) (using: {CURATOR_BENCHMARKING_DEBUG}).
    """)


def create_unique_volume_mounts(volume_mounts: list[tuple[str, str]]) -> list[str]:
    """
    Create unique container mount points based on the host paths.
    This simply prepends the container mount point to the host path.
    """
    unique_volume_mounts = []
    for host_path, container_path in volume_mounts:
        abs_host_path = Path(host_path).absolute().expanduser().resolve()
        unique_container_path = Path(f"{container_path}/{abs_host_path}")
        unique_volume_mounts.append(f"--volume {abs_host_path}:{unique_container_path}")
    return unique_volume_mounts


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    global BASH_ENTRYPOINT_OVERRIDE  # noqa: PLW0603
    volume_mounts_to_prepend = []

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
        # consolidate all config files passed in into a single dict - last one wins.
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

        # process the final path settings into the list of volume mounts.
        for path_type, container_dir in [
            ("results_path", CONTAINER_RESULTS_DIR_ROOT),
            ("artifacts_path", CONTAINER_ARTIFACTS_DIR_ROOT),
            ("datasets_path", CONTAINER_DATASETS_DIR_ROOT),
        ]:
            if path_type in config_data:
                path_value = config_data[path_type]
                if path_value.startswith("/"):
                    volume_mounts_to_prepend.append((path_value, container_dir))
                else:
                    msg = f"Path value {path_value} for {path_type} must be an absolute path."
                    raise ValueError(msg)
            else:
                msg = f"Path value {path_type} not found in config file(s)."
                raise ValueError(msg)

    # Add volume mounts for each config file so the script in the container can read each one and add each to ENTRYPOINT_ARGS.
    for config_file in args.config:
        config_file_host = config_file.absolute().expanduser().resolve()
        volume_mounts_to_prepend.append((config_file_host, CONTAINER_CONFIG_DIR_ROOT))
        # Only add modified --config args if running the benchmark tool entrypoint, not the shell entrypoint.
        if not args.shell:
            container_config_file_path = Path(f"{CONTAINER_CONFIG_DIR_ROOT}/{config_file_host}")
            ENTRYPOINT_ARGS.append(f"--config={container_config_file_path}")

    # The volume mounts collected here may not be unique (i.e. some may be mapped to the same container path),
    # so generate unique continaer mount points based on the host paths.
    VOLUME_MOUNTS.extend(create_unique_volume_mounts(volume_mounts_to_prepend))

    # Print final vars for eval in bash
    print(f"BASH_ENTRYPOINT_OVERRIDE={BASH_ENTRYPOINT_OVERRIDE}")
    print(f"DOCKER_IMAGE={DOCKER_IMAGE}")
    print(f"GPUS={GPUS}")
    print(f"HOST_CURATOR_DIR={HOST_CURATOR_DIR}")
    print(f"CURATOR_BENCHMARKING_DEBUG={CURATOR_BENCHMARKING_DEBUG}")
    vms = f'"{" ".join(VOLUME_MOUNTS)}"' if VOLUME_MOUNTS else ""
    print(f"VOLUME_MOUNTS={vms}")
    # Print ENTRYPOINT_ARGS as a bash array suitable for eval
    array_str = "ENTRYPOINT_ARGS=(" + " ".join([f'"{arg}"' for arg in ENTRYPOINT_ARGS]) + ")"
    print(array_str)


if __name__ == "__main__":
    main()
