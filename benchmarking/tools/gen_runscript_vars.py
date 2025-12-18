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

# ruff: noqa: E402
from runner.path_resolver import (
    CONTAINER_CONFIG_DIR_ROOT,
    CONTAINER_CURATOR_DIR,
    CONTAINER_DATASETS_DIR_ROOT,
    CONTAINER_RESULTS_DIR_ROOT,
)
from runner.utils import get_total_memory_bytes

_KB = 1024
_MB = 1024 * _KB
_GB = 1024 * _MB
_TB = 1024 * _GB
_max_container_memory_bytes = 1 * _TB
_shm_size_container_memory_percentage = 0.5  # 50%

CURATOR_BENCHMARKING_IMAGE = os.environ.get("CURATOR_BENCHMARKING_IMAGE", "nemo_curator_benchmarking:latest")
GPUS = os.environ.get("GPUS", "all")
HOST_CURATOR_DIR = os.environ.get("HOST_CURATOR_DIR", str(this_script_path.parent.parent.absolute()))
CURATOR_BENCHMARKING_DEBUG = os.environ.get("CURATOR_BENCHMARKING_DEBUG", "0")

BASH_ENTRYPOINT_OVERRIDE = ""
ENTRYPOINT_ARGS = []
VOLUME_MOUNTS = []


def print_help(script_name: str) -> None:
    """Print usage and help message for the run script (not this script) to stderr."""
    sys.stderr.write(f"""
  Usage: {script_name} [OPTIONS] [ARGS ...]

  Options:
      --use-host-curator       Mount $HOST_CURATOR_DIR into the container for benchmarking/debugging curator sources without rebuilding the image.
      --shell                  Start an interactive bash shell instead of running benchmarks. ARGS, if specified, will be passed to 'bash -c'.
                               For example: '--shell uv pip list | grep cugraph' will run 'uv pip list | grep cugraph' to display the version of cugraph installed in the container.
      --config <path>          Path to a YAML config file. Can be specified multiple times to merge configs. This arg is required if not using --shell.
      -h, --help               Show this help message and exit.

      ARGS, if specified, are passed to the container entrypoint, either the default benchmarking entrypoint or the --shell bash entrypoint.

  Optional environment variables to override config and defaults:
      GPUS                          Value for --gpus option to docker run (using: {GPUS}).
      CURATOR_BENCHMARKING_IMAGE    Docker image to use (using: {CURATOR_BENCHMARKING_IMAGE}).
      HOST_CURATOR_DIR              Curator repo path used with --use-host-curator (see above) (using: {HOST_CURATOR_DIR}).
      CURATOR_BENCHMARKING_DEBUG    Set to 1 for debug mode (regular output, possibly more in the future) (using: {CURATOR_BENCHMARKING_DEBUG}).
    """)


def combine_dir_paths(patha: str | Path, pathb: str | Path) -> Path:
    """Combine two paths, ensuring the result is an absolute path."""
    return Path(f"{patha}/{pathb}").absolute().expanduser().resolve()


def get_runscript_eval_str(argv: list[str]) -> str:  # noqa: C901, PLR0912, PLR0915
    """
    Parse CLI args and output env variables for bash integration.
    argv is a list of strings to be treated like sys.argv, where argv[0] is the name of this script.
    argv[1] must be the name of the run.sh bash script, then all other args follow.
    Example: ["this_script.py", "run.sh", "--use-host-curator", "--shell", "--config", "config.yaml"]
    Returns a string of env variables to eval in bash.
    """
    # Make a copy of argv to avoid modifying the caller's list.
    argv = argv.copy()
    # Initialize with defaults, these will be modified in this function.
    bash_entrypoint_override = BASH_ENTRYPOINT_OVERRIDE
    entrypoint_args = ENTRYPOINT_ARGS.copy()
    volume_mounts = VOLUME_MOUNTS.copy()

    # This script must be called with the run.sh bash script as the first arg.
    # It will be removed before parsing the rest of the args.
    if len(argv) > 1:
        script_name = Path(argv[1]).name
        # Show help and exit if -h|--help is passed as first arg. All other options are passed to the
        # container entrypoint including -h|--help if other args are present. This provides a way for
        # -h|--help to be passed to the container entrypoint while still allowing for -h|--help output
        # for the run.sh script.
        if len(argv) > 2 and argv[2] in ("-h", "--help"):  # noqa: PLR2004
            print_help(script_name)
            sys.exit(1)
        argv.pop(1)
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

    args, unknown_args = parser.parse_known_args(argv[1:])

    # Set volume mount for host curator directory.
    if args.use_host_curator:
        # Do not use combine_dir_paths here since CONTAINER_CURATOR_DIR is assumed to be a unique absolute
        # path (e.g., /opt/Curator from Dockerfile).
        volume_mounts.append(f"--volume {Path(HOST_CURATOR_DIR).absolute()}:{CONTAINER_CURATOR_DIR}")

    # Set entrypoint to bash if --shell is passed.
    if args.shell:
        bash_entrypoint_override = "--entrypoint=bash"
        if len(unknown_args) > 0:
            entrypoint_args.extend(["-c", " ".join(unknown_args)])
    else:
        entrypoint_args.extend(unknown_args)

    # Parse config files and set volume mounts for results and datasets.
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
            ("datasets_path", CONTAINER_DATASETS_DIR_ROOT),
        ]:
            if path_type in config_data:
                path_value = config_data[path_type]
                if path_value.startswith("/"):
                    container_dir_path = combine_dir_paths(container_dir, path_value)
                    volume_mounts.append(f"--volume {path_value}:{container_dir_path}")
                else:
                    msg = f"Path value {path_value} for {path_type} must be an absolute path."
                    raise ValueError(msg)
            else:
                msg = f"Path value {path_type} not found in config file(s)."
                raise ValueError(msg)

    # Add volume mounts for each config file so the script in the container can read each one
    # and add each to ENTRYPOINT_ARGS.
    for config_file in args.config:
        config_file_host = config_file.absolute().expanduser().resolve()
        container_dir_path = combine_dir_paths(CONTAINER_CONFIG_DIR_ROOT, config_file_host)
        volume_mounts.append(f"--volume {config_file_host}:{container_dir_path}")
        # Only add modified --config args if running the benchmark tool entrypoint, not the
        # bash shell entrypoint.
        if not args.shell:
            entrypoint_args.append(f"--config={container_dir_path}")

    # The total available container memory will be the total host memory, but no more
    # than _max_container_memory_bytes.
    # The container shared memory size is a percentage of the container memory.
    container_memory_bytes = min(get_total_memory_bytes(), _max_container_memory_bytes)
    shm_size_bytes = int(container_memory_bytes * _shm_size_container_memory_percentage)

    # Build and return the string to eval in bash.
    eval_str = ""
    eval_str += f"BASH_ENTRYPOINT_OVERRIDE={bash_entrypoint_override}\n"
    eval_str += f"CURATOR_BENCHMARKING_IMAGE={CURATOR_BENCHMARKING_IMAGE}\n"
    eval_str += f"GPUS={GPUS}\n"
    eval_str += f"CONTAINER_MEMORY_BYTES={container_memory_bytes}\n"
    eval_str += f"SHM_SIZE_BYTES={shm_size_bytes}\n"
    eval_str += f"HOST_CURATOR_DIR={HOST_CURATOR_DIR}\n"
    eval_str += f"CURATOR_BENCHMARKING_DEBUG={CURATOR_BENCHMARKING_DEBUG}\n"
    vms = f'"{" ".join(volume_mounts)}"' if volume_mounts else ""
    eval_str += f"VOLUME_MOUNTS={vms}\n"
    eval_str += "ENTRYPOINT_ARGS=(" + " ".join([f'"{arg}"' for arg in entrypoint_args]) + ")\n"
    return eval_str


if __name__ == "__main__":
    print(get_runscript_eval_str(sys.argv))
