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

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import git
import pynvml
from loguru import logger
from runner.session import Session
from runner.utils import get_obj_for_json, get_total_memory_bytes, run_shm_size_check


def dump_env(session_obj: Session, output_path: Path) -> dict[str, Any]:
    env_data = get_env()
    env_data["object_store_size"] = session_obj.object_store_size

    # Try package managers in order of preference for capturing the environment
    # package_managers = [("uv", "pip freeze"), ("pip", "freeze"), ("micromamba", "list --explicit"), ("conda", "list --explicit")]  # noqa: ERA001
    package_managers = [("uv", "pip freeze")]
    env_dumped = False
    for package_manager, cmd in package_managers:
        if shutil.which(package_manager):
            cmd_list = [package_manager, *cmd.split(" ")]
            exp = subprocess.check_output(cmd_list, text=True, timeout=120)  # noqa: S603
            packages_txt_path = output_path / "packages.txt"
            packages_txt_path.write_text(exp)
            env_data["packages_txt"] = str(packages_txt_path)
            logger.info(f"Captured packages from {package_manager} {cmd} to {packages_txt_path}")
            env_dumped = True
            break
    if not env_dumped:
        logger.warning(
            f"No package manager ({', '.join([pm for pm, _ in package_managers])}) found in PATH, skipping environment capture"
        )

    # Write env data to file as JSON and return the dictionary written
    (output_path / "env.json").write_text(json.dumps(get_obj_for_json(env_data)))
    return env_data


def get_env() -> dict[str, Any]:
    try:
        import ray

        ray_version = ray.__version__
    except ModuleNotFoundError:
        ray_version = "not_installed"

    git_commit_string = get_git_commit_string()
    cuda_visible_devices = get_gpu_info_string()
    shm_size_bytes, _ = run_shm_size_check(human_readable=False)

    # The image digest is not known at image build time and is not available inside the
    # container, so it must be passed in when the container is run.
    # Get the image digest via the env var set from tools/run.sh
    image_digest = os.getenv("IMAGE_DIGEST", "unknown")

    # Better support container environment usage by looking for the env var HOST_HOSTNAME
    # which can be set to the container hostname (see tools/run.sh) since the name of the
    # host machine is expected by most users.
    return {
        "hostname": os.getenv("HOST_HOSTNAME", platform.node()),
        "platform": platform.platform(),
        "total_system_memory_bytes": get_total_memory_bytes(),
        "shm_size_bytes": shm_size_bytes,
        "ray_version": ray_version,
        "git_commit": git_commit_string,
        "image_digest": image_digest,
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "cuda_visible_devices": cuda_visible_devices,
    }


def get_git_commit_string() -> str:
    """Returns the git commit string for Curator."""
    # Use the directory where this script is located (which is assumed to be the Curator repo)
    # Another option is to use the file location of the nemo_curator __init__.py file, but that may not be the location of the repo if nemo_curator is installed as a package.
    # Note: if the benchmarking tools (i.e. this file and others) eventually become an installable package, this approach may not work if these tools are installed as a package.
    try:
        repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
        commit_str = repo.head.commit.hexsha
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get git commit string: {e}")
        commit_str = "unknown"

    return commit_str


def get_gpu_info_string() -> str:
    """Returns a string describing the GPUs visible to the process.
    If no GPUs are visible, returns "No GPUs found".
    If multiple GPUs are visible, returns a string with the number of each GPU model (e.g. "2X H100").
    """
    try:
        pynvml.nvmlInit()
        # Rather than assume all visible GPUs are the same model (which is by
        # far the most common case), count the number of each model just in case.
        gpu_names_count = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_names_count[name] = gpu_names_count.get(name, 0) + 1
        pynvml.nvmlShutdown()
        if len(gpu_names_count) > 0:
            counts = [f"{count}X {name}" for name, count in gpu_names_count.items()]
            gpu_info_str = ", ".join(counts)
        else:
            gpu_info_str = "No GPUs found"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get GPU info: {e}")
        gpu_info_str = "unknown"

    return gpu_info_str
