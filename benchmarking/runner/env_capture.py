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
from pathlib import Path
from typing import Any

from loguru import logger
from runner.utils import get_obj_for_json


def dump_env(output_path: Path) -> dict[str, Any]:
    env_data = get_env()

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
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "ray_version": os.getenv("RAY_VERSION", "unknown"),
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "image_digest": os.getenv("IMAGE_DIGEST", "unknown"),
        "python_version": platform.python_version(),
        "executable": os.getenv("_"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }
