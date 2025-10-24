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

    try:
        freeze = subprocess.check_output(["pip", "freeze"], text=True, timeout=120)  # noqa: S603, S607
        freeze_txt_path = output_path / "pip-freeze.txt"
        freeze_txt_path.write_text(freeze)
        env_data["pip_freeze_txt"] = freeze_txt_path
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to capture pip freeze: {e}")
    try:
        # Try micromamba first, then conda as fallback
        cmd = None
        if shutil.which("micromamba"):
            cmd = ["micromamba", "list", "--explicit"]
        elif shutil.which("conda"):
            cmd = ["conda", "list", "--explicit"]

        if cmd:
            exp = subprocess.check_output(cmd, text=True, timeout=120)  # noqa: S603
            conda_explicit_txt_path = output_path / "conda-explicit.txt"
            conda_explicit_txt_path.write_text(exp)
            env_data["conda_explicit_txt"] = conda_explicit_txt_path
        else:
            logger.warning("Neither micromamba nor conda found in PATH, skipping conda-explicit.txt")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to capture conda list: {e}")

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
