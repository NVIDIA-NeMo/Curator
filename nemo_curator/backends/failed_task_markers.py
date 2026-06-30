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

import os
from pathlib import Path

from loguru import logger

from nemo_curator.tasks.sentinels import FailedTask
from nemo_curator.utils.atomic_io import write_json_atomically

FAILED_TASKS_DIR_ENV_VAR = "NEMO_CURATOR_FAILED_TASKS_DIR"
FAILED_TASK_MANIFEST_FILENAME = "failed_tasks.json"


def record_failed_tasks(_stage_name: str, failed_tasks: list[FailedTask]) -> None:
    """Write one attempt-scoped manifest after any FailedTask is detected."""
    manifest_dir = os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not manifest_dir or not failed_tasks:
        return

    manifest_path = Path(manifest_dir, FAILED_TASK_MANIFEST_FILENAME)
    if manifest_path.is_file():
        return

    try:
        write_json_atomically(
            manifest_path,
            {"status": "failed_tasks"},
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to write FailedTask manifest to {manifest_path}: {e}")


def failed_task_manifest_exists(manifest_dir: str | Path | None = None) -> bool:
    """Return whether the current attempt has recorded any FailedTask."""
    resolved_manifest_dir = manifest_dir if manifest_dir is not None else os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not resolved_manifest_dir:
        return False
    return Path(resolved_manifest_dir, FAILED_TASK_MANIFEST_FILENAME).is_file()
