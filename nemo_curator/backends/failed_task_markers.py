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
import uuid
from pathlib import Path

from nemo_curator.utils.retry_manifest import METADATA_DIRNAME

FAILED_TASKS_DIR_ENV_VAR = "NEMO_CURATOR_FAILED_TASKS_DIR"
FAILED_TASK_MANIFEST_FILENAME = "failed_tasks.json"


def _configure_failed_task_manifest_dir(default_dir: Path) -> Path:
    existing = os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if existing:
        return Path(existing)

    manifest_dir = default_dir.absolute()
    os.environ[FAILED_TASKS_DIR_ENV_VAR] = str(manifest_dir)
    return manifest_dir


def configure_failed_task_manifest_dir(checkpoint_path: str | Path) -> Path:
    """Configure a local attempt-scoped FailedTask manifest directory unless overridden."""
    manifest_dir = Path(
        checkpoint_path,
        METADATA_DIRNAME,
        ".failed_tasks",
        f"local_attempt_{uuid.uuid4().hex}",
    )
    return _configure_failed_task_manifest_dir(manifest_dir)


def configure_slurm_array_failed_task_manifest_dir(checkpoint_path: str | Path, shard_index: int) -> Path:
    """Configure an attempt-scoped FailedTask manifest directory unless overridden."""
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
    manifest_dir = Path(
        checkpoint_path,
        METADATA_DIRNAME,
        ".failed_tasks",
        f"slurm_job_{job_id}",
        f"array_task_{array_task_id}",
        f"restart_{restart_count}",
        f"shard_{shard_index}",
    )
    return _configure_failed_task_manifest_dir(manifest_dir)


def record_failed_tasks() -> None:
    """Write one attempt-scoped manifest after any FailedTask is detected."""
    manifest_dir = os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not manifest_dir:
        return

    manifest_path = Path(manifest_dir, FAILED_TASK_MANIFEST_FILENAME)
    if manifest_path.is_file():
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.touch(exist_ok=True)


def failed_task_manifest_exists(manifest_dir: str | Path | None = None) -> bool:
    """Return whether the current attempt has recorded any FailedTask."""
    resolved_manifest_dir = manifest_dir if manifest_dir is not None else os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not resolved_manifest_dir:
        return False
    return Path(resolved_manifest_dir, FAILED_TASK_MANIFEST_FILENAME).is_file()
