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

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.tasks.sentinels import FailedTask
from nemo_curator.utils.atomic_io import write_json_atomically

FAILED_TASKS_DIR_ENV_VAR = "NEMO_CURATOR_FAILED_TASKS_DIR"
FAILED_TASK_MARKER_PATTERN = "failed_task_*.json"


@dataclass(frozen=True)
class FailedTaskMarker:
    """Identity recorded for one failed stage task."""

    stage_name: str
    task_id: str
    path: Path


# TODO: Single marker
def _write_failed_task_marker(marker_dir: Path, stage_name: str, task: FailedTask) -> None:
    """Write one compact marker for a failed stage/task pair."""
    payload = {
        "stage_name": stage_name,
        "task_id": task.task_id,
    }

    marker_identity = f"{stage_name}\0{task.task_id}".encode()
    marker_digest = hashlib.sha256(marker_identity).hexdigest()[:16]
    filename = f"failed_task_{marker_digest}.json"
    final_path = marker_dir / filename
    write_json_atomically(final_path, payload, separators=(",", ":"), sort_keys=True)


def record_failed_tasks(stage_name: str, failed_tasks: list[FailedTask]) -> None:
    """Record FailedTask markers when ``NEMO_CURATOR_FAILED_TASKS_DIR`` is set."""
    marker_dir = os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not marker_dir or not failed_tasks:
        return

    marker_path = Path(marker_dir)
    for task in failed_tasks:
        try:
            _write_failed_task_marker(marker_path, stage_name, task)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to write FailedTask marker to {marker_path}: {e}")


def read_failed_task_markers(
    marker_dir: str | Path | None = None,
) -> list[FailedTaskMarker]:
    """Read FailedTask identities from ``marker_dir`` or the configured env dir."""
    resolved_marker_dir = marker_dir if marker_dir is not None else os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not resolved_marker_dir:
        return []

    marker_path = Path(resolved_marker_dir).absolute()
    if not marker_path.exists():
        return []

    markers = []
    for path in sorted(marker_path.glob(FAILED_TASK_MARKER_PATTERN)):
        if not path.is_file():
            continue

        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            msg = f"Failed to read FailedTask marker {path}: {e}"
            raise ValueError(msg) from e

        if not isinstance(payload, dict):
            msg = f"FailedTask marker must contain a JSON object: {path}"
            raise ValueError(msg)

        stage_name = payload.get("stage_name")
        task_id = payload.get("task_id")
        if not isinstance(stage_name, str) or not isinstance(task_id, str):
            msg = f"FailedTask marker must contain string stage_name and task_id fields: {path}"
            raise ValueError(msg)

        markers.append(FailedTaskMarker(stage_name=stage_name, task_id=task_id, path=path))

    return markers


def summarize_failed_task_markers(
    marker_dir: str | Path | None = None,
) -> dict[str, object]:
    """Count FailedTask markers in ``marker_dir`` or the configured env dir."""
    resolved_marker_dir = marker_dir if marker_dir is not None else os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not resolved_marker_dir:
        return {
            "failed_task_marker_count": 0,
        }

    marker_path = Path(resolved_marker_dir).absolute()
    if not marker_path.exists():
        return {
            "failed_task_marker_count": 0,
        }

    marker_count = 0
    for path in marker_path.glob(FAILED_TASK_MARKER_PATTERN):
        if not path.is_file():
            continue
        marker_count += 1

    return {
        "failed_task_marker_count": marker_count,
    }
