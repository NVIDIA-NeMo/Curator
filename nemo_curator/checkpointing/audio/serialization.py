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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemo_curator.tasks import AudioTask
from nemo_curator.tasks.audio_task import ensure_sample_key
from nemo_curator.utils.performance_utils import StagePerfStats

from .io_utils import write_text_atomic


def serialize_audio_task(task: AudioTask) -> dict[str, Any]:
    """Convert an AudioTask into a JSON-serializable payload."""
    return {
        "task_id": task.task_id,
        "dataset_name": task.dataset_name,
        "data": dict(task.data),
        "sample_key": ensure_sample_key(task),
        "filepath_key": task.filepath_key,
        "_metadata": task._metadata,
        "_stage_perf": [perf.to_dict() for perf in task._stage_perf],
    }


def deserialize_audio_task(payload: dict[str, Any]) -> AudioTask:
    """Reconstruct an AudioTask from serialized checkpoint payload."""
    return AudioTask(
        task_id=payload["task_id"],
        dataset_name=payload["dataset_name"],
        data=payload["data"],
        sample_key=payload.get("sample_key", ""),
        filepath_key=payload.get("filepath_key"),
        _metadata=payload.get("_metadata", {}),
        _stage_perf=[StagePerfStats(**perf) for perf in payload.get("_stage_perf", [])],
    )


def dump_audio_task_manifest(tasks: list[AudioTask], path: str | Path) -> None:
    """Write tasks as JSONL manifest."""
    manifest_path = Path(path)
    text = "\n".join(json.dumps(serialize_audio_task(task), sort_keys=True) for task in tasks)
    if text:
        text += "\n"
    write_text_atomic(manifest_path, text)


def load_audio_task_manifest(path: str | Path) -> list[AudioTask]:
    """Read tasks from JSONL manifest."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []

    tasks: list[AudioTask] = []
    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            tasks.append(deserialize_audio_task(json.loads(line)))
    return tasks
