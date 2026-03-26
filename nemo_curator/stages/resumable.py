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

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fsspec.core import url_to_fs
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.tasks import Task


def _is_json_serializable(value: object) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    else:
        return True


class ResumableStage:
    """Mixin for stages that can record per-task completion to a checkpoint directory.

    The last ResumableStage in the pipeline (marked _is_pipeline_checkpoint=True by the
    executor) writes to {checkpoint_dir}/pipeline_complete/{task_id}.json.
    All other ResumableStages write to {checkpoint_dir}/{stage_name}/{task_id}.json.

    Concrete classes must expose ``checkpoint_dir`` and ``resume`` instance attributes
    (typically as dataclass fields) and a ``name`` attribute (inherited from ProcessingStage).
    """

    # Defaults — concrete subclasses override via dataclass fields.
    checkpoint_dir: str | None = None
    resume: bool = False
    # Set to True by the executor on the last ResumableStage before serializing to workers.
    _is_pipeline_checkpoint: bool = False

    @property
    def _completed_dir(self) -> str:
        base = self.checkpoint_dir.rstrip("/")  # type: ignore[union-attr]
        if self._is_pipeline_checkpoint:
            return f"{base}/pipeline_complete"
        return f"{base}/{self.name}"  # type: ignore[attr-defined]

    def record_completion(self, input_task: Task, output_task: Task) -> None:
        """Write a JSON completion record to ``_completed_dir/{checkpoint_key}.json``.

        Called from inside ``process()`` after the output is committed to storage.
        Safe to call concurrently from multiple actors — each task gets its own file.

        The checkpoint key is taken from ``input_task._metadata["original_task_id"]`` when
        present (set by FilePartitioningStage before readers transform the task_id).
        This ensures the filename matches what ResumableInputStage looks for on resume.
        """
        from nemo_curator.tasks import FileGroupTask

        fs, path = url_to_fs(self._completed_dir)
        fs.makedirs(path, exist_ok=True)
        record = {
            "task_id": input_task.task_id,
            "dataset_name": input_task.dataset_name,
            "output_paths": output_task.data if isinstance(output_task, FileGroupTask) else [],
            "metadata": {k: v for k, v in input_task._metadata.items() if _is_json_serializable(v)},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint_key = input_task._metadata.get("original_task_id", input_task.task_id)
        record_path = f"{path}/{checkpoint_key}.json"
        with fs.open(record_path, "w") as f:
            json.dump(record, f)
        logger.debug(f"Recorded completion for task '{checkpoint_key}' to {record_path}")


class ResumableInputStage(ResumableStage):
    """Mixin for input/reader stages that self-filter already-completed inputs.

    These stages read from ``{checkpoint_dir}/pipeline_complete/`` to know which
    task_ids have already been fully processed through the entire pipeline and
    only emit tasks for incomplete inputs.
    """

    def load_completed_task_ids(self) -> set[str]:
        """Return the set of task_ids that completed the full pipeline.

        Performs a single ``fs.ls()`` on ``pipeline_complete/`` and returns the
        filenames (without ``.json`` extension) as a set of completed task ids.
        Returns an empty set when ``checkpoint_dir`` is not set or the directory
        does not exist yet.
        """
        if not self.checkpoint_dir:
            return set()
        pipeline_complete_dir = f"{self.checkpoint_dir.rstrip('/')}/pipeline_complete"
        fs, path = url_to_fs(pipeline_complete_dir)
        if not fs.exists(path):
            return set()
        return {os.path.splitext(os.path.basename(f))[0] for f in fs.ls(path, detail=False) if f.endswith(".json")}
