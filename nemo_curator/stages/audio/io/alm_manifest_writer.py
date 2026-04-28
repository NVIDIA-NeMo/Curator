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

"""ALM Manifest Writer Stage — writes AudioTask dicts to a JSONL manifest."""

import json
from dataclasses import dataclass
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask


@dataclass
class ALMManifestWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Append a single AudioTask to a JSONL manifest file.

    The output file is truncated once per node in ``setup_on_node()``
    so repeated pipeline runs produce a clean output.
    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
    """

    name: str = "alm_manifest_writer"
    output_path: str = ""

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ALMManifestWriterStage"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        fs, path = url_to_fs(self.output_path)
        parent_dir = "/".join(path.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        with fs.open(path, "w", encoding="utf-8"):
            pass
        logger.info(f"ALMManifestWriterStage: writing to {self.output_path}")

    def process(self, task: AudioTask) -> FileGroupTask:
        fs, path = url_to_fs(self.output_path)
        with fs.open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[self.output_path],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
