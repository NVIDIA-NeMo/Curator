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

"""ALM Manifest Writer Stage — writes AudioBatch entries to a JSONL manifest."""

import json
from dataclasses import dataclass
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch


@dataclass
class ALMManifestWriterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Append AudioBatch entries to a JSONL manifest file.

    Each processed AudioBatch has its data entries appended to the output
    file. The file is truncated on ``setup()`` so repeated pipeline runs
    produce a clean output. Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
    """

    output_path: str
    name: str = "alm_manifest_writer"

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        fs, path = url_to_fs(self.output_path)
        parent_dir = "/".join(path.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        with fs.open(path, "w", encoding="utf-8"):
            pass
        logger.info(f"ALMManifestWriterStage: writing to {self.output_path}")

    def process(self, task: AudioBatch) -> AudioBatch:
        fs, path = url_to_fs(self.output_path)
        with fs.open(path, "a", encoding="utf-8") as f:
            for entry in task.data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return task

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
