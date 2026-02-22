# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""ALM Manifest Reader Stage — reads JSONL manifests on the worker, not the driver."""

import json
from dataclasses import dataclass
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, _EmptyTask


@dataclass
class ALMManifestReaderStage(ProcessingStage[_EmptyTask, AudioBatch]):
    """Read a JSONL manifest on a worker and produce one AudioBatch per entry.

    This stage accepts an EmptyTask and fans out into individual AudioBatch
    tasks, keeping manifest I/O off the driver. Supports local and cloud
    paths via fsspec.

    Args:
        manifest_path: Path to the input JSONL manifest (local or cloud).
    """

    manifest_path: str
    name: str = "alm_manifest_reader"

    def process(self, _: _EmptyTask) -> list[AudioBatch]:
        fs, path = url_to_fs(self.manifest_path)
        entries: list[dict[str, Any]] = []
        with fs.open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))

        logger.info(f"ALMManifestReaderStage: loaded {len(entries)} entries from {self.manifest_path}")

        return [AudioBatch(data=[entry]) for entry in entries]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}
