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

"""Sharded Manifest Writer -- writes per-shard JSONL files mirroring input paths with .done markers."""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask


@dataclass
class ShardedManifestWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Write AudioTasks to per-shard JSONL files mirroring the input manifest path structure.

    Output structure mirrors the input manifest paths::

        output_dir/
          yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl
          yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl.done

    The shard key is extracted from ``task._metadata["_shard_key"]``
    which is set by ``NemoTarShardReaderStage`` as a relative path
    (e.g. ``yodas/0_from_captions/en/sharded_manifests/manifest_42``).

    Args:
        output_dir: Root directory for output manifests.
    """

    name: str = "sharded_manifest_writer"
    output_dir: str = ""
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for ShardedManifestWriterStage"
            raise ValueError(msg)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ShardedManifestWriterStage: output_dir={self.output_dir}")

    def process(self, task: AudioTask) -> FileGroupTask:
        shard_key = task._metadata.get("_shard_key", "unknown/shard_0")

        out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")

        self._shard_counts[shard_key] += 1

        shard_total = task._metadata.get("_shard_total", 0)
        if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
            done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
            with open(done_path, "w") as f:
                f.write(f"{self._shard_counts[shard_key]}\n")
            logger.info(f"Shard {shard_key} complete: {self._shard_counts[shard_key]} utterances, wrote {done_path}")

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[out_path],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def process_batch(self, tasks: list[AudioTask]) -> list[FileGroupTask]:
        return [self.process(task) for task in tasks]

    def teardown(self) -> None:
        total = sum(self._shard_counts.values())
        done = sum(
            1 for k in self._shard_counts
            if os.path.exists(os.path.join(self.output_dir, f"{k}.jsonl.done"))
        )
        logger.info(f"ShardedManifestWriter: {total} utterances across {len(self._shard_counts)} shards, {done} completed with .done")

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
