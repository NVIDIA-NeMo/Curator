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

"""Sharded Manifest Writer -- writes per-corpus/shard JSONL with .done markers."""

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
    """Write AudioTasks to per-corpus/shard JSONL files with .done markers.

    Output structure::

        output_dir/
          {corpus}/
            {shard_id}.jsonl
            {shard_id}.done    # written after all utterances in the shard

    On resume, shards with ``.done`` files are skipped by the discovery stage.
    Partial shards (no ``.done``) are deleted and re-processed.

    The shard key is extracted from ``task._metadata["_shard_key"]``
    which is set by ``NemoTarShardReaderStage`` as ``{corpus}_{shard_idx}``.

    Args:
        output_dir: Root directory for output manifests.
    """

    name: str = "sharded_manifest_writer"
    output_dir: str = ""
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _shard_expected: dict[str, int] = field(default_factory=dict, repr=False)

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

    def _get_shard_info(self, task: AudioTask) -> tuple[str, str]:
        """Extract corpus and shard_id from task metadata."""
        shard_key = task._metadata.get("_shard_key", "")
        if "_" in shard_key:
            parts = shard_key.rsplit("_", 1)
            return parts[0], parts[1]
        corpus = task.data.get("corpus", task.dataset_name or "unknown")
        shard_id = str(task.data.get("shard_id", "0"))
        return corpus, shard_id

    def process(self, task: AudioTask) -> FileGroupTask:
        corpus, shard_id = self._get_shard_info(task)
        shard_key = f"{corpus}_{shard_id}"

        corpus_dir = os.path.join(self.output_dir, corpus)
        os.makedirs(corpus_dir, exist_ok=True)

        out_path = os.path.join(corpus_dir, f"{shard_id}.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")

        self._shard_counts[shard_key] += 1

        shard_total = task._metadata.get("_shard_total", 0)
        if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
            done_path = os.path.join(corpus_dir, f"{shard_id}.jsonl.done")
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
        results = []
        for task in tasks:
            results.append(self.process(task))

        completed = set()
        for task in tasks:
            corpus, shard_id = self._get_shard_info(task)
            shard_key = f"{corpus}_{shard_id}"
            if shard_key not in completed:
                completed.add(shard_key)

        return results

    def teardown(self) -> None:
        total = sum(self._shard_counts.values())
        done = sum(1 for k in self._shard_counts if os.path.exists(
            os.path.join(self.output_dir, k.rsplit("_", 1)[0], f"{k.rsplit('_', 1)[1]}.jsonl.done")
        ) if "_" in k)
        logger.info(f"ShardedManifestWriter: {total} utterances across {len(self._shard_counts)} shards, {done} completed with .done")

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
