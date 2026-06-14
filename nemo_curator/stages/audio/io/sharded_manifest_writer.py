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
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io.manifest_writer_utils import (
    AudioManifestWriterMetrics,
    manifest_lines,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask


@dataclass
class ShardedManifestWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Write AudioTasks to per-shard JSONL files mirroring the input manifest paths.

    Output mirrors the input manifest paths, e.g.
    ``output_dir/yodas/.../manifest_42.jsonl`` plus a ``.jsonl.done`` marker, with
    an aggregate ``perf_summary.json`` at the root. The shard key comes from
    ``task._metadata["_shard_key"]`` (set by ``NemoTarShardReaderStage`` as a
    relative path).

    Args:
        output_dir: Root directory for output manifests.
        final_manifest_path: Optional aggregate JSONL rebuilt from completed
            shard outputs at teardown; sharded files stay primary.
        write_perf_stats: If True, record per-task stage perf into the aggregate
            and refresh ``perf_summary.json`` on each shard completion.
    """

    output_dir: str
    name: str = "sharded_manifest_writer"
    final_manifest_path: str | None = None
    write_perf_stats: bool = True
    duration_key: str = "duration"
    drop_manifest_keys: tuple[str, ...] = ("waveform",)
    _writer_metrics: AudioManifestWriterMetrics = field(init=False, repr=False)
    _final_shards_materialized: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._writer_metrics = AudioManifestWriterMetrics(
            stage_name=self.name,
            duration_key=self.duration_key,
            write_perf_stats=self.write_perf_stats,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def _has_completed_shards(self) -> bool:
        if not os.path.isdir(self.output_dir):
            return False
        for _root, _dirs, files in os.walk(self.output_dir):
            if any(name.endswith(".jsonl.done") for name in files):
                return True
        return False

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._final_shards_materialized = set()
        if self.final_manifest_path:
            final_parent = os.path.dirname(self.final_manifest_path)
            if final_parent:
                os.makedirs(final_parent, exist_ok=True)
            if os.path.exists(self.final_manifest_path):
                if self._has_completed_shards():
                    self._final_shards_materialized.update(self._completed_shard_keys())
                    logger.info(
                        "Preserving final manifest until teardown rebuild: {}",
                        self.final_manifest_path,
                    )
                else:
                    os.remove(self.final_manifest_path)
        self._writer_metrics.reset_wall_timer()
        logger.info(f"ShardedManifestWriterStage: output_dir={self.output_dir}")

    @staticmethod
    def _shard_key_of(task: AudioTask) -> str:
        return task._metadata.get("_shard_key", "unknown/shard_0")

    def _write_shard_group(self, shard_key: str, group: list[AudioTask]) -> str:
        """Persist all utterances of one shard with one open/close per file.

        Rows are serialized in memory and written with a single ``writelines``
        (one open per shard manifest, not one per utterance).
        """
        out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        lines = manifest_lines(group, self.drop_manifest_keys)
        write_t0 = time.perf_counter()
        with open(out_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        self._writer_metrics.add_manifest_write_time(time.perf_counter() - write_t0)

        for task in group:
            self._writer_metrics.record_task(task, shard_key=shard_key)

        # Completion: the reader stamps every utterance with the shard's total.
        shard_total = group[-1]._metadata.get("_shard_total", 0)
        if shard_total > 0 and self._writer_metrics.shard_count(shard_key) >= shard_total:
            done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
            done_t0 = time.perf_counter()
            with open(done_path, "w") as f:
                f.write(f"{self._writer_metrics.shard_count(shard_key)}\n")
            self._writer_metrics.add_done_write_time(time.perf_counter() - done_t0)
            logger.info(
                f"Shard {shard_key} complete: "
                f"{self._writer_metrics.shard_count(shard_key)} utterances, wrote {done_path}"
            )
            self._append_completed_shard_to_final(shard_key, out_path)
            if self.write_perf_stats:
                self._write_perf_summary()
        return out_path

    def process(self, task: AudioTask) -> FileGroupTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[FileGroupTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        self._writer_metrics.record_invocation(len(tasks))

        # Group by shard (dict preserves first-seen order) so each shard writes
        # in a single open/append rather than once per utterance.
        groups: dict[str, list[AudioTask]] = {}
        for task in tasks:
            groups.setdefault(self._shard_key_of(task), []).append(task)
        out_path_by_shard = {
            shard_key: self._write_shard_group(shard_key, group) for shard_key, group in groups.items()
        }

        return [
            FileGroupTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                data=[out_path_by_shard[self._shard_key_of(task)]],
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )
            for task in tasks
        ]

    def _write_perf_summary(self) -> None:
        """Write aggregate perf_summary.json at the output root."""
        summary = self._writer_metrics.build_perf_summary()
        summary_path = os.path.join(self.output_dir, "perf_summary.json")
        write_t0 = time.perf_counter()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._writer_metrics.add_perf_write_time(time.perf_counter() - write_t0)
        logger.info(f"Wrote perf_summary.json: {summary_path}")

    def _completed_shard_manifest_paths(self) -> list[str]:
        """Return completed shard JSONL paths, excluding the aggregate final manifest."""
        if not os.path.isdir(self.output_dir):
            return []
        final_abs = os.path.abspath(self.final_manifest_path) if self.final_manifest_path else ""
        paths: list[str] = []
        for root, _dirs, files in os.walk(self.output_dir):
            for fname in files:
                if not fname.endswith(".jsonl.done"):
                    continue
                manifest_path = os.path.join(root, fname[:-len(".done")])
                if not os.path.isfile(manifest_path):
                    continue
                if final_abs and os.path.abspath(manifest_path) == final_abs:
                    continue
                paths.append(manifest_path)
        return sorted(paths)

    def _completed_shard_keys(self) -> set[str]:
        """Return shard keys whose done markers are present."""
        output_abs = os.path.abspath(self.output_dir)
        keys: set[str] = set()
        for manifest_path in self._completed_shard_manifest_paths():
            rel_path = os.path.relpath(os.path.abspath(manifest_path), output_abs)
            keys.add(rel_path.removesuffix(".jsonl"))
        return keys

    def _append_completed_shard_to_final(self, shard_key: str, shard_path: str) -> None:
        """Append one completed shard into the aggregate manifest for eager consumers."""
        if not self.final_manifest_path or shard_key in self._final_shards_materialized:
            return
        final_parent = os.path.dirname(self.final_manifest_path)
        if final_parent:
            os.makedirs(final_parent, exist_ok=True)

        write_t0 = time.perf_counter()
        with open(self.final_manifest_path, "a", encoding="utf-8") as out_f, open(
            shard_path,
            encoding="utf-8",
        ) as in_f:
            out_f.writelines(in_f)
        self._writer_metrics.add_manifest_write_time(time.perf_counter() - write_t0)
        self._final_shards_materialized.add(shard_key)
        logger.info(
            "Appended completed shard {} into final manifest {}",
            shard_key,
            self.final_manifest_path,
        )

    def _write_final_manifest_from_shards(self) -> None:
        """Rebuild the aggregate final manifest from completed shard outputs."""
        if not self.final_manifest_path:
            return
        final_parent = os.path.dirname(self.final_manifest_path)
        if final_parent:
            os.makedirs(final_parent, exist_ok=True)

        shard_paths = self._completed_shard_manifest_paths()
        tmp_path = f"{self.final_manifest_path}.tmp"
        write_t0 = time.perf_counter()
        with open(tmp_path, "w", encoding="utf-8") as out_f:
            for shard_path in shard_paths:
                with open(shard_path, encoding="utf-8") as in_f:
                    out_f.writelines(in_f)
        os.replace(tmp_path, self.final_manifest_path)
        self._writer_metrics.add_manifest_write_time(time.perf_counter() - write_t0)
        self._final_shards_materialized = self._completed_shard_keys()
        logger.info(
            "Rebuilt final manifest {} from {} completed shard file(s)",
            self.final_manifest_path,
            len(shard_paths),
        )

    def teardown(self) -> None:
        self._write_final_manifest_from_shards()

        total = self._writer_metrics.total_utterances
        done = sum(
            1 for k in self._writer_metrics.shard_keys
            if os.path.exists(os.path.join(self.output_dir, f"{k}.jsonl.done"))
        )
        logger.info(
            f"ShardedManifestWriter: {total} utterances across "
            f"{len(self._writer_metrics.shard_keys)} shards, {done} completed with .done"
        )

        if self.write_perf_stats and (
            self._writer_metrics.items_processed > 0 or self._writer_metrics.total_utterances > 0
        ):
            self._write_perf_summary()
        elif self.write_perf_stats:
            logger.info("Skipping perf_summary.json write because no tasks were processed")

    def num_workers(self) -> int | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
