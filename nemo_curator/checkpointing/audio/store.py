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

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.tasks.audio_task import ensure_checkpoint_shard_id, ensure_sample_key

from .io_utils import normalize_for_json, write_json_atomic, write_jsonl_atomic
from .serialization import dump_audio_task_manifest, load_audio_task_manifest, serialize_audio_task

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask

RecordStatus = Literal["done", "filtered", "failed_retriable", "failed_permanent"]


def _utcnow() -> str:
    return datetime.now(tz=UTC).isoformat()


def fingerprint_stage(stage: Any) -> str:  # noqa: ANN401
    payload = {
        "class_name": type(stage).__name__,
        "config": normalize_for_json(stage.get_config()),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


@dataclass
class SampleCheckpointRecord:
    sample_key: str
    status: RecordStatus
    checkpoint_shard_id: str | None = None
    task: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempt: int = 1
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageCheckpointStore:
    checkpoint_dir: str | Path
    stage_index: int
    stage_name: str
    config_fingerprint: str

    def __post_init__(self) -> None:
        self.root_dir = Path(self.checkpoint_dir)
        self.stage_dir = self.root_dir / f"{self.stage_index:02d}_{self.stage_name}"
        self.stage_json_path = self.stage_dir / "stage.json"
        self.records_dir = self.stage_dir / "records"
        self.outputs_dir = self.stage_dir / "outputs"

    def is_complete(self) -> bool:
        if not self.stage_json_path.exists():
            return False
        stage_payload = json.loads(self.stage_json_path.read_text())
        return (
            stage_payload.get("status") == "done"
            and stage_payload.get("config_fingerprint") == self.config_fingerprint
        )

    def load_output_tasks(self) -> list[AudioTask]:
        tasks: list[AudioTask] = []
        for manifest_path in sorted(self.outputs_dir.glob("*.jsonl")):
            tasks.extend(load_audio_task_manifest(manifest_path))
        return tasks

    def load_records(self) -> list[SampleCheckpointRecord]:
        if not self.records_dir.exists():
            return []
        records: list[SampleCheckpointRecord] = []
        for records_path in sorted(self.records_dir.glob("*.jsonl")):
            with records_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    records.append(SampleCheckpointRecord(**json.loads(line)))
        return records

    def mark_failed(self, error: Exception) -> None:
        payload = {
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "config_fingerprint": self.config_fingerprint,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "updated_at": _utcnow(),
        }
        write_json_atomic(self.stage_json_path, payload)

    def write_stage_result(
        self,
        *,
        input_tasks: list[AudioTask] | None,
        output_tasks: list[AudioTask],
        failed_records: list[SampleCheckpointRecord] | None = None,
    ) -> None:
        for task in output_tasks:
            ensure_sample_key(task)
            ensure_checkpoint_shard_id(task)
        if input_tasks is not None:
            for task in input_tasks:
                ensure_sample_key(task)
                ensure_checkpoint_shard_id(task)

        failed_records = failed_records or []
        failed_by_key = {record.sample_key: record for record in failed_records}
        output_by_key = {task.sample_key: task for task in output_tasks}
        if len(output_by_key) != len(output_tasks):
            logger.warning(
                "Duplicate sample_keys detected in {} output tasks for stage checkpoint {}",
                len(output_tasks),
                self.stage_name,
            )

        records: list[SampleCheckpointRecord] = [
            SampleCheckpointRecord(
                sample_key=task.sample_key,
                status="done",
                checkpoint_shard_id=ensure_checkpoint_shard_id(task),
                task=serialize_audio_task(task),
            )
            for task in output_tasks
        ]
        if input_tasks is not None:
            for task in input_tasks:
                if task.sample_key in output_by_key or task.sample_key in failed_by_key:
                    continue
                records.append(
                    SampleCheckpointRecord(
                        sample_key=task.sample_key,
                        status="filtered",
                        checkpoint_shard_id=ensure_checkpoint_shard_id(task),
                    )
                )
        records.extend(failed_records)

        self._clear_jsonl_files(self.outputs_dir)
        self._clear_jsonl_files(self.records_dir)
        output_tasks_by_shard = self._group_tasks_by_shard(output_tasks)
        for shard_id, shard_tasks in output_tasks_by_shard.items():
            dump_audio_task_manifest(shard_tasks, self.outputs_dir / f"{shard_id}.jsonl")

        records_by_shard = self._group_records_by_shard(records)
        for shard_id, shard_records in records_by_shard.items():
            record_payloads = [record.to_dict() for record in shard_records]
            write_jsonl_atomic(self.records_dir / f"{shard_id}.jsonl", record_payloads)

        status_payload = {
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "config_fingerprint": self.config_fingerprint,
            "status": "done",
            "input_count": len(input_tasks) if input_tasks is not None else None,
            "done_count": sum(record.status == "done" for record in records),
            "filtered_count": sum(record.status == "filtered" for record in records),
            "failed_count": sum(record.status.startswith("failed_") for record in records),
            "shards": sorted(records_by_shard),
            "updated_at": _utcnow(),
        }
        write_json_atomic(self.stage_json_path, status_payload)

    def _group_tasks_by_shard(self, tasks: list[AudioTask]) -> dict[str, list[AudioTask]]:
        grouped: dict[str, list[AudioTask]] = {}
        for task in tasks:
            shard_id = ensure_checkpoint_shard_id(task)
            grouped.setdefault(shard_id, []).append(task)
        return grouped

    def _group_records_by_shard(
        self,
        records: list[SampleCheckpointRecord],
    ) -> dict[str, list[SampleCheckpointRecord]]:
        grouped: dict[str, list[SampleCheckpointRecord]] = {}
        for record in records:
            shard_id = record.checkpoint_shard_id or "partition_unknown"
            grouped.setdefault(shard_id, []).append(record)
        return grouped

    @staticmethod
    def _clear_jsonl_files(directory: Path) -> None:
        if not directory.exists():
            return
        for path in directory.glob("*.jsonl"):
            path.unlink()
