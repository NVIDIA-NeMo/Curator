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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.tasks.audio_task import ensure_sample_key

from .io_utils import normalize_for_json, write_json_atomic, write_jsonl_atomic
from .serialization import dump_audio_task_manifest, load_audio_task_manifest, serialize_audio_task

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask

RecordStatus = Literal["done", "filtered", "failed_retriable", "failed_permanent"]


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
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
        self.records_path = self.stage_dir / "records" / "batch_00000.jsonl"
        self.output_manifest_path = self.stage_dir / "outputs" / "manifest.jsonl"

    def is_complete(self) -> bool:
        if not self.stage_json_path.exists() or not self.output_manifest_path.exists():
            return False
        stage_payload = json.loads(self.stage_json_path.read_text())
        return (
            stage_payload.get("status") == "done"
            and stage_payload.get("config_fingerprint") == self.config_fingerprint
        )

    def load_output_tasks(self) -> list[AudioTask]:
        return load_audio_task_manifest(self.output_manifest_path)

    def load_records(self) -> list[SampleCheckpointRecord]:
        if not self.records_path.exists():
            return []
        records: list[SampleCheckpointRecord] = []
        with self.records_path.open("r", encoding="utf-8") as fin:
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
        if input_tasks is not None:
            for task in input_tasks:
                ensure_sample_key(task)

        failed_records = failed_records or []
        failed_by_key = {record.sample_key: record for record in failed_records}
        output_by_key = {task.sample_key: task for task in output_tasks}

        records: list[SampleCheckpointRecord] = [
            SampleCheckpointRecord(
                sample_key=task.sample_key,
                status="done",
                task=serialize_audio_task(task),
            )
            for task in output_tasks
        ]
        if input_tasks is not None:
            for task in input_tasks:
                if task.sample_key in output_by_key or task.sample_key in failed_by_key:
                    continue
                records.append(SampleCheckpointRecord(sample_key=task.sample_key, status="filtered"))
        records.extend(failed_records)

        record_payloads = [record.to_dict() for record in records]
        dump_audio_task_manifest(output_tasks, self.output_manifest_path)
        write_jsonl_atomic(self.records_path, record_payloads)

        status_payload = {
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "config_fingerprint": self.config_fingerprint,
            "status": "done",
            "input_count": len(input_tasks) if input_tasks is not None else None,
            "done_count": sum(record.status == "done" for record in records),
            "filtered_count": sum(record.status == "filtered" for record in records),
            "failed_count": sum(record.status.startswith("failed_") for record in records),
            "updated_at": _utcnow(),
        }
        write_json_atomic(self.stage_json_path, status_payload)
