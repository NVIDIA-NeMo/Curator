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

"""SNS Curator stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

from omnifuse_tutorial.compat.curator import make_document_batch, records_from_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import write_jsonl
from omnifuse_tutorial.sns.backends import backend_factory
from omnifuse_tutorial.sns.processor import SNSProcessor


@dataclass
class SNSStage(ProcessingStage[Any, Any]):
    config: ExperimentConfig | None = None
    name: str = "SNS"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    backend_factory_fn: Any = backend_factory
    sharded_outputs: bool = False
    worker_count: int | None = None
    slots_per_actor: int = 1
    _backend: Any = field(default=None, init=False, repr=False)
    _processor: SNSProcessor | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "annotation", "modality"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sns_annotation", "sns_raw_text"]

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"slots_per_actor": self.slots_per_actor}
        if self.worker_count is not None:
            spec["num_workers"] = self.worker_count
        return spec

    def process(self, task: Any) -> Any:
        if self.config is None:
            raise ValueError("SNSStage requires config")
        if self.config.sns.sns_output_dir is None:
            self.config.sns.sns_output_dir = self.config.run_dir / "sns" / "media"
        processor = self._ensure_processor()
        output_records: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for record in records_from_task(task):
            output, manifest = processor.process_record(record)
            output_records.append(output)
            manifest_rows.append(manifest)

        metadata = dict(getattr(task, "_metadata", {}) or {})
        manifest_path = self._manifest_path(metadata)
        write_jsonl(manifest_path, manifest_rows)
        metadata["sns_manifest_path"] = str(manifest_path)
        return make_document_batch(
            task_id=f"{task.task_id}_sns",
            dataset_name=task.dataset_name,
            records=output_records,
            metadata=metadata,
            stage_perf=getattr(task, "_stage_perf", []),
        )

    def teardown(self) -> None:
        if self._backend is not None:
            unload = getattr(self._backend, "unload", None)
            if callable(unload):
                unload()
        self._backend = None
        self._processor = None

    def _ensure_processor(self) -> SNSProcessor:
        if self.config is None:
            raise ValueError("SNSStage requires config")
        if self._processor is None:
            self._backend = self.backend_factory_fn(self.config.sns, self.config.eee, self.config.runtime)
            self._processor = SNSProcessor(
                self.config.sns,
                embedding_dim=self.config.eee.embedding_dim,
                backend=self._backend,
            )
        return self._processor

    def _manifest_path(self, metadata: dict[str, Any]) -> Any:
        if self.config is None:
            raise ValueError("SNSStage requires config")
        if not self.sharded_outputs:
            return self.config.run_dir / "sns" / "manifest.jsonl"
        shard_index = metadata.get("shard_index")
        if not isinstance(shard_index, int):
            raise ValueError("Sharded SNS tasks require integer shard_index metadata")
        shard_dir = self.config.run_dir / "sns" / "shards" / f"{shard_index:05d}"
        metadata["sns_shard_dir"] = str(shard_dir)
        return shard_dir / "manifest.jsonl"
