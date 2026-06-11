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

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "annotation", "modality"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sns_annotation", "sns_raw_text"]

    def process(self, task: Any) -> Any:
        if self.config is None:
            raise ValueError("SNSStage requires config")
        if self.config.sns.sns_output_dir is None:
            self.config.sns.sns_output_dir = self.config.run_dir / "sns" / "media"
        backend = backend_factory(self.config.sns, self.config.eee, self.config.runtime)
        processor = SNSProcessor(self.config.sns, embedding_dim=self.config.eee.embedding_dim, backend=backend)
        output_records: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for record in records_from_task(task):
            output, manifest = processor.process_record(record)
            output_records.append(output)
            manifest_rows.append(manifest)

        manifest_path = self.config.run_dir / "sns" / "manifest.jsonl"
        write_jsonl(manifest_path, manifest_rows)
        metadata = dict(getattr(task, "_metadata", {}) or {})
        metadata["sns_manifest_path"] = str(manifest_path)
        return make_document_batch(
            dataset_name=task.dataset_name,
            records=output_records,
            metadata=metadata,
            stage_perf=getattr(task, "_stage_perf", []),
        )
