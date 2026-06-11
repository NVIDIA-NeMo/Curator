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

"""Expert Embedding Engine stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omnifuse_tutorial.compat.curator import ProcessingStage, Resources, make_document_batch, records_from_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import write_json, write_npy
from omnifuse_tutorial.eee.backends import BackendFactory, backend_factory
from omnifuse_tutorial.eee.results import EmbeddingBundle


@dataclass
class EEEEmbeddingStage(ProcessingStage[Any, Any]):
    config: ExperimentConfig | None = None
    backend_factory_fn: BackendFactory = backend_factory
    name: str = "EEEEmbedding"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "sns_annotation", "modality"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: Any) -> Any:
        if self.config is None:
            raise ValueError("EEEEmbeddingStage requires config")
        records = records_from_task(task)
        backend = self.backend_factory_fn(self.config.eee, self.config.runtime)
        experts = list(self.config.eee.experts)
        embeddings: dict[str, list[list[float]]] = {}
        for expert in experts:
            rows: list[list[float]] = []
            for record in records:
                rows.append(backend.embed_raw(record, expert))
                rows.append(backend.embed_annotation(record, expert))
            embeddings[expert] = rows

        bundle = EmbeddingBundle(
            pair_ids=[record["pair_id"] for record in records],
            modalities=[record["modality"] for record in records],
            records=records,
            experts=experts,
            embeddings=embeddings,
        )

        output_dir = self.config.run_dir / "embeddings"
        for expert, rows in embeddings.items():
            safe_name = expert.replace("-", "_")
            write_npy(output_dir / f"{safe_name}_interleaved.npy", rows)
            write_npy(output_dir / f"{safe_name}_raw.npy", rows[0::2])
            write_npy(output_dir / f"{safe_name}_annotation.npy", rows[1::2])
        metadata_path = write_json(
            output_dir / "metadata.json",
            {
                "pair_ids": bundle.pair_ids,
                "modalities": bundle.modalities,
                "experts": bundle.experts,
                "embedding_dim": bundle.embedding_dim,
            },
        )

        metadata = dict(getattr(task, "_metadata", {}) or {})
        metadata["embedding_bundle"] = bundle
        metadata["embedding_metadata_path"] = str(metadata_path)
        return make_document_batch(
            task_id=f"{task.task_id}_eee",
            dataset_name=task.dataset_name,
            records=records,
            metadata=metadata,
            stage_perf=getattr(task, "_stage_perf", []),
        )
