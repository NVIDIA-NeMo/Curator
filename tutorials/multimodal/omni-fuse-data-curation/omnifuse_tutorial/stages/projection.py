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

"""Projection training stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omnifuse_tutorial.compat.curator import ProcessingStage, Resources, make_document_batch, records_from_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import write_json, write_npy
from omnifuse_tutorial.eee.results import EmbeddingBundle
from omnifuse_tutorial.projection.trainer import ProjectionTrainer


@dataclass
class ProjectionTrainingStage(ProcessingStage[Any, Any]):
    config: ExperimentConfig | None = None
    name: str = "ProjectionTraining"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: Any) -> Any:
        if self.config is None:
            raise ValueError("ProjectionTrainingStage requires config")
        metadata = dict(getattr(task, "_metadata", {}) or {})
        bundle = metadata.get("embedding_bundle")
        if not isinstance(bundle, EmbeddingBundle):
            raise ValueError("ProjectionTrainingStage requires embedding_bundle metadata")

        trainer = ProjectionTrainer(self.config.projection)
        result = trainer.train_and_project(bundle)
        output_dir = self.config.run_dir / "projection"
        projected_path = write_npy(output_dir / "projected_embeddings.npy", result.projected_raw)
        annotations_path = write_npy(output_dir / "annotation_embeddings.npy", result.annotation_embeddings)
        model_path = write_json(output_dir / "model.json", result.model)
        loss_path = write_json(output_dir / "loss_history.json", {"loss": result.loss_history})
        metrics_path = write_json(output_dir / "metrics.json", {"recall_at_10": result.recall_at_10})

        metadata.update(
            {
                "projection_result": result,
                "projection_model_path": str(model_path),
                "projection_loss_path": str(loss_path),
                "projection_metrics_path": str(metrics_path),
                "projected_embeddings_path": str(projected_path),
                "annotation_embeddings_path": str(annotations_path),
            }
        )
        return make_document_batch(
            task_id=f"{task.task_id}_projection",
            dataset_name=task.dataset_name,
            records=records_from_task(task),
            metadata=metadata,
            stage_perf=getattr(task, "_stage_perf", []),
        )
