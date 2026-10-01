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

"""Pipeline construction and execution facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omnifuse_tutorial.compat.curator import make_curator_pipeline, make_empty_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import write_json
from omnifuse_tutorial.eee.backends import BackendFactory, backend_factory
from omnifuse_tutorial.stages import (
    DatablendRankingStage,
    EEEEmbeddingStage,
    PairManifestReaderStage,
    ProjectionTrainingStage,
    SNSStage,
)


@dataclass
class OmniFusePipeline:
    config: ExperimentConfig
    curator_pipeline: Any

    def run(self) -> dict[str, Any]:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.config.run_dir / "config.resolved.json", self.config.to_dict())
        tasks = self.curator_pipeline.run(initial_tasks=[make_empty_task()])
        if not tasks:
            raise RuntimeError("Pipeline produced no output tasks")
        final_task = tasks[-1]
        metadata = dict(getattr(final_task, "_metadata", {}) or {})
        return {
            "run_dir": str(self.config.run_dir),
            "sns_manifest_path": metadata.get("sns_manifest_path"),
            "embedding_metadata_path": metadata.get("embedding_metadata_path"),
            "projection_model_path": metadata.get("projection_model_path"),
            "projection_loss_path": metadata.get("projection_loss_path"),
            "projection_metrics_path": metadata.get("projection_metrics_path"),
            "projected_embeddings_path": metadata.get("projected_embeddings_path"),
            "datablend_ranked_path": metadata.get("datablend_ranked_path"),
            "datablend_topk_path": metadata.get("datablend_topk_path"),
            "datablend_size": metadata.get("datablend_size"),
        }


def build_pipeline(
    config: ExperimentConfig,
    backend_factory_fn: BackendFactory = backend_factory,
) -> OmniFusePipeline:
    stages = [
        PairManifestReaderStage(config=config),
        SNSStage(config=config),
        EEEEmbeddingStage(config=config, backend_factory_fn=backend_factory_fn),
        ProjectionTrainingStage(config=config),
        DatablendRankingStage(config=config, backend_factory_fn=backend_factory_fn),
    ]
    curator_pipeline = make_curator_pipeline(
        name=f"omnifuse-{config.experiment_id}",
        description="Omni-Fuse SNS, EEE, projection, and datablend pipeline",
        stages=stages,
    )
    return OmniFusePipeline(config=config, curator_pipeline=curator_pipeline)
