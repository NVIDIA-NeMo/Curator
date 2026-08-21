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

"""Datablend ranking/export stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

from omnifuse_tutorial.compat.curator import make_document_batch, records_from_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import write_jsonl
from omnifuse_tutorial.datablend.ranker import DatablendRanker
from omnifuse_tutorial.eee.backends import BackendFactory, backend_factory
from omnifuse_tutorial.projection.trainer import ProjectionResult


@dataclass
class DatablendRankingStage(ProcessingStage[Any, Any]):
    config: ExperimentConfig | None = None
    backend_factory_fn: BackendFactory = backend_factory
    name: str = "DatablendRanking"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["rank", "score"]

    def process(self, task: Any) -> Any:
        if self.config is None:
            raise ValueError("DatablendRankingStage requires config")
        metadata = dict(getattr(task, "_metadata", {}) or {})
        projection = metadata.get("projection_result")
        if not isinstance(projection, ProjectionResult):
            raise ValueError("DatablendRankingStage requires projection_result metadata")
        records = records_from_task(task)
        backend = self.backend_factory_fn(self.config.eee, self.config.runtime)
        ranker = DatablendRanker(self.config.datablend, backend)
        ranked = ranker.rank(records, projection)
        selected = ranker.select_top(ranked)

        output_dir = self.config.run_dir / "datablend"
        ranked_path = write_jsonl(output_dir / "datablend_ranked.jsonl", ranked)
        topk_path = write_jsonl(output_dir / "datablend_topk.jsonl", selected)
        metadata.update(
            {
                "datablend_ranked_path": str(ranked_path),
                "datablend_topk_path": str(topk_path),
                "datablend_size": len(selected),
            }
        )
        return make_document_batch(
            task_id=f"{task.task_id}_datablend",
            dataset_name=task.dataset_name,
            records=selected,
            metadata=metadata,
            stage_perf=getattr(task, "_stage_perf", []),
        )
