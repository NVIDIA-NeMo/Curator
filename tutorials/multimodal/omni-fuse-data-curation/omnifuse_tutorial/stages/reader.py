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

"""Reader stage for paired raw/annotation manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask

from omnifuse_tutorial.compat.curator import make_document_batch
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.loader import load_all_pools


@dataclass
class PairManifestReaderStage(ProcessingStage[Any, Any]):
    config: ExperimentConfig | None = None
    name: str = "PairManifestReader"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "raw_path", "annotation", "modality"]

    def process(self, task: EmptyTask) -> Any:
        if self.config is None:
            raise ValueError("PairManifestReaderStage requires config")
        records = load_all_pools(self.config.data_pools)
        return make_document_batch(
            task_id=f"{self.config.experiment_id}_pairs",
            dataset_name=self.config.experiment_id,
            records=records,
            metadata={"experiment_id": self.config.experiment_id},
        )
