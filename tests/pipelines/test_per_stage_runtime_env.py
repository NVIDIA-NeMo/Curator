# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for per-stage runtime environment: different Python package versions per stage.

Each stage declares runtime_env and Ray creates an isolated virtualenv per unique spec set.
Both the RayData and Xenna backends use Ray's native runtime_env mechanism.
"""

from typing import ClassVar

import pandas as pd
import pytest

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class VersionStage1(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage 1: packaging==23.2."""

    name = "version_stage_1"
    resources = Resources(cpus=0.5)
    batch_size = 1
    runtime_env: ClassVar[dict] = {"pip": ["packaging==23.2"]}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_packaging_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import packaging

        batch = task.to_pandas().copy()
        batch["stage1_packaging_version"] = packaging.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=batch,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


class VersionStage2(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage 2: packaging==24.0."""

    name = "version_stage_2"
    resources = Resources(cpus=0.5)
    batch_size = 1
    runtime_env: ClassVar[dict] = {"pip": ["packaging==24.0"]}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_packaging_version"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage2_packaging_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import packaging

        batch = task.to_pandas().copy()
        batch["stage2_packaging_version"] = packaging.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=batch,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@pytest.mark.usefixtures("shared_ray_client")
def test_per_stage_different_package_versions_ray_data() -> None:
    """Run two stages with different packaging versions; assert each sees its own version."""
    initial = DocumentBatch(task_id="test", dataset_name="test", data=pd.DataFrame({"text": ["hello"]}))
    results = Pipeline(name="test", stages=[VersionStage1(), VersionStage2()]).run(
        executor=RayDataExecutor(), initial_tasks=[initial]
    )
    assert results is not None
    result = results[0].to_pandas()
    assert result["stage1_packaging_version"].iloc[0] == "23.2"
    assert result["stage2_packaging_version"].iloc[0] == "24.0"


@pytest.mark.usefixtures("shared_ray_client")
def test_per_stage_different_package_versions_xenna() -> None:
    """Run two stages with different packaging versions using XennaExecutor."""
    from nemo_curator.backends.xenna import XennaExecutor

    initial = DocumentBatch(task_id="test", dataset_name="test", data=pd.DataFrame({"text": ["hello"]}))
    results = Pipeline(name="test", stages=[VersionStage1(), VersionStage2()]).run(
        executor=XennaExecutor(config={"execution_mode": "streaming"}), initial_tasks=[initial]
    )
    assert results is not None
    result = results[0].to_pandas()
    assert result["stage1_packaging_version"].iloc[0] == "23.2"
    assert result["stage2_packaging_version"].iloc[0] == "24.0"
