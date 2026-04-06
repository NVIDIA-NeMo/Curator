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
RayData, Xenna, and RayActorPool backends are all tested via parametrization.
"""

from typing import Any, ClassVar

import pandas as pd
import pytest

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


def _record_packaging_and_loguru(task: DocumentBatch, version_col: str, loguru_col: str) -> DocumentBatch:
    """Helper: record packaging.__version__ and whether loguru is importable."""
    import packaging

    try:
        from loguru import logger

        loguru_available = logger is not None
    except ImportError:
        loguru_available = False

    batch = task.to_pandas().copy()
    batch[version_col] = packaging.__version__
    batch[loguru_col] = loguru_available
    return DocumentBatch(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        data=batch,
        _metadata=task._metadata,
        _stage_perf=task._stage_perf,
    )


class BaseEnvStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage with no runtime_env — runs in the base environment."""

    name = "base_env"
    resources = Resources(cpus=0.5)
    batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["base_packaging_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import packaging

        batch = task.to_pandas().copy()
        batch["base_packaging_version"] = packaging.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=batch,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


class VersionStage1(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage 1: packaging==23.2."""

    name = "version_stage_1"
    resources = Resources(cpus=0.5)
    batch_size = 1
    runtime_env: ClassVar[dict] = {"pip": ["packaging==23.2"]}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["base_packaging_version"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_packaging_version", "stage1_loguru_available"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return _record_packaging_and_loguru(task, "stage1_packaging_version", "stage1_loguru_available")


class VersionStage2(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage 2: packaging==24.0."""

    name = "version_stage_2"
    resources = Resources(cpus=0.5)
    batch_size = 1
    runtime_env: ClassVar[dict] = {"pip": ["packaging==24.0"]}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_packaging_version", "stage1_loguru_available"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage2_packaging_version", "stage2_loguru_available"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return _record_packaging_and_loguru(task, "stage2_packaging_version", "stage2_loguru_available")


@pytest.mark.parametrize(
    "backend_config",
    [
        pytest.param((RayDataExecutor, {}), id="ray_data"),
        pytest.param((XennaExecutor, {"execution_mode": "streaming"}), id="xenna_streaming"),
        pytest.param((XennaExecutor, {"execution_mode": "batch"}), id="xenna_batch"),
        pytest.param((RayActorPoolExecutor, {}), id="ray_actor_pool"),
    ],
    indirect=True,
)
class TestPerStageRuntimeEnv:
    """Stages with different runtime_env see different package versions across all backends."""

    backend_cls: type[BaseExecutor] | None = None
    config: dict[str, Any] | None = None
    results: list[DocumentBatch] | None = None

    @pytest.fixture(scope="class", autouse=True)
    def backend_config(self, request: pytest.FixtureRequest, shared_ray_cluster: str):
        """Run the three-stage pipeline once per backend and store results."""
        backend_cls, config = request.param
        request.cls.backend_cls = backend_cls
        request.cls.config = config

        initial = DocumentBatch(
            task_id="test",
            dataset_name="test",
            data=pd.DataFrame({"text": ["hello"]}),
        )
        pipeline = Pipeline(
            name="per_stage_runtime_env_test",
            stages=[BaseEnvStage(), VersionStage1(), VersionStage2()],
        )
        request.cls.results = pipeline.run(executor=backend_cls(config), initial_tasks=[initial])

    def test_stage1_packaging_version(self):
        assert self.results is not None
        result = self.results[0].to_pandas()
        assert result["stage1_packaging_version"].iloc[0] == "23.2"

    def test_stage2_packaging_version(self):
        assert self.results is not None
        result = self.results[0].to_pandas()
        assert result["stage2_packaging_version"].iloc[0] == "24.0"

    def test_base_env_uses_installed_version(self):
        """Stage with no runtime_env sees the base environment's packaging version."""
        assert self.results is not None
        result = self.results[0].to_pandas()
        assert "base_packaging_version" in result.columns
        assert result["base_packaging_version"].iloc[0]  # non-empty

    def test_all_three_versions_differ(self):
        """Base env, 23.2, and 24.0 must all be distinct."""
        assert self.results is not None
        result = self.results[0].to_pandas()
        versions = {
            result["base_packaging_version"].iloc[0],
            result["stage1_packaging_version"].iloc[0],
            result["stage2_packaging_version"].iloc[0],
        }
        assert len(versions) == 3, f"Expected 3 distinct versions, got {versions}"

    def test_runtime_env_is_additive(self):
        """Stages with runtime_env can still import base-env packages (loguru is a Curator dep, not a Ray dep)."""
        assert self.results is not None
        result = self.results[0].to_pandas()
        assert bool(result["stage1_loguru_available"].iloc[0]), "loguru not importable in stage1 runtime_env"
        assert bool(result["stage2_loguru_available"].iloc[0]), "loguru not importable in stage2 runtime_env"
