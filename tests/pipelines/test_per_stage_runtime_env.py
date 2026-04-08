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

"""Tests for per-stage runtime_env support.

Verifies that stages can declare different runtime_env (pip/uv packages) and
that Ray's native runtime_env creates isolated venvs per actor on each node.
Also verifies that runtime_env is additive: base-env packages remain importable.
"""

from typing import Any

import pandas as pd
import pytest

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class RecordPackagingVersionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Records the packaging library version visible to this worker.

    The column name is derived from self.name so multiple instances with
    different runtime_env can coexist in the same pipeline.
    Also checks whether loguru (a Curator dep, not a Ray dep) is importable.
    """

    name = "record_packaging_version"
    resources = Resources(cpus=0.5)
    batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import packaging

        try:
            from loguru import logger

            loguru_available = logger is not None
        except ImportError:
            loguru_available = False

        batch = task.to_pandas().copy()
        batch[f"{self.name}_version"] = packaging.__version__
        batch[f"{self.name}_loguru_available"] = loguru_available
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=batch,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _make_initial_task() -> DocumentBatch:
    return DocumentBatch(
        task_id="runtime_env_test",
        dataset_name="test",
        data=pd.DataFrame({"text": ["hello"]}),
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend_config",
    [
        pytest.param((RayDataExecutor, {}), id="ray_data"),
        pytest.param((XennaExecutor, {"execution_mode": "streaming"}), id="xenna_streaming"),
    ],
    indirect=True,
)
class TestPerStageRuntimeEnv:
    """Stages with different runtime_env see different package versions."""

    backend_cls: type[BaseExecutor] | None = None
    config: dict[str, Any] | None = None
    results: list[DocumentBatch] | None = None

    @pytest.fixture(scope="class", autouse=True)
    def backend_config(self, request: pytest.FixtureRequest, shared_ray_cluster: str):
        """Execute a 3-stage pipeline: base env, packaging==23.2 (pip), packaging==24.0 (uv)."""
        backend_cls, config = request.param
        request.cls.backend_cls = backend_cls
        request.cls.config = config

        base_stage = RecordPackagingVersionStage().with_(name="base_env")
        stage_v232 = RecordPackagingVersionStage().with_(
            name="pinned_v232",
            runtime_env={"pip": ["packaging==23.2"]},
        )
        stage_v240 = RecordPackagingVersionStage().with_(
            name="pinned_v240",
            runtime_env={"uv": ["packaging==24.0"]},
        )

        pipeline = Pipeline(name="runtime_env_test", stages=[base_stage, stage_v232, stage_v240])
        request.cls.results = pipeline.run(backend_cls(config), initial_tasks=[_make_initial_task()])

    def test_output_count(self):
        assert self.results is not None
        assert len(self.results) == 1

    def test_base_env_uses_installed_version(self):
        """Stage with no runtime_env should see the base environment's packaging version."""
        df = self.results[0].to_pandas()
        assert "base_env_version" in df.columns
        assert df["base_env_version"].iloc[0]  # non-empty

    def test_pinned_v232(self):
        df = self.results[0].to_pandas()
        assert df["pinned_v232_version"].iloc[0] == "23.2"

    def test_pinned_v240(self):
        df = self.results[0].to_pandas()
        assert df["pinned_v240_version"].iloc[0] == "24.0"

    def test_all_three_versions_differ(self):
        """Base env, 23.2, and 24.0 should all be distinct."""
        df = self.results[0].to_pandas()
        versions = {
            df["base_env_version"].iloc[0],
            df["pinned_v232_version"].iloc[0],
            df["pinned_v240_version"].iloc[0],
        }
        assert len(versions) == 3, f"Expected 3 distinct versions, got {versions}"

    def test_runtime_env_is_additive(self):
        """Stages with runtime_env can still import base-env packages (loguru is a Curator dep, not Ray)."""
        df = self.results[0].to_pandas()
        assert bool(df["pinned_v232_loguru_available"].iloc[0])
        assert bool(df["pinned_v240_loguru_available"].iloc[0])
