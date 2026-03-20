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

Uses pip_specs + resolve_stage_pip_envs (uv CLI) so each stage runs with its own
packaging version. Requires `uv` on PATH; test is skipped if uv is not available.
See tutorials/per_stage_runtime_env_example.py and docs/design/per-stage-runtime-environment.md.
"""

import shutil  # noqa: I001
import subprocess

import pandas as pd
import pytest

from typing import ClassVar

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


def _uv_available() -> bool:
    uv_exe = shutil.which("uv")
    if not uv_exe:
        return False
    try:
        subprocess.run([uv_exe, "--version"], check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError:
        return False
    else:
        return True


@pytest.fixture
def require_uv():
    if not _uv_available():
        pytest.skip("uv not on PATH; per-stage pip_specs tests require uv")


class VersionStage1(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage 1: packaging==23.2 via pip_specs (resolver creates venv, PYTHONPATH)."""

    name = "version_stage_1"
    resources = Resources(cpus=0.5)
    batch_size = 1
    pip_specs: ClassVar[list[str]] = ["packaging==23.2"]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_packaging_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import packaging

        df = task.to_pandas().copy()
        df["stage1_packaging_version"] = packaging.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


class VersionStage2(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage 2: packaging==24.0 via pip_specs (resolver creates venv, PYTHONPATH)."""

    name = "version_stage_2"
    resources = Resources(cpus=0.5)
    batch_size = 1
    pip_specs: ClassVar[list[str]] = ["packaging==24.0"]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_packaging_version"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage2_packaging_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import packaging

        df = task.to_pandas().copy()
        df["stage2_packaging_version"] = packaging.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@pytest.mark.usefixtures("shared_ray_client", "require_uv")
def test_per_stage_different_package_versions_ray_data() -> None:
    """Run two stages with different packaging versions via pip_specs; assert each sees its own version.

    Pipeline.run() calls resolve_stage_pip_envs() to create venvs with uv; Ray Data adapter
    injects PYTHONPATH so workers load the correct site-packages per stage.
    """
    initial = DocumentBatch(
        task_id="per_stage_version_test",
        dataset_name="test",
        data=pd.DataFrame({"text": ["hello"]}),
    )
    pipeline = Pipeline(
        name="per_stage_version_test",
        stages=[VersionStage1(), VersionStage2()],
    )
    results = pipeline.run(
        executor=RayDataExecutor(),
        initial_tasks=[initial],
    )
    assert results is not None
    assert len(results) == 1
    out = results[0]
    df = out.to_pandas()
    assert "stage1_packaging_version" in df.columns
    assert "stage2_packaging_version" in df.columns
    assert df["stage1_packaging_version"].iloc[0] == "23.2", "Stage 1 should see packaging 23.2"
    assert df["stage2_packaging_version"].iloc[0] == "24.0", "Stage 2 should see packaging 24.0"


@pytest.mark.usefixtures("shared_ray_client", "require_uv")
def test_per_stage_different_package_versions_xenna() -> None:
    """Run two stages with different packaging versions via pip_specs using XennaExecutor.

    Pipeline.run() calls resolve_stage_pip_envs() to create venvs with uv; Xenna adapter
    uses env_info() to set PYTHONPATH so workers load the correct site-packages per stage.
    """
    initial = DocumentBatch(
        task_id="per_stage_version_test_xenna",
        dataset_name="test",
        data=pd.DataFrame({"text": ["hello"]}),
    )
    pipeline = Pipeline(
        name="per_stage_version_test_xenna",
        stages=[VersionStage1(), VersionStage2()],
    )
    results = pipeline.run(
        executor=XennaExecutor(config={"execution_mode": "streaming"}),
        initial_tasks=[initial],
    )
    assert results is not None
    assert len(results) == 1
    out = results[0]
    df = out.to_pandas()
    assert "stage1_packaging_version" in df.columns
    assert "stage2_packaging_version" in df.columns
    assert df["stage1_packaging_version"].iloc[0] == "23.2", "Stage 1 should see packaging 23.2"
    assert df["stage2_packaging_version"].iloc[0] == "24.0", "Stage 2 should see packaging 24.0"
