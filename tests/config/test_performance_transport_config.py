# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_curator.config.run import create_executor_from_yaml, create_pipeline_from_yaml


def test_yaml_applies_extended_metrics_and_executor_config() -> None:
    cfg = OmegaConf.create(
        {
            "backend": "ray_data",
            "executor_config": {"pipeline_hardware_sampler_enabled": True},
            "stages": [
                {
                    "_target_": "nemo_curator.stages.file_partitioning.FilePartitioningStage",
                    "file_paths": "./data",
                    "extended_performance_metrics": True,
                }
            ],
        }
    )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    executor = create_executor_from_yaml(cfg)

    assert pipeline.stages[0].extended_performance_metrics is True
    assert executor is not None
    assert executor.config["pipeline_hardware_sampler_enabled"] is True


def test_yaml_propagates_extended_metrics_through_manifest_reader_composite(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "stages": [
                {
                    "_target_": "nemo_curator.stages.audio.common.ManifestReader",
                    "manifest_path": str(tmp_path / "input.jsonl"),
                    "extended_performance_metrics": True,
                }
            ]
        }
    )

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    assert pipeline.stages[0].extended_performance_metrics is True

    pipeline.build()

    assert [stage.name for stage in pipeline.stages] == ["file_partitioning", "manifest_reader_stage"]
    assert all(stage.extended_performance_metrics for stage in pipeline.stages)


@pytest.mark.parametrize("nested_mode", ["bogus", "invalid"])
@patch("hydra.utils.get_class")
def test_xenna_rejects_invalid_nested_execution_mode(mock_get_class: MagicMock, nested_mode: str) -> None:
    cfg = OmegaConf.create({"backend": "xenna", "executor_config": {"execution_mode": nested_mode}})

    with pytest.raises(ValueError, match="Unknown Xenna execution mode"):
        create_executor_from_yaml(cfg)

    mock_get_class.assert_called_once()


@patch("hydra.utils.get_class")
def test_xenna_rejects_conflicting_execution_modes(mock_get_class: MagicMock) -> None:
    cfg = OmegaConf.create(
        {"backend": "xenna", "execution_mode": "streaming", "executor_config": {"execution_mode": "batch"}}
    )

    with pytest.raises(ValueError, match="Conflicting Xenna execution modes"):
        create_executor_from_yaml(cfg)

    mock_get_class.assert_called_once()


def test_qwen_tutorial_writes_complete_performance_report() -> None:
    config_path = Path(__file__).parents[2] / "tutorials/audio/qwen_omni_inprocess/pipeline.yaml"
    cfg = OmegaConf.load(config_path)

    assert cfg.performance_report_path == "./qwen_omni_performance.json"
    assert cfg.stages[2].extended_performance_metrics is True
    assert cfg.stages[3].performance_report_path == cfg.performance_report_path


def test_qwen_tutorial_install_includes_advertised_s3_filesystem_extra() -> None:
    repository_root = Path(__file__).parents[2]
    with (repository_root / "pyproject.toml").open("rb") as pyproject_file:
        optional_dependencies = tomllib.load(pyproject_file)["project"]["optional-dependencies"]
    readme = (repository_root / "tutorials/audio/qwen_omni_inprocess/README.md").read_text(encoding="utf-8")

    assert any(dependency.startswith("s3fs") for dependency in optional_dependencies["cloud_filesystems"])
    assert "--extra cloud_filesystems" in readme
