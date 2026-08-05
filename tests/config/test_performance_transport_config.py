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

from omegaconf import OmegaConf


def test_qwen_tutorial_writes_complete_performance_report() -> None:
    config_path = Path(__file__).parents[2] / "tutorials/audio/qwen_omni_inprocess/pipeline.yaml"
    cfg = OmegaConf.load(config_path)

    assert cfg.performance_report_path == "./qwen_omni_performance.json"
    assert cfg.stages[3].performance_report_path == cfg.performance_report_path


def test_qwen_tutorial_install_includes_advertised_s3_filesystem_extra() -> None:
    repository_root = Path(__file__).parents[2]
    with (repository_root / "pyproject.toml").open("rb") as pyproject_file:
        optional_dependencies = tomllib.load(pyproject_file)["project"]["optional-dependencies"]
    readme = (repository_root / "tutorials/audio/qwen_omni_inprocess/README.md").read_text(encoding="utf-8")

    assert any(dependency.startswith("s3fs") for dependency in optional_dependencies["cloud_filesystems"])
    assert "--extra cloud_filesystems" in readme
