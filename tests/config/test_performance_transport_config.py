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

from pathlib import Path

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


def test_qwen_tutorial_writes_complete_performance_report() -> None:
    config_path = Path(__file__).parents[2] / "tutorials/audio/qwen_omni_inprocess/pipeline.yaml"
    cfg = OmegaConf.load(config_path)

    assert cfg.performance_report_path == "./qwen_omni_performance.json"
    assert cfg.stages[2].extended_performance_metrics is True
    assert cfg.stages[3].performance_report_path == cfg.performance_report_path
