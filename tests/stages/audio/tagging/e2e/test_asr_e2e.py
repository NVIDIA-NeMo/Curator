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

"""End-to-end test for the ASR-specific audio tagging pipeline stages.

Starts from base_manifest.jsonl (pre-computed upstream output) and runs:
  ManifestReader -> PrepareModuleSegments(asr) -> ASRAlignment2 -> ComputeWER -> Write

This isolates the ASR-specific stages from non-deterministic upstream
stages (diarization, 1st-pass ASR alignment) whose output varies across
GPU hardware.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml

from .conftest import AUDIO_FIXTURES_DIR, CONFIGS_DIR, REFERENCE_DIR
from .utils import check_output


@pytest.mark.gpu
def test_asr_e2e(tmp_path: Path) -> None:
    """ASR pipeline e2e: PrepareModuleSegments(asr) + 2nd-pass ASR + WER."""
    config_path = CONFIGS_DIR / "asr_pipeline.yaml"
    input_manifest = str(AUDIO_FIXTURES_DIR / "base_manifest.jsonl")
    reference_manifest = str(REFERENCE_DIR / "asr" / "test_data_reference.jsonl")

    cfg = OmegaConf.load(config_path)

    cfg.input_manifest = input_manifest
    cfg.final_manifest = str(tmp_path / "asr_output.jsonl")
    cfg.workspace_dir = str(tmp_path)
    cfg.language_short = "en"

    pipeline = create_pipeline_from_yaml(cfg)
    executor = XennaExecutor(config={"execution_mode": "batch"})
    pipeline.run(executor)

    check_output(cfg.final_manifest, reference_manifest, text_key="text")
