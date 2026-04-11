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

"""End-to-end test for the TTS audio tagging pipeline.

Runs the full pipeline:
  ManifestReader -> Resample -> Diarize -> SplitASRAlignJoin (composite) ->
  Merge -> ITN -> Bandwidth -> SQUIM -> PrepareModuleSegments -> Write

Uses create_pipeline_from_yaml + pipeline.run(executor) as shown in
tutorials/audio/tagging/main.py.
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml

from .conftest import CONFIGS_DIR, REFERENCE_DIR
from .utils import check_output


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN required for PyAnnote models")
def test_tts_e2e(tmp_path: Path, get_input_manifest: str) -> None:
    """TTS tagging pipeline e2e: Resample + Diarize + Split + ASR Align + Join + Merge + ITN + BW + SQUIM + Segments."""
    config_path = CONFIGS_DIR / "tts_pipeline.yaml"
    reference_manifest = str(REFERENCE_DIR / "tts" / "test_data_reference.jsonl")

    cfg = OmegaConf.load(config_path)

    cfg.input_manifest = get_input_manifest
    cfg.final_manifest = str(tmp_path / "tts_output.jsonl")
    cfg.workspace_dir = str(tmp_path)
    cfg.resampled_audio_dir = str(tmp_path / "audio_resampled")
    cfg.hf_token = os.getenv("HF_TOKEN", "")
    cfg.language_short = "en"

    # Override NeMoASRAlignerStage (index 4) to use CTC model for CPU testing
    cfg.stages[4].model_name = "nvidia/stt_en_fastconformer_ctc_large"
    cfg.stages[4].is_fastconformer = True
    cfg.stages[4].decoder_type = "ctc"
    cfg.stages[4].transcribe_batch_size = 1

    pipeline = create_pipeline_from_yaml(cfg)
    executor = XennaExecutor(
        {
            "cpu_allocation_percentage": 0.9,
        }
    )
    pipeline.run(executor)

    check_output(cfg.final_manifest, reference_manifest, text_key="text")
