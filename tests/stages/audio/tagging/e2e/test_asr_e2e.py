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

"""End-to-end test for the ASR audio tagging pipeline.

Runs the full pipeline:
  ManifestReader -> Resample -> Diarize -> SplitASRAlignJoin (composite) ->
  Merge -> ITN -> Bandwidth -> SQUIM -> PrepareModuleSegments -> Write

Uses create_pipeline_from_yaml + pipeline.run(executor) as shown in
tutorials/audio/tagging/main.py.

ASR pipeline differs from TTS in PrepareModuleSegmentsStage:
  - module=asr, full_utterance_ratio=0.8 (allows partial utterances)
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml

from .conftest import CONFIGS_DIR
from .utils import load_manifest


@pytest.mark.gpu
@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN required for PyAnnote models")
def test_asr_e2e(tmp_path: Path, get_input_manifest: str) -> None:
    """ASR tagging pipeline e2e: Resample + Diarize + Split + ASR Align + Join + Merge + ITN + BW + SQUIM + Segments."""
    config_path = CONFIGS_DIR / "asr_pipeline.yaml"
    # reference_manifest = str(REFERENCE_DIR / "asr" / "test_data_reference.jsonl")

    cfg = OmegaConf.load(config_path)

    cfg.input_manifest = get_input_manifest
    cfg.final_manifest = str(tmp_path / "asr_output.jsonl")
    cfg.workspace_dir = str(tmp_path)
    cfg.resampled_audio_dir = str(tmp_path / "audio_resampled")
    cfg.hf_token = os.getenv("HF_TOKEN", "")
    cfg.language_short = "en"

    cfg.stages[4].model_name = "nvidia/stt_en_fastconformer_ctc_large"
    cfg.stages[4].is_fastconformer = True
    cfg.stages[4].decoder_type = "ctc"

    pipeline = create_pipeline_from_yaml(cfg)
    executor = XennaExecutor(config={"execution_mode": "batch"})
    pipeline.run(executor)

    assert Path(cfg.final_manifest).exists(), "Output manifest not created"
    output_data = load_manifest(cfg.final_manifest)
    assert len(output_data) == 2, f"Expected 2 entries, got {len(output_data)}"

    required_keys = {"audio_filepath", "audio_item_id", "segments", "alignment"}
    for entry in output_data:
        missing = required_keys - set(entry.keys())
        assert not missing, f"Missing keys {missing} for {entry.get('audio_item_id')}"

        assert len(entry["segments"]) > 0, f"No segments for {entry['audio_item_id']}"
        for seg in entry["segments"]:
            assert seg["end"] > seg["start"] >= 0, (
                f"Bad segment bounds [{seg['start']}, {seg['end']}] in {entry['audio_item_id']}"
            )
            assert "text" in seg, f"Segment missing 'text' in {entry['audio_item_id']}"
            if "metrics" in seg:
                assert isinstance(seg["metrics"], dict), f"metrics not a dict in {entry['audio_item_id']}"

        assert len(entry["alignment"]) > 0, f"No alignment for {entry['audio_item_id']}"
        for word in entry["alignment"]:
            assert "word" in word
            assert "start" in word
            assert "end" in word
