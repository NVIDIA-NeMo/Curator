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

"""Construction-level compatibility checks for the runnable audio tutorials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

TUTORIAL_PIPELINES = (
    (
        "fleurs",
        "tutorials/audio/fleurs/pipeline.yaml",
        "stages",
        (
            "CreateInitialManifestFleursStage",
            "ASRStage",
            "GetPairwiseWerStage",
            "GetAudioDurationStage",
            "PreserveByValueStage",
            "AudioToDocumentStage",
            "JsonlWriter",
        ),
    ),
    (
        "alm",
        "tutorials/audio/alm/pipeline.yaml",
        "stages",
        (
            "ManifestReader",
            "ALMDataBuilderStage",
            "ALMDataOverlapStage",
            "ManifestWriterStage",
        ),
    ),
    (
        "readspeech",
        "tutorials/audio/readspeech/pipeline.yaml",
        "processors",
        (
            "CreateInitialManifestReadSpeechStage",
            "AudioDataFilterStage",
            "AudioToDocumentStage",
            "JsonlWriter",
        ),
    ),
    (
        "tagging_asr",
        "tutorials/audio/tagging/asr_pipeline.yaml",
        "stages",
        (
            "ManifestReader",
            "ResampleAudioStage",
            "PyAnnoteDiarizationStage",
            "SplitLongAudioStage",
            "NeMoASRAlignerStage",
            "JoinSplitAudioMetadataStage",
            "MergeAlignmentDiarizationStage",
            "BandwidthEstimationStage",
            "TorchSquimQualityMetricsStage",
            "PrepareModuleSegmentsStage",
            "NeMoASRAlignerStage",
            "ComputeWERStage",
            "ManifestWriterStage",
        ),
    ),
    (
        "tagging_tts",
        "tutorials/audio/tagging/tts_pipeline.yaml",
        "stages",
        (
            "ManifestReader",
            "ResampleAudioStage",
            "PyAnnoteDiarizationStage",
            "SplitLongAudioStage",
            "NeMoASRAlignerStage",
            "JoinSplitAudioMetadataStage",
            "MergeAlignmentDiarizationStage",
            "BandwidthEstimationStage",
            "TorchSquimQualityMetricsStage",
            "PrepareModuleSegmentsStage",
            "ManifestWriterStage",
        ),
    ),
)


def _load_config(relative_path: str, tmp_path: Path) -> DictConfig:
    cfg = OmegaConf.load(REPO_ROOT / relative_path)
    overrides: dict[str, Any] = {
        "raw_data_dir": str(tmp_path / "raw"),
        "manifest_path": str(REPO_ROOT / "tests/fixtures/audio/alm/sample_input.jsonl"),
        "input_manifest": str(REPO_ROOT / "tests/fixtures/audio/tagging/sample_input.jsonl"),
        "final_manifest": str(tmp_path / "final.jsonl"),
        "workspace_dir": str(tmp_path / "tagging"),
    }
    declared_keys = set(cfg.keys())
    for key, value in overrides.items():
        if key in declared_keys:
            cfg[key] = value
    return cfg


def _construct_stages(cfg: DictConfig, collection: str) -> list[Any]:
    if collection == "stages":
        return list(create_pipeline_from_yaml(cfg, log_config=False).stages)
    return [hydra.utils.instantiate(stage_cfg) for stage_cfg in cfg.processors]


@pytest.mark.parametrize(
    ("_name", "relative_path", "collection", "expected_classes"),
    TUTORIAL_PIPELINES,
)
def test_tutorial_yaml_stage_order_remains_compatible(
    _name: str,
    relative_path: str,
    collection: str,
    expected_classes: tuple[str, ...],
    tmp_path: Path,
) -> None:
    stages = _construct_stages(_load_config(relative_path, tmp_path), collection)
    assert tuple(type(stage).__name__ for stage in stages) == expected_classes


def test_tutorial_sensitive_defaults_remain_compatible(tmp_path: Path) -> None:
    fleurs = _construct_stages(
        _load_config("tutorials/audio/fleurs/pipeline.yaml", tmp_path),
        "stages",
    )
    assert fleurs[0].lang == "hy_am"
    assert fleurs[0].split == "dev"
    assert fleurs[0].batch_size == 4
    assert fleurs[1].adapter_target == "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter"
    assert fleurs[1].model_id == "nvidia/stt_hy_fastconformer_hybrid_large_pc"
    assert fleurs[1].audio_filepath_key == "audio_filepath"
    assert fleurs[1].resources.gpus == 1.0
    assert fleurs[2].text_key == "text"
    assert fleurs[2].pred_text_key == "pred_text"
    assert fleurs[2].wer_key == "wer_pct"
    assert fleurs[4].operator.__name__ == "le"
    assert fleurs[4].target_value == 5.5
    assert fleurs[5].batch_size == 64

    tagging = _construct_stages(
        _load_config("tutorials/audio/tagging/asr_pipeline.yaml", tmp_path),
        "stages",
    )
    resample = tagging[1]
    assert resample.target_sample_rate == 16000
    assert resample.target_nchannels == 1
    assert resample.write_to_disk is True
    assert resample.keep_waveform_in_task is False
    assert resample.update_audio_filepath is False
    assert resample.audio_filepath_key == "audio_filepath"
    assert resample.resampled_audio_filepath_key == "resampled_audio_filepath"
    assert tagging[2].audio_filepath_key == "resampled_audio_filepath"
    assert tagging[8].audio_filepath_key == "resampled_audio_filepath"
