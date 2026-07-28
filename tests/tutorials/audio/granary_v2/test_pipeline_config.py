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

from pathlib import Path

import pytest
import yaml

from tutorials.audio.granary_v2.pipeline_config import (
    GranaryV2PipelineSettings,
    build_stage_configs,
)


def _settings(**overrides: object) -> GranaryV2PipelineSettings:
    values = {
        "input_config": "input.yaml",
        "output_dir": "output",
        "language": "en",
        "primary_model": "qwen_omni",
        "recovery_model": "qwen_asr",
        "regex_yaml": "common.yaml",
        "hallucination_phrases": "phrases.txt",
        "qwen_omni_model_id": "Qwen/Qwen3-Omni",
        "qwen_asr_model_id": "Qwen/Qwen3-ASR",
        "parakeet_v3_model_id": "nvidia/parakeet-v3",
        "parakeet_riva_model_id": "/models/parakeet-riva.nemo",
        "indic_model_id": "ai4bharat/indic_{language}.nemo",
    }
    values.update(overrides)
    return GranaryV2PipelineSettings(**values)


def _suffixes(stages: list[dict[str, object]]) -> list[str]:
    return [str(stage["_target_"]).rsplit(".", 1)[-1] for stage in stages]


def test_default_contract_is_complete_and_ordered() -> None:
    stages = build_stage_configs(_settings())

    assert _suffixes(stages) == [
        "NeMoSpeechAudioReader",
        "InitializeFieldsStage",
        "ASRStage",
        "WhisperHallucinationStage",
        "ASRStage",
        "WhisperHallucinationStage",
        "SelectBestPredictionStage",
        "RegexSubstitutionStage",
        "AbbreviationConcatStage",
        "ShardedManifestWriterStage",
    ]
    assert stages[2]["pred_text_key"] == "primary_model_prediction"
    assert stages[4]["pred_text_key"] == "fallback_model_prediction"
    assert stages[-1]["output_dir"] == "output"


def test_sed_is_an_atomic_optional_pair_before_asr() -> None:
    stages = build_stage_configs(_settings(sed_checkpoint="cnn14.pth"))

    assert _suffixes(stages)[2:5] == [
        "SEDInferenceStage",
        "SEDPostprocessingStage",
        "ASRStage",
    ]
    assert stages[2]["checkpoint_path"] == "cnn14.pth"


def test_no_recovery_route_omits_recovery_inference_and_recheck() -> None:
    stages = build_stage_configs(
        _settings(
            language="he",
            primary_model="whisper",
            recovery_model="none",
        )
    )

    assert _suffixes(stages).count("InferenceFasterWhisperStage") == 1
    assert _suffixes(stages).count("WhisperHallucinationStage") == 1
    selection = next(stage for stage in stages if _suffixes([stage])[0] == "SelectBestPredictionStage")
    assert selection["primary_text_key"] == "primary_model_prediction"


@pytest.mark.parametrize(
    ("model", "expected_suffix"),
    [
        ("whisper", "InferenceFasterWhisperStage"),
        ("indic_monolingual", "InferenceIndicConformerHybridStage"),
        ("parakeet_v3", "InferenceAsrNemoStage"),
        ("parakeet_riva", "InferenceAsrNemoStage"),
    ],
)
def test_non_qwen_models_use_dedicated_stage_adapters(
    model: str,
    expected_suffix: str,
) -> None:
    stages = build_stage_configs(
        _settings(
            primary_model=model,
            recovery_model="none",
            language="hi" if model in {"indic_monolingual", "parakeet_riva"} else "en",
        )
    )

    assert _suffixes(stages)[2] == expected_suffix


def test_local_nemo_checkpoint_is_wired_as_model_path() -> None:
    stages = build_stage_configs(
        _settings(
            language="hi",
            primary_model="parakeet_riva",
            recovery_model="none",
        )
    )

    assert stages[2]["model_path"] == "/models/parakeet-riva.nemo"
    assert "model_name" not in stages[2]


def test_turn_two_is_absent_from_code_and_yaml() -> None:
    tutorial_dir = Path(__file__).parents[4] / "tutorials" / "audio" / "granary_v2"
    serialized = repr(build_stage_configs(_settings())).lower()
    yaml_text = (tutorial_dir / "pipeline.yaml").read_text(encoding="utf-8").lower()

    assert "followup_prompt" not in serialized
    assert "secondary_text" not in serialized
    assert "disfluencywerguardstage" not in serialized
    assert "followup_prompt" not in yaml_text


def test_packaged_assets_are_valid() -> None:
    tutorial_dir = Path(__file__).parents[4] / "tutorials" / "audio" / "granary_v2"
    rules = yaml.safe_load((tutorial_dir / "common.yaml").read_text(encoding="utf-8"))
    phrases = (tutorial_dir / "phrases.txt").read_text(encoding="utf-8").splitlines()

    assert len(rules) >= 20
    assert all({"pattern", "repl"} <= set(rule) for rule in rules)
    assert len([phrase for phrase in phrases if phrase.strip()]) >= 50
