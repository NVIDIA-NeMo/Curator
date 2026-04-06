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

import pickle
from pathlib import Path

import pytest

from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter.audio_data_filter import (
    AudioDataFilterStage,
)
from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter.config import (
    _deep_merge,
    _validate,
    get_enabled_stages,
    load_config,
)
from nemo_curator.stages.audio.filtering import BandFilterStage, SIGMOSFilterStage, UTMOSFilterStage
from nemo_curator.stages.audio.preprocessing import MonoConversionStage, SegmentConcatenationStage
from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage, VADSegmentationStage

# ---------------------------------------------------------------------------
# 1. Config: _deep_merge()
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_deep_merge_overrides_scalar(self) -> None:
        base = {"a": 1, "b": 2}
        overrides = {"b": 99}
        result = _deep_merge(base, overrides)
        assert result["a"] == 1
        assert result["b"] == 99

    def test_deep_merge_nested_partial(self) -> None:
        base = {"vad": {"threshold": 0.5, "min_duration_sec": 2.0}}
        overrides = {"vad": {"threshold": 0.8}}
        result = _deep_merge(base, overrides)
        assert result["vad"]["threshold"] == 0.8
        assert result["vad"]["min_duration_sec"] == 2.0

    def test_deep_merge_does_not_mutate_base(self) -> None:
        base = {"vad": {"threshold": 0.5}}
        overrides = {"vad": {"threshold": 0.9}}
        _deep_merge(base, overrides)
        assert base["vad"]["threshold"] == 0.5


# ---------------------------------------------------------------------------
# 2. Config: load_config()
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_config_defaults(self) -> None:
        cfg = load_config(None)
        expected_keys = {
            "mono_conversion",
            "vad",
            "band_filter",
            "utmos",
            "sigmos",
            "concatenation",
            "speaker_separation",
            "timestamp_mapper",
        }
        assert expected_keys.issubset(set(cfg.keys()))

    def test_load_config_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_user_overrides(self, tmp_path: Path) -> None:
        user_yaml = tmp_path / "override.yaml"
        user_yaml.write_text("utmos:\n  mos_threshold: 4.0\n")

        cfg = load_config(str(user_yaml))
        assert cfg["utmos"]["mos_threshold"] == 4.0
        assert cfg["vad"]["threshold"] == 0.5

    def test_load_config_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        cfg = load_config(str(empty_yaml))
        assert cfg["vad"]["enable"] is True
        assert cfg["utmos"]["mos_threshold"] == 3.4


# ---------------------------------------------------------------------------
# 3. Config: _validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_vad_min_ge_max_raises(self) -> None:
        cfg = load_config(None)
        cfg["vad"]["min_duration_sec"] = 60.0
        cfg["vad"]["max_duration_sec"] = 10.0
        with pytest.raises(ValueError, match="min_duration_sec"):
            _validate(cfg)

    def test_validate_vad_threshold_out_of_range(self) -> None:
        cfg = load_config(None)
        cfg["vad"]["threshold"] = 1.5
        with pytest.raises(ValueError, match=r"vad\.threshold"):
            _validate(cfg)

    def test_validate_utmos_threshold_out_of_range(self) -> None:
        cfg = load_config(None)
        cfg["utmos"]["mos_threshold"] = 6.0
        with pytest.raises(ValueError, match=r"utmos\.mos_threshold"):
            _validate(cfg)

    def test_validate_sigmos_threshold_out_of_range(self) -> None:
        cfg = load_config(None)
        cfg["sigmos"]["noise_threshold"] = -1.0
        with pytest.raises(ValueError, match=r"sigmos\.noise_threshold"):
            _validate(cfg)

    def test_validate_speaker_min_duration_zero(self) -> None:
        cfg = load_config(None)
        cfg["speaker_separation"]["min_duration"] = 0
        with pytest.raises(ValueError, match=r"speaker_separation\.min_duration"):
            _validate(cfg)

    def test_validate_sample_rate_zero(self) -> None:
        cfg = load_config(None)
        cfg["mono_conversion"]["output_sample_rate"] = 0
        with pytest.raises(ValueError, match="output_sample_rate"):
            _validate(cfg)

    def test_validate_silence_negative(self) -> None:
        cfg = load_config(None)
        cfg["concatenation"]["silence_duration_sec"] = -1
        with pytest.raises(ValueError, match="silence_duration_sec"):
            _validate(cfg)


# ---------------------------------------------------------------------------
# 4. Config: get_enabled_stages()
# ---------------------------------------------------------------------------


class TestGetEnabledStages:
    def test_get_enabled_stages_all_enabled(self) -> None:
        cfg = load_config(None)
        stages = get_enabled_stages(cfg)
        assert "mono_conversion" in stages
        assert "vad" in stages
        assert "band_filter" in stages
        assert "utmos" in stages
        assert "sigmos" in stages
        assert "concatenation" in stages
        assert "speaker_separation" in stages
        assert "timestamp_mapper" in stages

    def test_get_enabled_stages_only_vad(self) -> None:
        cfg = load_config(None)
        cfg["band_filter"]["enable"] = False
        cfg["utmos"]["enable"] = False
        cfg["sigmos"]["enable"] = False
        cfg["speaker_separation"]["enable"] = False
        stages = get_enabled_stages(cfg)
        assert stages == ["mono_conversion", "vad", "timestamp_mapper"]

    def test_get_enabled_stages_no_speaker_no_concatenation(self) -> None:
        cfg = load_config(None)
        cfg["speaker_separation"]["enable"] = False
        stages = get_enabled_stages(cfg)
        assert "concatenation" not in stages
        assert "speaker_separation" not in stages


# ---------------------------------------------------------------------------
# 5. Decompose: Stage Count and Types
# ---------------------------------------------------------------------------


class TestDecomposeStageCount:
    def test_decompose_all_enabled_stage_count(self) -> None:
        stage = AudioDataFilterStage()
        stages = stage.decompose()
        assert len(stages) == 12

    def test_decompose_all_disabled_except_mono(self) -> None:
        stage = AudioDataFilterStage(config={
            "vad": {"enable": False},
            "band_filter": {"enable": False},
            "utmos": {"enable": False},
            "sigmos": {"enable": False},
            "speaker_separation": {"enable": False},
        })
        stages = stage.decompose()
        assert len(stages) == 1
        assert isinstance(stages[0], MonoConversionStage)

    def test_decompose_no_speaker_no_second_pass(self) -> None:
        stage = AudioDataFilterStage(config={
            "speaker_separation": {"enable": False},
        })
        stages = stage.decompose()
        assert len(stages) == 6
        stage_types = [type(s) for s in stages]
        assert stage_types.count(VADSegmentationStage) == 1
        assert stage_types.count(BandFilterStage) == 1
        assert stage_types.count(UTMOSFilterStage) == 1
        assert stage_types.count(SIGMOSFilterStage) == 1
        assert SegmentConcatenationStage not in stage_types
        assert SpeakerSeparationStage not in stage_types


# ---------------------------------------------------------------------------
# 6. Decompose: Stage Configuration
# ---------------------------------------------------------------------------


class TestDecomposeConfig:
    def test_decompose_vad_nested_flag(self) -> None:
        stage = AudioDataFilterStage()
        stages = stage.decompose()
        vad_stages = [s for s in stages if isinstance(s, VADSegmentationStage)]
        assert len(vad_stages) == 2
        assert vad_stages[0].nested is True
        assert vad_stages[1].nested is False

    def test_decompose_resources_from_config(self) -> None:
        stage = AudioDataFilterStage(config={"vad": {"gpus": 0.5}})
        stages = stage.decompose()
        vad_stages = [s for s in stages if isinstance(s, VADSegmentationStage)]
        assert vad_stages[0]._resources.gpus == 0.5

    def test_decompose_custom_thresholds(self) -> None:
        stage = AudioDataFilterStage(config={"utmos": {"mos_threshold": 4.2}})
        stages = stage.decompose()
        utmos_stages = [s for s in stages if isinstance(s, UTMOSFilterStage)]
        assert len(utmos_stages) == 2
        for s in utmos_stages:
            assert s.mos_threshold == 4.2


# ---------------------------------------------------------------------------
# 7. Decompose: Edge Cases
# ---------------------------------------------------------------------------


class TestDecomposeEdgeCases:
    def test_decompose_speaker_without_vad_no_concat(self) -> None:
        stage = AudioDataFilterStage(config={
            "vad": {"enable": False},
            "speaker_separation": {"enable": True},
        })
        stages = stage.decompose()
        stage_types = [type(s) for s in stages]
        assert SegmentConcatenationStage not in stage_types
        assert SpeakerSeparationStage in stage_types

    def test_decompose_config_path_and_dict_override(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("utmos:\n  mos_threshold: 4.0\n")

        stage = AudioDataFilterStage(
            config_path=str(yaml_path),
            config={"utmos": {"mos_threshold": 4.5}},
        )
        stages = stage.decompose()
        utmos_stages = [s for s in stages if isinstance(s, UTMOSFilterStage)]
        assert utmos_stages[0].mos_threshold == 4.5


# ---------------------------------------------------------------------------
# 8. Default YAML Consistency
# ---------------------------------------------------------------------------


class TestPickling:
    def test_audio_data_filter_stage_pickling(self) -> None:
        stage = AudioDataFilterStage(config={"utmos": {"mos_threshold": 4.0}})
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored.name == "AudioDataFilter"
        assert restored._cfg["utmos"]["mos_threshold"] == 4.0


class TestDefaultYAMLConsistency:
    def test_default_yaml_all_stages_have_resources(self) -> None:
        cfg = load_config(None)
        stages_with_cpus = [
            "mono_conversion", "vad", "band_filter", "utmos",
            "sigmos", "concatenation", "speaker_separation", "timestamp_mapper",
        ]
        for stage_name in stages_with_cpus:
            assert "cpus" in cfg[stage_name], f"{stage_name} missing 'cpus' in config"

        stages_with_gpus = ["vad", "band_filter", "utmos", "sigmos", "speaker_separation"]
        for stage_name in stages_with_gpus:
            assert "gpus" in cfg[stage_name], f"{stage_name} missing 'gpus' in config"

