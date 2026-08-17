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
from omegaconf import OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.text_filtering.whisper_hallucination import WhisperHallucinationStage
from nemo_curator.tasks import AudioTask

_TEXT_KEY = "pred_text"
_SKIP_KEY = "_skipme"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = _REPO_ROOT / "tutorials" / "audio" / "whisper_hallucination"


def _make_stage(tmp_path: Path, phrases: list[str], **kwargs) -> WhisperHallucinationStage:
    phrase_file = tmp_path / "phrases.txt"
    phrase_file.write_text("\n".join(phrases), encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(phrase_file), **kwargs)
    stage.setup()
    return stage


@pytest.mark.parametrize(
    ("text", "duration", "expected_reason"),
    [
        ("yes yes yes yes yes yes", 5.0, "repeated_ngrams"),
        (f"the {'a' * 30} here", 5.0, "long_word"),
        ("cat verylongwordindeed", 5.0, "long_word"),
        ("the quick brown fox jumps over the lazy dog", 0.1, "high_char_rate"),
    ],
)
def test_detector_flags_and_reports_reason(tmp_path: Path, text: str, duration: float, expected_reason: str) -> None:
    stage = _make_stage(tmp_path, [])
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: "", "duration": duration}))

    assert result.data[_SKIP_KEY] == "Hallucination:WhisperHallucination"
    assert expected_reason in result.data["additional_notes"]["WhisperHallucination"]


def test_clean_text_passes(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    result = stage.process(AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: ""}))

    assert result.data[_SKIP_KEY] == ""
    assert result.data["additional_notes"]["WhisperHallucination"] == "passed"


@pytest.mark.parametrize("text", ["Thank you", "Thank you.", "Thank-you", "Merci,", "Thank you for your time today."])
def test_phrase_matching_matches_reference_exact_and_prefix_behavior(tmp_path: Path, text: str) -> None:
    stage = _make_stage(tmp_path, ["Thank you", "Merci"])
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: ""}))

    assert result.data[_SKIP_KEY] == "Hallucination:WhisperHallucination"
    assert result.data["additional_notes"]["WhisperHallucination"] == "hallucination (phrase_match)"


@pytest.mark.parametrize("text", ["thank you", "THANK YOU"])
def test_phrase_matching_remains_case_sensitive(tmp_path: Path, text: str) -> None:
    stage = _make_stage(tmp_path, ["Thank you", "Merci"])
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: "", "duration": 5.0}))

    assert result.data[_SKIP_KEY] == ""
    assert result.data["additional_notes"]["WhisperHallucination"] == "passed"


@pytest.mark.parametrize(
    ("phrase", "text"),
    [
        ("Fifty-four", "Fifty-four"),
        ("Sì, è vero", "Sì, è vero"),
    ],
)
def test_punctuated_corpus_phrases_remain_reachable(tmp_path: Path, phrase: str, text: str) -> None:
    stage = _make_stage(tmp_path, [phrase])
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: ""}))

    assert result.data[_SKIP_KEY] == "Hallucination:WhisperHallucination"
    assert result.data["additional_notes"]["WhisperHallucination"] == "hallucination (phrase_match)"


def test_setup_normalizes_reference_phrases(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you!", "MERCI,", "Fifty-four"])
    assert stage._phrases == {"Thank you", "MERCI", "Fifty four"}


@pytest.mark.parametrize("text", ["", None])
def test_empty_or_non_string_text_is_skipped(tmp_path: Path, text: str | None) -> None:
    stage = _make_stage(tmp_path, [])
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: ""}))

    assert result.data[_SKIP_KEY] == ""
    assert result.data["additional_notes"]["WhisperHallucination"] == "skipped (empty text)"


def test_preserves_existing_skip_reason_when_overwrite_is_disabled(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    result = stage.process(AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Wrong language"}))

    assert result.data[_SKIP_KEY] == "Wrong language"
    assert result.data["additional_notes"]["WhisperHallucination"] == "skipped (flagged)"


@pytest.mark.parametrize(
    ("language", "word_length", "threshold", "expected_flagged"),
    [
        ("fi", 34, 35, False),
        ("fi", 36, 35, True),
        ("fi", 26, 25, True),
        ("en", 26, 25, True),
    ],
)
def test_absolute_long_word_threshold_is_configurable_for_all_languages(
    tmp_path: Path, language: str, word_length: int, threshold: int, expected_flagged: bool
) -> None:
    stage = _make_stage(tmp_path, [], long_word_threshold=threshold)
    text = f"the {'a' * word_length} here"
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: "", "language": language}))

    assert bool(result.data[_SKIP_KEY]) is expected_flagged


def test_agglutinative_language_skips_relative_long_word_check(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    result = stage.process(AudioTask(data={_TEXT_KEY: "a természetvédelmi cat", _SKIP_KEY: "", "language": "hu"}))
    assert result.data[_SKIP_KEY] == ""


def test_missing_duration_disables_character_rate_check(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    result = stage.process(AudioTask(data={_TEXT_KEY: "the quick brown fox jumps", _SKIP_KEY: ""}))
    assert result.data[_SKIP_KEY] == ""


def test_overwrite_recovers_owned_hallucination_flag(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True, recovery_value="RECOVERED_BY_QWEN")
    task = AudioTask(
        data={
            _TEXT_KEY: "the cat sat on the mat today",
            _SKIP_KEY: "Hallucination:First",
            "additional_notes": {"Earlier": "keep"},
        }
    )
    result = stage.process(task)

    assert result.data[_SKIP_KEY] == ""
    assert result.data["additional_notes"] == {
        "Earlier": "keep",
        "WhisperHallucination": "recovered_by_qwen",
    }


def test_overwrite_keeps_flagging_owned_hallucination(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    result = stage.process(AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Hallucination:First"}))

    assert result.data[_SKIP_KEY] == "Hallucination:WhisperHallucination"


@pytest.mark.parametrize("text", ["the cat sat on the mat today", "yes yes yes yes yes yes"])
def test_overwrite_never_replaces_a_foreign_flag(tmp_path: Path, text: str) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    result = stage.process(AudioTask(data={_TEXT_KEY: text, _SKIP_KEY: "Wrong language"}))

    assert result.data[_SKIP_KEY] == "Wrong language"


def test_requires_common_hall_file() -> None:
    with pytest.raises(TypeError, match="common_hall_file"):
        WhisperHallucinationStage()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="common_hall_file is required"):
        WhisperHallucinationStage(common_hall_file="")


def test_missing_required_text_key_raises(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    with pytest.raises(KeyError, match=_TEXT_KEY):
        stage.process(AudioTask(data={_SKIP_KEY: ""}))


def test_stage_declares_its_io_contract(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage.inputs() == ([], [_TEXT_KEY, _SKIP_KEY, "duration"])
    assert stage.outputs() == ([], [_SKIP_KEY, "additional_notes"])


def test_example_yaml_exposes_supported_filter_controls(tmp_path: Path) -> None:
    cfg = OmegaConf.load(_EXAMPLE_DIR / "pipeline.yaml")
    cfg.manifest_path = str(tmp_path / "asr_manifest.jsonl")
    cfg.output_path = str(tmp_path / "filtered_manifest.jsonl")

    pipeline = create_pipeline_from_yaml(cfg, log_config=False)

    assert isinstance(pipeline, Pipeline)
    stage = pipeline.stages[1]
    assert isinstance(stage, WhisperHallucinationStage)
    assert stage.common_hall_file == "tutorials/audio/whisper_hallucination/phrases.txt"
    assert stage.unique_words_threshold == 0.4
    assert stage.long_word_threshold == 25
    assert stage.long_word_rel_threshold == 3.0
    assert stage.max_char_rate == 40.0
    assert stage.overwrite is False
    assert stage.recovery_value == ""
