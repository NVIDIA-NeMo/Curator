# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import hashlib
import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.text_filtering.whisper_hallucination import WhisperHallucinationStage
from nemo_curator.tasks import AudioTask


def _make_stage(tmp_path: Path, phrases: list[str], **kwargs) -> WhisperHallucinationStage:
    p = tmp_path / "phrases.txt"
    p.write_text("\n".join(phrases), encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(p), **kwargs)
    stage.setup()
    return stage


_TEXT_KEY = "pred_text"
_SKIP_KEY = "_skipme"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = _REPO_ROOT / "tutorials" / "audio" / "whisper_hallucination"


def test_clean_text_passes(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["lorem ipsum"])
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_repeated_ngrams_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_long_word_absolute_threshold_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    long_word = "a" * 30
    task = AudioTask(data={_TEXT_KEY: f"the {long_word} here", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_long_word_relative_threshold_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "cat verylongwordindeed", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_frequent_phrase_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={_TEXT_KEY: "Thank you", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_frequent_phrase_strips_punctuation(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={_TEXT_KEY: "Thank you.", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_frequent_phrase_strips_trailing_comma(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={_TEXT_KEY: "Thank you,", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_setup_is_called_lazily_from_process(tmp_path: Path) -> None:
    phrase_file = tmp_path / "phrases.txt"
    phrase_file.write_text("Thank you\n", encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(phrase_file))

    result = stage.process(AudioTask(data={_TEXT_KEY: "Thank you", _SKIP_KEY: "", "duration": 1.0}))

    assert stage._setup_called is True
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_non_string_text_returns_task_unchanged(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: None, _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_preserves_existing_skip_me_reason(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Wrong language"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Wrong language"


def test_empty_words_not_flagged_by_ngram(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage._repeated_ngrams([]) is False


def test_empty_words_not_flagged_by_long_word(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage._long_word([]) is False


def test_phrases_file_loads_lines_as_is(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you", "Amen", "Yeah"])
    assert "Thank you" in stage._phrases
    assert "Amen" in stage._phrases
    assert "Yeah" in stage._phrases


def test_requires_common_hall_file() -> None:
    with pytest.raises(ValueError, match="common_hall_file is required"):
        WhisperHallucinationStage(common_hall_file="")


def test_agglutinative_lang_respects_configured_threshold(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], long_word_threshold=35)
    word_34 = "a" * 34
    task = AudioTask(data={_TEXT_KEY: f"the {word_34} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_agglutinative_lang_flags_above_threshold(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], long_word_threshold=35)
    word_36 = "a" * 36
    task = AudioTask(data={_TEXT_KEY: f"the {word_36} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_agglutinative_lang_skips_relative_check(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "a természetvédelmi cat", _SKIP_KEY: "", "language": "hu"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_non_agglutinative_lang_uses_default_threshold(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    word_26 = "a" * 26
    task = AudioTask(data={_TEXT_KEY: f"the {word_26} here", _SKIP_KEY: "", "language": "en"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_agglutinative_lang_uses_default_absolute_threshold(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    word_26 = "a" * 26
    task = AudioTask(data={_TEXT_KEY: f"the {word_26} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_high_char_rate_flags_dense_text_over_a_short_clip(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(
        data={
            _TEXT_KEY: "the quick brown fox jumps over the lazy dog",
            _SKIP_KEY: "",
            "duration": 0.1,
        }
    )
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]
    assert "high_char_rate" in result.data["additional_notes"]["WhisperHallucination"]


def test_plausible_char_rate_is_not_flagged(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(
        data={
            _TEXT_KEY: "the quick brown fox jumps over the lazy dog",
            _SKIP_KEY: "",
            "duration": 4.0,
        }
    )
    assert stage.process(task).data[_SKIP_KEY] == ""


def test_missing_duration_disables_the_char_rate_check(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "the quick brown fox jumps", _SKIP_KEY: ""})
    assert stage.process(task).data[_SKIP_KEY] == ""


def test_overwrite_clears_a_stale_hallucination_flag_when_text_is_now_clean(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: "Hallucination:WhisperHallucination"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""
    assert result.data["additional_notes"]["WhisperHallucination"] == "recovered"


def test_overwrite_uses_a_custom_recovery_note(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True, recovery_value="RECOVERED_BY_QWEN")
    task = AudioTask(data={_TEXT_KEY: "a perfectly normal sentence", _SKIP_KEY: "Hallucination:First"})
    result = stage.process(task)
    assert result.data["additional_notes"]["WhisperHallucination"] == "recovered_by_qwen"


def test_overwrite_keeps_flagging_text_that_is_still_bad(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Hallucination:First"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Hallucination:WhisperHallucination"


def test_overwrite_does_not_clear_a_flag_owned_by_another_filter(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: "Wrong language"})
    assert stage.process(task).data[_SKIP_KEY] == "Wrong language"


def test_flagging_does_not_clobber_a_flag_owned_by_another_filter(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Wrong language"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Wrong language"
    assert "hallucination" in result.data["additional_notes"]["WhisperHallucination"]


def test_notes_are_keyed_per_stage_instance(tmp_path: Path) -> None:
    first = _make_stage(tmp_path, [], name="First")
    second = _make_stage(tmp_path, [], name="Second", overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: ""})

    second.process(first.process(task))

    notes = task.data["additional_notes"]
    assert notes["First"] == "passed"
    assert notes["Second"] == "passed"


def test_passing_text_records_a_passed_note(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: ""})
    assert stage.process(task).data["additional_notes"]["WhisperHallucination"] == "passed"


def test_already_flagged_row_is_noted_as_skipped(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Wrong language"})
    assert stage.process(task).data["additional_notes"]["WhisperHallucination"] == "skipped (flagged)"


def test_reasons_are_reported_for_every_check_that_fired(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["yes"])
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "", "duration": 0.1})
    note = stage.process(task).data["additional_notes"]["WhisperHallucination"]
    assert "repeated_ngrams" in note
    assert "high_char_rate" in note


def test_missing_required_text_key_raises(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_SKIP_KEY: ""})
    with pytest.raises(KeyError, match=_TEXT_KEY):
        stage.process(task)


def test_prefix_match_length_is_a_constant_not_a_constructor_argument() -> None:
    parameters = inspect.signature(WhisperHallucinationStage).parameters
    assert "_PREFIX_MATCH_MIN_LEN" not in parameters
    assert "agglutinative_long_word_threshold" not in parameters


def test_stage_declares_its_io_contract(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    required_attrs, required_columns = stage.inputs()
    _required_out, optional_out = stage.outputs()
    assert required_attrs == []
    assert required_columns == [_TEXT_KEY, _SKIP_KEY, "duration"]
    assert {_SKIP_KEY, "additional_notes"} <= set(optional_out)


def test_stage_is_cpu_only(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage.resources.gpus == 0
    assert stage.resources.cpus == 1.0


def test_stage_is_exported_from_the_audio_package() -> None:
    import nemo_curator.stages.audio as audio_pkg

    assert audio_pkg.WhisperHallucinationStage is WhisperHallucinationStage


def test_stage_is_exported_from_text_filtering_package() -> None:
    from nemo_curator.stages.audio import text_filtering

    assert text_filtering.__all__ == ["WhisperHallucinationStage"]
    assert text_filtering.WhisperHallucinationStage is WhisperHallucinationStage


def test_example_yaml_exposes_all_supported_filter_controls(tmp_path: Path) -> None:
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


def test_bundled_phrase_file_matches_reference_hash() -> None:
    phrase_file = _EXAMPLE_DIR / "phrases.txt"
    assert hashlib.sha256(phrase_file.read_bytes()).hexdigest() == (
        "34ba2fcd7756f193e80ba4ac34a6b5db0dab92adeb0750beb796b1bf57f6bc42"
    )
