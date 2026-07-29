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

import inspect
from pathlib import Path

import pytest

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


def test_agglutinative_lang_uses_higher_threshold(tmp_path: Path) -> None:
    """A 34-char word in Finnish should NOT be flagged (threshold=35 for agglutinative)."""
    stage = _make_stage(tmp_path, [])
    word_34 = "a" * 34
    task = AudioTask(data={_TEXT_KEY: f"the {word_34} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_agglutinative_lang_flags_above_threshold(tmp_path: Path) -> None:
    """A 36-char word in Finnish SHOULD be flagged (above threshold=35)."""
    stage = _make_stage(tmp_path, [])
    word_36 = "a" * 36
    task = AudioTask(data={_TEXT_KEY: f"the {word_36} here", _SKIP_KEY: "", "language": "fi"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


def test_agglutinative_lang_skips_relative_check(tmp_path: Path) -> None:
    """A long word in Hungarian shouldn't trigger relative threshold (skip_relative=True)."""
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "a természetvédelmi cat", _SKIP_KEY: "", "language": "hu"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_non_agglutinative_lang_uses_default_threshold(tmp_path: Path) -> None:
    """A 26-char word in English SHOULD be flagged (default threshold=25)."""
    stage = _make_stage(tmp_path, [])
    word_26 = "a" * 26
    task = AudioTask(data={_TEXT_KEY: f"the {word_26} here", _SKIP_KEY: "", "language": "en"})
    result = stage.process(task)
    assert "Hallucination" in result.data[_SKIP_KEY]


# ----------------------------------------------------------------------
# High character rate
# ----------------------------------------------------------------------


def test_high_char_rate_flags_dense_text_over_a_short_clip(tmp_path: Path) -> None:
    """A full sentence confabulated over 0.1 s is an impossible speaking rate."""
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


# ----------------------------------------------------------------------
# Recovery re-check (overwrite=True)
# ----------------------------------------------------------------------


def test_overwrite_clears_a_stale_hallucination_flag_when_text_is_now_clean(tmp_path: Path) -> None:
    """A recovery pass promotes a clean second-pass transcript."""
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
    """Only hallucination flags are ours to clear; someone else's reason stands."""
    stage = _make_stage(tmp_path, [], overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: "Wrong language"})
    assert stage.process(task).data[_SKIP_KEY] == "Wrong language"


def test_flagging_does_not_clobber_a_flag_owned_by_another_filter(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [], overwrite=True)
    task = AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: "Wrong language"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Wrong language"
    assert "hallucination" in result.data["additional_notes"]["WhisperHallucination"]


# ----------------------------------------------------------------------
# Notes, batching, and stage plumbing
# ----------------------------------------------------------------------


def test_notes_are_keyed_per_stage_instance(tmp_path: Path) -> None:
    """Two instances must not overwrite each other's decision."""
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


def test_process_batch_flags_each_row_independently(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    tasks = [
        AudioTask(data={_TEXT_KEY: "the cat sat on the mat today", _SKIP_KEY: ""}),
        AudioTask(data={_TEXT_KEY: "yes yes yes yes yes yes", _SKIP_KEY: ""}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data[_SKIP_KEY] == ""
    assert "Hallucination" in results[1].data[_SKIP_KEY]


def test_missing_text_key_is_treated_as_empty(tmp_path: Path) -> None:
    """The filter treats a missing transcript as empty text."""
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""
    assert result.data["additional_notes"]["WhisperHallucination"] == "skipped (empty text)"


def test_prefix_match_length_is_a_constant_not_a_constructor_argument() -> None:
    """It is annotated at class scope, so guard against it becoming a field."""
    assert "_PREFIX_MATCH_MIN_LEN" not in inspect.signature(WhisperHallucinationStage).parameters


def test_stage_declares_its_io_contract(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    required_attrs, required_columns = stage.inputs()
    _required_out, optional_out = stage.outputs()
    assert required_attrs == []
    assert required_columns == []
    assert {_SKIP_KEY, "additional_notes"} <= set(optional_out)


def test_stage_is_cpu_only(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage.resources.gpus == 0
    assert stage.resources.cpus == 1.0


def test_stage_is_exported_from_the_audio_package() -> None:
    import nemo_curator.stages.audio as audio_pkg

    assert audio_pkg.WhisperHallucinationStage is WhisperHallucinationStage
