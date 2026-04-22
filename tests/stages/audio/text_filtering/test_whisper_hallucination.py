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

from pathlib import Path

import pytest

from nemo_curator.stages.audio.text_filtering.whisper_hallucination import WhisperHallucinationStage
from nemo_curator.tasks import AudioTask


def _make_stage(tmp_path: Path, phrases: list[str]) -> WhisperHallucinationStage:
    p = tmp_path / "phrases.txt"
    p.write_text("\n".join(phrases), encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(p), text_key="cleaned_text")
    stage.setup()
    return stage


def test_clean_text_passes(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["lorem ipsum"])
    task = AudioTask(data={"cleaned_text": "the cat sat on the mat today", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == ""


def test_repeated_ngrams_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={"cleaned_text": "yes yes yes yes yes yes", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_long_word_absolute_threshold_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    long_word = "a" * 30
    task = AudioTask(data={"cleaned_text": f"the {long_word} here", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_long_word_relative_threshold_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={"cleaned_text": "cat verylongwordindeed", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_frequent_phrase_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you 1297"])
    task = AudioTask(data={"cleaned_text": "Thank you", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_frequent_phrase_strips_punctuation(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={"cleaned_text": "Thank you.", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_frequent_phrase_strips_trailing_comma(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you"])
    task = AudioTask(data={"cleaned_text": "Thank you,", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_setup_called_lazily_when_skipped(tmp_path: Path) -> None:
    p = tmp_path / "phrases.txt"
    p.write_text("Thank you\n", encoding="utf-8")
    stage = WhisperHallucinationStage(common_hall_file=str(p), text_key="cleaned_text")
    task = AudioTask(data={"cleaned_text": "Thank you", "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == "Hallucination"


def test_non_string_text_returns_task_unchanged(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={"cleaned_text": None, "skip_me": ""})
    result = stage.process(task)
    assert result.data["skip_me"] == ""


def test_preserves_existing_skip_me_reason(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={"cleaned_text": "yes yes yes yes yes yes", "skip_me": "Wrong language"})
    result = stage.process(task)
    assert result.data["skip_me"] == "Wrong language"


def test_empty_words_not_flagged_by_ngram(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage._repeated_ngrams([]) is False


def test_empty_words_not_flagged_by_long_word(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    assert stage._long_word([]) is False


def test_phrases_file_strips_frequency_count(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, ["Thank you 1297", "Amen -1", "Yeah 217"])
    assert "Thank you" in stage._phrases
    assert "Amen" in stage._phrases
    assert "Yeah" in stage._phrases


def test_requires_common_hall_file() -> None:
    with pytest.raises(ValueError, match="common_hall_file is required"):
        WhisperHallucinationStage(common_hall_file="")
