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

from nemo_curator.stages.audio.asr.normalization import TranscriptNormalizationStage
from nemo_curator.stages.audio.asr.normalization.transcript import (
    _PUNCTUATION_CHARS_BY_LANG,
    _RESOURCE_ROOT,
    _load_alphabet,
)
from nemo_curator.tasks import AudioTask


def test_language_resources_are_flattened_without_data_subdirectory() -> None:
    assert (_RESOURCE_ROOT / "remove_chars.txt").exists()
    for lang in ["gu", "hi", "kn", "te", "ml", "mr", "pa", "ta", "bn", "ur", "en"]:
        lang_dir = _RESOURCE_ROOT / lang
        assert (lang_dir / "alphabet.txt").exists()
        assert (lang_dir / "pretok.jsonl").exists()
        assert not (lang_dir / "remove_chars.txt").exists()
        assert not (lang_dir / "pnc_chars.txt").exists()
        assert not (lang_dir / "data").exists()
        assert lang in _PUNCTUATION_CHARS_BY_LANG


def test_standard_punctuation_dictionary_preserves_language_specific_chars() -> None:
    assert _PUNCTUATION_CHARS_BY_LANG["bn"] == ",?\u0964"
    assert _PUNCTUATION_CHARS_BY_LANG["ml"] == ".,?!\u0964\u0965"
    assert _PUNCTUATION_CHARS_BY_LANG["pa"] == ".,?\u0964"
    assert _PUNCTUATION_CHARS_BY_LANG["ur"] == "\u060c\u061f\u06d4"


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("kn", "ಕನ್ನಡ ವಾಕ್ಯ"),
        ("te", "తెలుగు వాక్యం"),
        ("ml", "മലയാളം വാക്യം"),
        ("mr", "मराठी वाक्य"),
        ("pa", "ਪੰਜਾਬੀ ਵਾਕ"),
        ("ta", "தமிழ் வாக்கியம்"),
        ("bn", "বাংলা বাক্য"),
        ("ur", "اردو جملہ"),
    ],
)
def test_additional_indic_language_resources_are_loadable(lang: str, text: str) -> None:
    stage = TranscriptNormalizationStage()
    task = AudioTask(data={"text": text, "lang": lang})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == text
    assert task.data["transcript_error"] is False
    assert task.data["unknown_chars"] == {}


def test_alphabet_loader_includes_uppercase_variants_for_each_letter(tmp_path: Path) -> None:
    alphabet_path = tmp_path / "alphabet.txt"
    alphabet_path.write_text("a\nb\n", encoding="utf-8")

    assert _load_alphabet(alphabet_path) == {"a", "A", "b", "B"}


def test_gujarati_text_is_cleaned_in_place_and_marked_valid() -> None:
    stage = TranscriptNormalizationStage()
    task = AudioTask(data={"text": " ગુજરાતી—વાક્ય @@@ ", "lang": "gu", "duration": 1.5})

    result = stage.process(task)
    metrics = stage._consume_custom_metrics()

    assert result is task
    assert task.data["text"] == "ગુજરાતી વાક્ય"
    assert task.data["text_original"] == " ગુજરાતી—વાક્ય @@@ "
    assert task.data["transcript_error"] is False
    assert task.data["unknown_chars"] == {}
    assert metrics["input_tasks"] == 1
    assert metrics["emitted_tasks"] == 1


def test_lowercase_text_can_be_enabled() -> None:
    stage = TranscriptNormalizationStage(lowercase_text=True)
    task = AudioTask(data={"text": "Hello WORLD", "lang": "en"})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == "hello world"
    assert task.data["transcript_error"] is False
    assert task.data["unknown_chars"] == {}


def test_code_switch_languages_extend_known_alphabet() -> None:
    stage = TranscriptNormalizationStage(code_switch_langs=["en"])
    task = AudioTask(data={"text": "ગુજરાતી Hello", "lang": "gu"})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == "ગુજરાતી Hello"
    assert task.data["transcript_error"] is False
    assert task.data["unknown_chars"] == {}


def test_code_switch_language_accepts_single_string() -> None:
    stage = TranscriptNormalizationStage(code_switch_langs="en")
    task = AudioTask(data={"text": "ગુજરાતી Hello", "lang": "gu"})

    result = stage.process(task)

    assert result is task
    assert task.data["transcript_error"] is False
    assert task.data["unknown_chars"] == {}


def test_code_switch_languages_extend_punctuation_removal() -> None:
    stage = TranscriptNormalizationStage(code_switch_langs=["en"])
    task = AudioTask(data={"text": "ગુજરાતી - Hello", "lang": "gu"})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == "ગુજરાતી Hello"
    assert task.data["transcript_error"] is False
    assert task.data["unknown_chars"] == {}


def test_hindi_pretok_replacements_are_applied() -> None:
    stage = TranscriptNormalizationStage()
    task = AudioTask(data={"text": "₹ ऎॊय़ऱॆ ळ <hi-IN> शब्द", "lang": "hi"})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == "रुपये ऐोयरे ल शब्द"
    assert task.data["transcript_error"] is False


def test_language_must_match_resource_directory_name() -> None:
    stage = TranscriptNormalizationStage()
    task = AudioTask(data={"text": "शब्द", "lang": "hindi"})

    with pytest.raises(ValueError, match="Unsupported ASR normalization language"):
        stage.process(task)


def test_unknown_chars_are_recorded_and_task_is_retained() -> None:
    stage = TranscriptNormalizationStage()
    task = AudioTask(data={"text": "ગુજરાતી xyz x ૧", "lang": "gu", "duration": 2.0})

    result = stage.process(task)
    metrics = stage._consume_custom_metrics()

    assert result is task
    assert task.data["transcript_error"] is True
    assert task.data["unknown_chars"] == {"x": 2, "y": 1, "z": 1, "૧": 1}
    assert metrics["input_tasks"] == 1
    assert metrics["emitted_tasks"] == 1
    assert metrics["unknown_duration_seconds"] == pytest.approx(2.0)


def test_punctuation_removal_can_be_disabled() -> None:
    stage = TranscriptNormalizationStage(remove_pnc_chars=False)
    task = AudioTask(data={"text": "ગુજરાતી . વાક્ય", "lang": "gu", "duration": 2.0})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == "ગુજરાતી . વાક્ય"
    assert task.data["transcript_error"] is True
    assert task.data["unknown_chars"] == {".": 1}


def test_output_keys_keep_source_text_unchanged() -> None:
    stage = TranscriptNormalizationStage(output_text_key="normalized_text", output_original_text_key="raw_text")
    task = AudioTask(data={"text": " ગુજરાતી—વાક્ય ", "lang": "gu"})

    result = stage.process(task)

    assert result is task
    assert task.data["text"] == " ગુજરાતી—વાક્ય "
    assert task.data["raw_text"] == " ગુજરાતી—વાક્ય "
    assert task.data["normalized_text"] == "ગુજરાતી વાક્ય"
