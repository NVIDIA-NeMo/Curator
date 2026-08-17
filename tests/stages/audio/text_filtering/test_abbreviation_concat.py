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

import pytest

from nemo_curator.stages.audio import (
    AbbreviationConcatStage as PublicAbbreviationConcatStage,
)
from nemo_curator.stages.audio import (
    RegexSubstitutionStage as PublicRegexSubstitutionStage,
)
from nemo_curator.stages.audio.text_filtering.abbreviation_concat import (
    AbbreviationConcatStage,
    concat_abbreviations,
)
from nemo_curator.stages.audio.text_filtering.regex_substitution import RegexSubstitutionStage
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("the A P I uses G P U acceleration", "en", "the API uses GPU acceleration"),
        ("the U K's policy", "en", "the UK's policy"),
        ("A P Xs", "en", "APXs"),
        ("A P Is", "en", "AP Is"),
        ("A Is", "en", "A Is"),
        ("a A P I", "en", "a API"),
        ("A P I a", "en", "API a"),
        ("D N a and R N a", "en", "DNa and RNa"),
        ("I I and A A", "en", "I I and A A"),
        ("x I and A x", "en", "x I and A x"),
        ("A B I\u2019m", "en", "AB\u2019m"),
        ("A B I\u2019s", "en", "ABI\u2019s"),
        ("А Б В", "ru", "АБВ"),  # noqa: RUF001
        ("Α Β Γ", "el", "ΑΒΓ"),  # noqa: RUF001
        ("a cat sat nearby", "en", "a cat sat nearby"),
    ],
)
def test_concat_abbreviations(text: str, language: str, expected: str) -> None:
    result, _ = concat_abbreviations(text, language=language)

    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A P I and G P U", ["API", "GPU"]),
        ("a A P I", ["API"]),
        ("A P Is", ["AP"]),
    ],
)
def test_reports_each_changed_abbreviation(text: str, expected: list[str]) -> None:
    _, found = concat_abbreviations(text)

    assert found == expected


def test_stage_records_changed_abbreviations() -> None:
    task = AudioTask(data={"text": "a A P I"})

    AbbreviationConcatStage().process(task)

    assert task.data["text"] == "a API"
    assert task.data["additional_notes"]["AbbreviationConcat"] == "applied (A P I -> API)"


def test_stage_preserves_skipped_rows() -> None:
    task = AudioTask(data={"text": "A P I", "_skipme": "read_error"})

    AbbreviationConcatStage().process(task)

    assert task.data["text"] == "A P I"
    assert "additional_notes" not in task.data


def test_stage_normalizes_language_code() -> None:
    task = AudioTask(data={"text": "А Б В", "source_lang": "RU"})  # noqa: RUF001

    AbbreviationConcatStage().process(task)

    assert task.data["text"] == "АБВ"


def test_stage_preserves_non_string_input() -> None:
    task = AudioTask(data={"text": None})

    AbbreviationConcatStage().process(task)

    assert task.data["text"] is None
    assert "additional_notes" not in task.data


def test_audio_package_exports_stages() -> None:
    assert PublicAbbreviationConcatStage is AbbreviationConcatStage
    assert PublicRegexSubstitutionStage is RegexSubstitutionStage
