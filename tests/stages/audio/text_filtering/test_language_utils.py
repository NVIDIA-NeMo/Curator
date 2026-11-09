# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# ruff: noqa: INP001

import pytest

from nemo_curator.stages.audio.text_filtering.language_utils import (
    normalize_language_to_code,
    parse_language_prediction,
    verify_language,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" HI ", "hi"),
        ('"Hindi-hi."', "hi"),
        ("The text is in hInDi.", "hi"),
        ("It is written in Bengali", "bn"),
        ("Bodo", "brx"),
        ("Maithili", "mai"),
        ("Filipino", "fil"),
        ("Tagalog", "tl"),
        (None, None),
        ("", None),
        ("unrecognized", None),
    ],
)
def test_normalize_language_labels(value: str | None, expected: str | None) -> None:
    assert normalize_language_to_code(value) == expected


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("Hindi", ("hi", ["hi"])),
        (" primary: Hindi\n LANGUAGES: English, Hindi, en, unknown", ("hi", ["hi", "en"])),
        ("Languages: Bodo, Hindi", ("brx", ["brx", "hi"])),
        ("Primary: unknown\nLanguages: Hindi, English", (None, ["hi", "en"])),
        ("Primary: unknown\nLanguages: Hindi, HI", (None, ["hi"])),
    ],
)
def test_parse_primary_and_unique_languages(prediction: str, expected: tuple[str | None, list[str]]) -> None:
    assert parse_language_prediction(prediction) == expected


@pytest.mark.parametrize(
    ("prediction", "source", "skip", "note"),
    [
        ("Hindi", "Hindi-hi", None, "passed (Hindi-hi)"),
        (
            "English",
            "hi",
            "Wrong language:LLMLanguageVerification",
            "wrong language (predicted=English-en, expected=Hindi-hi)",
        ),
        (
            "unknown",
            "hi",
            "Unparseable language:LLMLanguageVerification",
            "unparseable prediction ('unknown', expected=Hindi-hi)",
        ),
        (
            "Primary: unknown\nLanguages: English, French",
            "hi",
            None,
            "kept code-switch (primary=unparseable, langs=[English-en, French-fr], expected=Hindi-hi, source absent)",
        ),
        (
            "Primary: English\nLanguages: Hindi, English",
            "hi",
            None,
            "kept code-switch (primary=English-en, langs=[English-en, Hindi-hi], expected=Hindi-hi, source present)",
        ),
        (
            "Primary: English\nLanguages: English, en",
            "hi",
            "Wrong language:LLMLanguageVerification",
            "wrong language (predicted=English-en, expected=Hindi-hi)",
        ),
        ("English", "unknown", None, "skipped (unknown source_lang='unknown')"),
        ("", "hi", None, "skipped (empty prediction)"),
        (None, "hi", None, "skipped (empty prediction)"),
    ],
)
def test_verify_language_decisions(prediction: str | None, source: str, skip: str | None, note: str) -> None:
    data = {"prediction": prediction, "expected": source, "notes": {"EarlierStage": "kept"}, "unrelated": [1]}
    assert verify_language(data, "prediction", "expected", "skip", "notes") is None
    assert data.get("skip") == skip
    assert data["notes"] == {"EarlierStage": "kept", "LLMLanguageVerification": note}
    assert (data["prediction"], data["expected"], data["unrelated"]) == (prediction, source, [1])


def test_existing_skip_flag_is_never_overwritten() -> None:
    data = {"_skipme": "prior filter", "source_lang": "hi", "llm_language_prediction": "English"}
    verify_language(data, "llm_language_prediction", "source_lang", "_skipme", "additional_notes")
    assert data["_skipme"] == "prior filter"
    assert data["additional_notes"] == {"LLMLanguageVerification": "skipped (already flagged)"}


def test_invalid_notes_container_is_replaced_without_rejecting_missing_prediction() -> None:
    data = {"additional_notes": "old notes"}
    verify_language(data, "llm_language_prediction", "source_lang", "_skipme", "additional_notes")
    assert data == {"additional_notes": {"LLMLanguageVerification": "skipped (empty prediction)"}}
