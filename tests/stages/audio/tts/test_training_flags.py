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

# ruff: noqa: RUF001  (seeded test RNG; IPA test strings)


import pytest

from nemo_curator.stages.audio.tts.context import compute_text_overlap
from nemo_curator.stages.audio.tts.training_flags import (
    TrainingFlagsStage,
    compute_flags,
    ipa_value,
    is_early_cut_off,
    passes_training_filter,
    valid_ipa,
    valid_tn_raw_transcript,
)
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", True),
        ("Hello, world! «Ça va?» — oui…", True),
        ("Grüße aus Köln", True),
        ("Здравей свят", True),
        ("twenty five euros", True),
        ("25 euros", False),
        ("fifty %", True),  # "%" is Unicode punctuation (Po), accepted like the reference
        ("five $", False),  # currency symbols (Sc) are not
        ("five €", False),
        ("a+b", False),
        ("", False),
        ("   ", False),
        ("...", False),
        (None, False),
        (5, False),
        ("école", True),  # NFC-composed é
    ],
)
def test_valid_tn_raw_transcript(text: object, expected: bool) -> None:
    assert valid_tn_raw_transcript(text) is expected


def test_valid_ipa_and_ipa_value() -> None:
    assert valid_ipa("həˈloʊ")
    assert not valid_ipa("  ")
    assert not valid_ipa("həˈloʊ", "ipa_failed")
    assert not valid_ipa(None)
    assert ipa_value({"ipa": "x", "error": None}) == ("x", None)
    assert ipa_value({"ipa": None, "error": "empty_text"}) == (None, "empty_text")
    assert ipa_value("abc") == ("abc", None)
    assert ipa_value(None) == (None, None)


@pytest.mark.parametrize(
    ("decision", "expected"),
    [("DROP", True), ("review", True), (" REVIEW ", True), ("OK", False), (None, False), (1, False)],
)
def test_is_early_cut_off(decision: object, expected: bool) -> None:
    assert is_early_cut_off(decision) is expected


def test_compute_flags_and_training_predicate() -> None:
    flags = compute_flags("the cat sat", "ðə kæt sæt", None, "OK", "a dog ran far away")
    assert flags == {
        "valid_tn_raw_transcript": True,
        "valid_ipa": True,
        "early_cut_off": False,
        "valid_context_audio_text": True,
        "context_audio_text_overlap": round(compute_text_overlap("the cat sat", "a dog ran far away"), 4),
        "context_tn_raw": "a dog ran far away",
    }
    assert passes_training_filter(flags)
    assert not passes_training_filter({**flags, "early_cut_off": True})
    assert not passes_training_filter({**flags, "context_audio_text_overlap": 0.5})
    assert not passes_training_filter({**flags, "context_audio_text_overlap": None})
    no_ctx = compute_flags("the cat", "x", None, None, None)
    assert no_ctx["context_audio_text_overlap"] is None
    assert no_ctx["valid_context_audio_text"] is False


def _segment(member: str, tn_raw: str | None, **extra: object) -> dict:
    segment = {
        "GranaryV2": {"tn_raw": tn_raw, "pnc_text": (tn_raw or "").capitalize()},
        "GranaryHifi": {"AudioMetadata": {"tar_member": member, "tar_member_offset": 512, "tar_member_bytes": 10}},
    }
    segment.update(extra)
    return segment


def _ctx(member: str | None, **extra: object) -> dict:
    return {
        "context_recording_id": member,
        "context_audio_offset": 512,
        "context_audio_bytes": 10,
        "error": None if member else "no_context_segment_for_speaker",
        **extra,
    }


def test_stage_flags_from_context_annotation_and_lookup() -> None:
    a = _segment(
        "a.wav",
        "hello there friend",
        ipa={"ipa": "həloʊ", "error": None},
        early_cut_off_detection={"decision": "OK"},
        context=_ctx("b.wav", context_audio_tn_raw="totally different words"),
    )
    b = _segment("b.wav", "twenty one apples", ipa="ˈtwɛnti", context=_ctx("a.wav"))
    c = _segment("c.wav", "3 apples", early_cut_off_detection={"decision": "REVIEW"}, context=_ctx(None))
    d = _segment("d.wav", "hello", context=_ctx("gone.wav"))
    out = TrainingFlagsStage().process(AudioTask(data={"segments": [a, b, c, d]})).data["segments"]
    fa, fb, fc, fd = (s["tts_flags"] for s in out)
    assert fa["context_tn_raw"] == "totally different words"
    assert passes_training_filter(fa)
    assert fb["context_tn_raw"] == "hello there friend"  # looked up in the same row by tar member
    assert fb["valid_ipa"] is True
    assert fb["early_cut_off"] is False
    assert fc["valid_tn_raw_transcript"] is False
    assert fc["early_cut_off"] is True
    assert fc["valid_ipa"] is False
    assert fc["context_tn_raw"] is None
    assert fd["context_tn_raw"] is None
    assert fd["valid_context_audio_text"] is False


def test_lookup_disambiguates_by_byte_range() -> None:
    first = _segment("dup.wav", "first text")
    second = _segment("dup.wav", "second text")
    second["GranaryHifi"]["AudioMetadata"]["tar_member_offset"] = 2048
    target = _segment("t.wav", "target text", context=_ctx("dup.wav", context_audio_offset=2048))
    out = TrainingFlagsStage().process(AudioTask(data={"segments": [first, second, target]})).data
    assert out["segments"][2]["tts_flags"]["context_tn_raw"] == "second text"


def test_flat_row_custom_keys_and_batch() -> None:
    row = {
        "GranaryV2": {"tn_raw": "flat row"},
        "IPA": {"ipa": "flæt", "error": None},
        "ECO": {"decision": "DROP"},
    }
    stage = TrainingFlagsStage(output_key="GranaryHifi.Flags", ipa_key="IPA", early_cut_off_key="ECO")
    out = stage.process_batch([AudioTask(data=row)])[0].data
    flags = out["GranaryHifi"]["Flags"]
    assert flags["valid_tn_raw_transcript"] is True
    assert flags["valid_ipa"] is True
    assert flags["early_cut_off"] is True
    assert stage.process_batch([]) == []
