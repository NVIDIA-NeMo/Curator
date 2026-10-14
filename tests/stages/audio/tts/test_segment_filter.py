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


import copy
import math

import pytest

from nemo_curator.stages.audio.tts.segment_filter import (
    FILTER_THRESHOLD_PRESETS,
    FilterCondition,
    SegmentFilterStage,
    context_conditions,
    default_conditions,
    evaluate_condition,
    logical_language,
    parse_conditions,
)
from nemo_curator.tasks import AudioTask


def good_segment(**overrides: object) -> dict:
    segment = {
        "duration": 4.0,
        "GranaryV2": {"pnc_text": "Hello there, how are you?"},
        "sed": {"weighted_speech_rate": 0.9, "weighted_model_confidence": 0.5, "weighted_classifier_confidence": 0.3},
        "diarization": {"diarized_speech_rate": 0.9, "num_speakers": 1},
        "speaker_cluster": {"speaker_id": "vid_speaker_0"},
        "utmosv2": {"score": 3.0},
    }
    for path, value in overrides.items():
        current = segment
        parts = path.split("__")
        for part in parts[:-1]:
            current = current[part]
        if value is _DELETE:
            current.pop(parts[-1])
        else:
            current[parts[-1]] = value
    return segment


_DELETE = object()


def run(stage: SegmentFilterStage, row: dict) -> dict:
    return dict(stage.process(AudioTask(data=row)).data)


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("default", (0.5, 0.2, 0.13, 0.7, 1, 1.0)),
        ("yodas", (0.5, 0.2, 0.13, 0.7, 1, 1.0)),
        ("ytc", (0.5, 0.15, 0.1, 0.6, 1, 1.5)),
        ("voxpopuli", (0.5, 0.15, 0.1, 0.6, 1, 1.5)),
        ("librilight", (0.4, 0.2, 0.1, 0.2, 1, 2.5)),
    ],
)
def test_preset_thresholds(preset: str, expected: tuple) -> None:
    by_id = {c.id: c for c in default_conditions(preset)}
    got = tuple(
        by_id[name].value
        for name in (
            "sed_weighted_speech_rate",
            "sed_weighted_model_confidence",
            "sed_weighted_classifier_confidence",
            "diarized_speech_rate",
            "num_speakers",
            "utmos_v2_score",
        )
    )
    assert got == expected
    assert by_id["min_duration"].value == 3.0
    assert set(FILTER_THRESHOLD_PRESETS) == {"default", "yodas", "ytc", "voxpopuli", "librilight"}


def test_librilight_reads_sample_level_speaker() -> None:
    cond = {c.id: c for c in default_conditions("librilight")}["speaker_id_not_null"]
    assert (cond.field, cond.scope) == ("speaker_id", "sample")
    stage = SegmentFilterStage(preset="librilight", language="en")
    seg = good_segment(speaker_cluster=_DELETE, utmosv2__score=2.6)
    row = run(stage, {"speaker_id": "1234", "segments": [seg]})
    assert row["segments"] == [seg]
    row = run(stage, {"speaker_id": None, "segments": [copy.deepcopy(seg)]})
    assert row["segments"] == []


def test_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown filter preset"):
        default_conditions("nope")


def test_default_filter_keeps_good_and_drops_each_failure() -> None:
    stage = SegmentFilterStage(preset="default")
    failures = {
        "short": good_segment(duration=2.99),
        "no_letter": good_segment(GranaryV2__pnc_text="123 456"),
        "sed_rate": good_segment(sed__weighted_speech_rate=0.49),
        "sed_model": good_segment(sed__weighted_model_confidence=0.19),
        "sed_clf": good_segment(sed__weighted_classifier_confidence=0.12),
        "diar": good_segment(diarization__diarized_speech_rate=0.69),
        "two_spk": good_segment(diarization__num_speakers=2),
        "no_cluster": good_segment(speaker_cluster__speaker_id=None),
        "mos": good_segment(utmosv2__score=0.99),
        "charset": good_segment(GranaryV2__pnc_text="Hello 3 people"),
        "missing_sed": good_segment(sed=_DELETE),
    }
    good = good_segment()
    row = run(stage, {"source_lang": "en", "segments": [good, *failures.values()]})
    assert row["segments"] == [good]
    assert stage.counters["kept_segments"] == 1
    assert stage.counters["filtered_segments"] == len(failures)
    assert stage.counters["condition_sed_weighted_speech_rate_missing"] == 1
    assert stage.counters["condition_min_duration_failed"] == 1


def test_thresholds_are_inclusive() -> None:
    stage = SegmentFilterStage(preset="default", language="en")
    seg = good_segment(
        duration=3.0,
        sed__weighted_speech_rate=0.5,
        sed__weighted_model_confidence=0.2,
        sed__weighted_classifier_confidence=0.13,
        diarization__diarized_speech_rate=0.7,
        utmosv2__score=1.0,
    )
    assert run(stage, {"segments": [seg]})["segments"] == [seg]


def test_row_kept_when_all_segments_fail_and_non_list_rows_untouched() -> None:
    stage = SegmentFilterStage(preset="ytc", language="en")
    row = run(stage, {"orig_sample_id": "x", "segments": [good_segment(duration=1.0), "junk"]})
    assert row == {"orig_sample_id": "x", "segments": []}
    assert stage.counters["samples_without_kept_segments"] == 1
    flat = {"duration": 1.0}
    assert run(stage, dict(flat)) == flat


def test_drop_false_records_reasons() -> None:
    stage = SegmentFilterStage(preset="default", language="en", drop=False, reasons_key="why")
    good, bad = good_segment(), good_segment(duration=1.0, utmosv2__score=0.5)
    row = run(stage, {"segments": [good, bad]})
    assert len(row["segments"]) == 2
    assert good["why"] == []
    assert bad["why"] == ["min_duration", "utmos_v2_score"]


def test_combine_any() -> None:
    stage = SegmentFilterStage(
        conditions=[
            {"id": "a", "field": "x", "operator": "ge", "value": 1},
            {"id": "b", "field": "y", "operator": "exists"},
        ],
        combine="any",
    )
    row = run(stage, {"segments": [{"x": 0, "y": 1}, {"x": 2}, {"x": 0}]})
    assert row["segments"] == [{"x": 0, "y": 1}, {"x": 2}]


@pytest.mark.parametrize(
    ("operator", "value", "field_value", "expected"),
    [
        ("eq", 1, 1, True),
        ("eq", 1, 1.0, True),
        ("eq", 1, "1", False),
        ("ne", 1, 2, True),
        ("gt", 1, 1, False),
        ("gt", 1, 1.5, True),
        ("ge", 1, 1, True),
        ("lt", 1, 0.5, True),
        ("le", 1, 1, True),
        ("ge", 1, "3.5", True),
        ("ge", 1, "abc", False),
        ("ge", 0, True, False),
        ("ge", 0, math.nan, False),
        ("le", 10, math.inf, False),
        ("ge", 0, None, False),
        ("in", [1, 2], 2, True),
        ("not_in", [1, 2], 2, False),
        ("is_null", None, None, True),
        ("not_null", None, 0, True),
        ("not_null", None, None, False),
        ("exists", None, None, True),
        ("not_exists", None, 0, False),
    ],
)
def test_operators(operator: str, value: object, field_value: object, expected: bool) -> None:
    cond = FilterCondition("c", "a.b", operator, value=value)
    assert evaluate_condition(cond, {"a": {"b": field_value}}, "") == (expected, False)


@pytest.mark.parametrize("operator", ["eq", "ne", "ge", "is_null", "not_null", "in", "regex", "exists"])
def test_missing_field_fails_and_is_counted_missing(operator: str) -> None:
    value = [1] if operator == "in" else 1
    cond = FilterCondition("c", "a.b", operator, value=value, pattern="x")
    assert evaluate_condition(cond, {"a": {}}, "en") == (False, True)


def test_not_exists_on_missing_passes() -> None:
    assert evaluate_condition(FilterCondition("c", "a.b", "not_exists"), {}, "") == (True, True)


def test_operator_aliases_and_validation() -> None:
    assert FilterCondition("c", "f", "==").operator == "eq"
    assert FilterCondition("c", "f", "!=").operator == "ne"
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        FilterCondition("c", "f", "approx")
    with pytest.raises(ValueError, match="requires field"):
        FilterCondition("c", " ", "eq")
    with pytest.raises(ValueError, match="scope"):
        FilterCondition("c", "f", "eq", scope="video")
    with pytest.raises(ValueError, match="requires pattern"):
        FilterCondition("c", "f", "regex")
    with pytest.raises(ValueError, match="Invalid regex"):
        FilterCondition("c", "f", "regex", pattern="(")
    with pytest.raises(TypeError, match="list value"):
        FilterCondition("c", "f", "in", value=3)
    with pytest.raises(ValueError, match="match_mode"):
        FilterCondition("c", "f", "regex", pattern="x", match_mode="partial")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_conditions([{"id": "a", "field": "x"}, {"id": "a", "field": "y"}])
    with pytest.raises(ValueError, match="non-empty"):
        parse_conditions([])


def test_regex_match_modes_and_language_patterns() -> None:
    search = FilterCondition("c", "t", "regex", pattern=r"\d", match_mode="search")
    match = FilterCondition("c", "t", "regex", pattern=r"\d", match_mode="match")
    full = FilterCondition("c", "t", "regex", pattern=r"\d", match_mode="fullmatch")
    assert evaluate_condition(search, {"t": "a1"}, "")[0]
    assert not evaluate_condition(match, {"t": "a1"}, "")[0]
    assert not evaluate_condition(full, {"t": "12"}, "")[0]
    assert evaluate_condition(FilterCondition("c", "t", "not_regex", pattern=r"\d"), {"t": "ab"}, "")[0]
    by_lang = FilterCondition("c", "t", "regex", pattern_by_language={"DE": "[a-zäöü ]+"})
    assert evaluate_condition(by_lang, {"t": "schön"}, "de")[0]
    assert not evaluate_condition(by_lang, {"t": "schön"}, "fr")[0]  # no pattern for language -> fail
    assert not evaluate_condition(by_lang, {"t": None}, "de")[0]


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        ("de", "Grüße aus Köln, Straße!", True),
        ("en", "Grüße aus Köln", False),
        ("fr", "C'est l'été « déjà » !", True),
        ("bg", "Здравей, свят!", True),
        ("bg", "Hello", False),
        ("uk", "Привіт, ґава!", True),
        ("el", "Καλημέρα κόσμε;", True),
        ("es", "¿Qué tal, señor?", True),
        ("en", "It costs 5 dollars", False),
        ("en", "50% off", False),
        ("it", "Perché è così…", True),
    ],
)
def test_charset_by_language(language: str, text: str, expected: bool) -> None:
    cond = {c.id: c for c in default_conditions()}["pnc_text_charset"]
    assert evaluate_condition(cond, {"GranaryV2": {"pnc_text": text}}, language)[0] is expected


def test_language_resolution() -> None:
    stage = SegmentFilterStage(manifest_path_key="manifest")
    assert stage.language_for_row({"source_lang": "de_DE"}) == "de"
    assert stage.language_for_row({"language": "FR"}) == "fr"
    assert stage.language_for_row({"manifest": "/data/ytc/bg000/manifest_0.jsonl"}) == "bg"
    assert stage.language_for_row({}) == ""
    assert SegmentFilterStage(language="it").language_for_row({"source_lang": "de"}) == "it"
    assert logical_language("en_x") == "en"
    assert logical_language("123") == "123"


def test_context_conditions() -> None:
    stage = SegmentFilterStage(conditions=context_conditions())
    ok = {
        "context": {
            "error": None,
            "context_recording_id": "a.wav",
            "context_audio_offset": 512,
            "context_audio_bytes": 10,
        }
    }
    bad = {"context": {**ok["context"], "error": "no_context_segment_for_speaker"}}
    assert run(stage, {"segments": [ok, bad, {}]})["segments"] == [ok]


def test_process_batch_empty_and_multiple() -> None:
    stage = SegmentFilterStage(language="en")
    assert stage.process_batch([]) == []
    tasks = [AudioTask(data={"segments": [good_segment()]}), AudioTask(data={"segments": [good_segment(duration=0)]})]
    out = stage.process_batch(tasks)
    assert [len(t.data["segments"]) for t in out] == [1, 0]


def test_warns_when_a_field_is_missing_on_every_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "nemo_curator.stages.audio.tts.segment_filter.logger.warning", lambda msg, *a: warnings.append(msg.format(*a))
    )
    stage = SegmentFilterStage(preset="ytc", language="en")
    stage.process_batch([AudioTask(data={"segments": [good_segment()]})])
    assert warnings == []

    bad = good_segment()
    del bad["utmosv2"]
    stage.process_batch([AudioTask(data={"segments": [bad, copy.deepcopy(bad)]})])
    assert len(warnings) == 1
    assert "utmos" in warnings[0]
    assert "2/2" in warnings[0]
