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

"""Truth-table tests for the reference audio-LID ensemble selector."""

from __future__ import annotations

import pytest

from nemo_curator.stages.audio.postprocessing import SelectAudioLanguageStage as PublicSelectAudioLanguageStage
from nemo_curator.stages.audio.postprocessing.lid_selection import SelectAudioLanguageStage
from nemo_curator.tasks import AudioTask

_SPEECHBRAIN = "SpeechBrainLangID"
_AMBERNET = "AmberNetLangID"
_CANARY = "IndicCanaryLangID"
_WHISPER = "WhisperLangID"


def _prediction(language: str, confidence: float, tag: str) -> dict[str, object]:
    return {"language": language, "confidence": confidence, "tag": tag}


def _task(results: dict[str, object] | None = None, **extra: object) -> AudioTask:
    data: dict[str, object] = dict(extra)
    if results is not None:
        data["lid"] = results
    return AudioTask(data=data)


def _full_results(*, primary: str = "hi", canary: str = "hi", whisper: str = "hi") -> dict[str, object]:
    return {
        _SPEECHBRAIN: _prediction(primary, 0.7, "primary"),
        _CANARY: _prediction(canary, 0.8, "secondary"),
        _WHISPER: _prediction(whisper, 0.9, "tertiary"),
    }


def test_missing_predictions_use_the_reference_empty_skip_sentinel() -> None:
    task = SelectAudioLanguageStage().process(_task())

    assert task.data["source_lang"] == ""
    assert task.data["_skipme"] == "skipped due to missing langID predictions."
    assert task.data["additional_notes"] == {
        "source_lid_confidence": 0.0,
        "SelectBestLIDPrediction": "skipped (missing predictions)",
    }


def test_whisper_english_is_accepted_before_any_agreement_check() -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results(primary="fr", canary="hi", whisper="en")))

    assert task.data["source_lang"] == "en"
    assert task.data["additional_notes"]["source_lid_confidence"] == 0.9
    assert task.data["additional_notes"]["SelectBestLIDPrediction"] == "used tertiary, English language."
    assert "_skipme" not in task.data
    assert "lid" not in task.data


@pytest.mark.parametrize(
    "results",
    [
        {
            _SPEECHBRAIN: _prediction("fr", 0.7, "primary"),
            _CANARY: _prediction("hi", 0.8, "secondary"),
        },
        {
            _SPEECHBRAIN: _prediction("fr", 0.7, "primary"),
            _WHISPER: _prediction("", 0.0, "tertiary"),
        },
    ],
)
def test_missing_or_empty_whisper_uses_the_reference_skip_reason(results: dict[str, object]) -> None:
    task = SelectAudioLanguageStage().process(_task(results))

    assert task.data["source_lang"] == ""
    assert task.data["_skipme"] == "skipped due to missing or empty whisper langID prediction."
    assert task.data["additional_notes"]["source_lid_confidence"] == 0.0
    assert (
        task.data["additional_notes"]["SelectBestLIDPrediction"]
        == "skipped (missing or empty whisper langID prediction)"
    )


def test_three_way_agreement_uses_the_canary_result() -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results()))

    assert task.data["source_lang"] == "hi"
    assert task.data["additional_notes"]["source_lid_confidence"] == 0.8
    assert (
        task.data["additional_notes"]["SelectBestLIDPrediction"]
        == "used secondary, agreement between all 3 langID models."
    )


def test_non_indic_primary_whisper_agreement_uses_whisper_even_when_canary_disagrees() -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results(primary="fr", canary="hi", whisper="fr")))

    assert task.data["source_lang"] == "fr"
    assert task.data["additional_notes"]["source_lid_confidence"] == 0.9
    assert (
        task.data["additional_notes"]["SelectBestLIDPrediction"]
        == "used tertiary, agreement between primary and tertiary langID models."
    )


def test_indic_language_without_three_way_agreement_is_rejected() -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results(primary="hi", canary="ta", whisper="hi")))

    assert task.data["source_lang"] == "skipped"
    assert task.data["_skipme"] == "skipped due to disagreement between langID models."
    assert task.data["additional_notes"]["source_lid_confidence"] == 0.0
    assert (
        task.data["additional_notes"]["SelectBestLIDPrediction"]
        == "skipped due to disagreement between langID models."
    )


def test_ambernet_can_fill_the_primary_role() -> None:
    results = {
        _AMBERNET: _prediction("de", 0.7, "primary"),
        _WHISPER: _prediction("de", 0.9, "tertiary"),
    }

    task = SelectAudioLanguageStage().process(_task(results))

    assert task.data["source_lang"] == "de"
    assert task.data["additional_notes"]["primary_lid_model"] == "ambernet"


def test_the_last_primary_in_mapping_order_matches_reference_behavior() -> None:
    results = {
        _SPEECHBRAIN: _prediction("de", 0.7, "speechbrain_primary"),
        _AMBERNET: _prediction("fr", 0.8, "ambernet_primary"),
        _WHISPER: _prediction("fr", 0.9, "tertiary"),
    }

    task = SelectAudioLanguageStage().process(_task(results))

    assert task.data["source_lang"] == "fr"
    assert (
        task.data["additional_notes"]["SelectBestLIDPrediction"]
        == "used tertiary, agreement between ambernet_primary and tertiary langID models."
    )


def test_component_notes_record_model_prediction_and_three_decimal_confidence() -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results()))
    notes = task.data["additional_notes"]

    assert notes["primary_lid_model"] == "speechbrain"
    assert notes["primary_lid_prediction"] == "hi"
    assert notes["primary_lid_confidence"] == "0.700"
    assert notes["secondary_lid_model"] == "indic_canary"
    assert notes["tertiary_lid_model"] == "whisper"


def test_custom_model_ids_and_notes_key_work_on_the_english_fast_path() -> None:
    stage = SelectAudioLanguageStage(
        speechbrain_model_id="primary-id",
        ambernet_model_id="alternate-primary-id",
        indic_canary_model_id="canary-id",
        whisper_model_id="whisper-id",
        notes_key="audit",
    )
    task = _task(
        None,
        lid={
            "primary-id": _prediction("fr", 0.7, "p"),
            "whisper-id": _prediction("en", 0.6, "w"),
        },
        audit="legacy non-mapping value",
    )

    result = stage.process(task)

    assert result.data["source_lang"] == "en"
    assert result.data["audit"]["source_lid_confidence"] == 0.6
    assert result.data["audit"]["SelectBestLIDPrediction"] == "used w, English language."
    assert result.data["audit"]["p_lid_model"] == "speechbrain"
    assert "additional_notes" not in result.data


@pytest.mark.parametrize(
    "owned_reason",
    [
        "skipped due to missing langID predictions.",
        "skipped due to missing or empty whisper langID prediction.",
        "skipped due to disagreement between langID models.",
    ],
)
def test_success_clears_only_a_stale_selector_owned_skip_reason(owned_reason: str) -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results(whisper="en"), _skipme=owned_reason))
    assert "_skipme" not in task.data


def test_success_preserves_an_unrelated_upstream_skip_reason() -> None:
    task = SelectAudioLanguageStage().process(_task(_full_results(whisper="en"), _skipme="bad transcript"))
    assert task.data["source_lang"] == "en"
    assert task.data["_skipme"] == "bad transcript"


def test_custom_indic_set_changes_only_the_non_indic_agreement_rule() -> None:
    stage = SelectAudioLanguageStage(indic_languages=frozenset({"xx"}))
    task = stage.process(_task(_full_results(primary="hi", canary="ta", whisper="hi")))
    assert task.data["source_lang"] == "hi"
    assert "_skipme" not in task.data


def test_unknown_model_ids_are_rejected() -> None:
    task = _task({"UnknownLID": _prediction("en", 0.5, "unknown")})
    with pytest.raises(ValueError, match="Invalid model name: UnknownLID"):
        SelectAudioLanguageStage().process(task)
    assert "lid" not in task.data


@pytest.mark.parametrize(
    "raw",
    [
        "not a mapping",
        {"language": "en", "confidence": 0.5},
        {"language": "en", "confidence": float("nan"), "tag": "tertiary"},
    ],
)
def test_invalid_component_results_are_rejected(raw: object) -> None:
    with pytest.raises(ValueError, match="Invalid LID result"):
        SelectAudioLanguageStage().process(_task({_WHISPER: raw}))


def test_process_batch_selects_each_task_independently() -> None:
    accepted, rejected = SelectAudioLanguageStage().process_batch(
        [_task(_full_results(whisper="en")), _task(_full_results(primary="hi", canary="ta", whisper="hi"))]
    )
    assert accepted.data["source_lang"] == "en"
    assert rejected.data["source_lang"] == "skipped"


def test_process_batch_routes_a_task_without_predictions() -> None:
    (task,) = SelectAudioLanguageStage().process_batch([_task(_skipme="rejected upstream")])

    assert task.data["source_lang"] == ""
    assert task.data["_skipme"] == "rejected upstream"


def test_rejection_preserves_an_audio_preparation_root_cause() -> None:
    task = _task(
        {
            _SPEECHBRAIN: _prediction("", 0.0, "primary"),
        },
        _skipme="audio_load_error",
    )

    result = SelectAudioLanguageStage().process_batch([task])[0]

    assert result.data["source_lang"] == ""
    assert result.data["_skipme"] == "audio_load_error"
    assert result.data["additional_notes"]["SelectBestLIDPrediction"] == (
        "skipped (missing or empty whisper langID prediction)"
    )


def test_selector_declares_its_io_contract_and_is_publicly_exported() -> None:
    stage = SelectAudioLanguageStage(results_key="predictions", output_key="language", notes_key="audit")
    assert stage.inputs() == ([], [])
    assert stage.outputs() == ([], ["language", "audit", "_skipme"])
    assert stage.resources.gpus == 0
    assert PublicSelectAudioLanguageStage is SelectAudioLanguageStage


def test_duplicate_configured_model_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        SelectAudioLanguageStage(ambernet_model_id=_SPEECHBRAIN)
