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

# ruff: noqa: INP001

from copy import deepcopy

import pytest

from nemo_curator.stages.audio.text_filtering import SelectBestPredictionStage
from nemo_curator.tasks import AudioTask

_JA_PRIMARY = "なんかおばあちゃんのレシピみたいなあのどんな思い出とか食べ物とかでどんな思い出とかあったりしますか"
_JA_FALLBACK = "なんかばあちゃんのレシピみたいなあのどんな思い出とか食べ物とかでどんな思い出とかあったりしますか。"


def test_uses_recovery_prediction_after_hallucination_recheck() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "loop loop loop",
            "fallback_model_prediction": "a valid transcript",
            "_skipme": "Hallucination",
            "additional_notes": {"recheck": "Recovered"},
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == "a valid transcript"
    assert task.data["best_prediction_source"] == "fallback"
    assert task.data["_skipme"] == "Hallucination"


def test_selects_supported_fallback_and_clears_primary_language_skip() -> None:
    """The primary adapter was skipped; a separate supported model produced the fallback."""
    task = AudioTask(
        data={
            "primary_model_prediction": "",
            "fallback_model_prediction": "bonjour",
            "_skipme": "language_not_supported",
            "additional_notes": {"primary_model_prediction": "lang_not_supported:fr"},
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == "bonjour"
    assert task.data["best_prediction_source"] == "fallback"
    assert task.data["_skipme"] == ""
    assert task.data["additional_notes"] == {
        "primary_model_prediction": "lang_not_supported:fr",
        "SelectBestPrediction": "used fallback (primary lang unsupported)",
    }


def test_unsupported_primary_without_fallback_matches_reference_fallthrough() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "",
            "fallback_model_prediction": "",
            "_skipme": "language_not_supported",
            "additional_notes": {"primary_model_prediction": "lang_not_supported:fr"},
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == ""
    assert task.data["best_prediction_source"] == "primary"
    assert task.data["_skipme"] == "language_not_supported"


def test_accepts_reference_asr_key_name() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "loop loop loop",
            "recovery_prediction": "a valid transcript",
            "_skipme": "Hallucination",
            "additional_notes": {"recheck": "Recovered"},
        }
    )

    SelectBestPredictionStage(asr_text_key="recovery_prediction").process(task)

    assert task.data["best_prediction"] == "a valid transcript"
    assert task.data["best_prediction_source"] == "fallback"
    assert task.data["_skipme"] == "Hallucination"


def test_uses_reference_when_no_model_supports_language() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "",
            "fallback_model_prediction": "",
            "original": "dataset transcript",
            "additional_notes": {"primary_model_prediction": "lang_not_supported:xx"},
        }
    )
    stage = SelectBestPredictionStage(reference_text_key="original")

    stage.process(task)

    assert task.data["best_prediction"] == "dataset transcript"
    assert task.data["best_prediction_source"] == "ground_truth"


def test_marks_sample_unsupported_when_neither_model_supports_language() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "",
            "fallback_model_prediction": "",
            "additional_notes": {
                "primary_model_prediction": "lang_not_supported:xx",
                "fallback_model_prediction": "lang_not_supported:xx",
            },
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == ""
    assert task.data["best_prediction_source"] == "none"
    assert task.data["_skipme"] == "not_supported"


def test_uses_reference_to_recover_a_hallucinated_primary() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "loop loop loop",
            "original": "dataset transcript",
            "_skipme": "Hallucination",
        }
    )
    stage = SelectBestPredictionStage(
        reference_text_key="original",
        use_reference_on_hallucination=True,
    )

    stage.process(task)

    assert task.data["best_prediction"] == "dataset transcript"
    assert task.data["best_prediction_source"] == "reference"
    assert task.data["_skipme"] == ""


def test_forces_reference_without_consulting_model_results() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "primary transcript",
            "fallback_model_prediction": "fallback transcript",
            "original": "dataset transcript",
            "_skipme": "Hallucination",
            "additional_notes": {"recheck": "Recovered"},
        }
    )
    stage = SelectBestPredictionStage(reference_text_key="original", force_reference=True)

    stage.process(task)

    assert task.data["best_prediction"] == "dataset transcript"
    assert task.data["best_prediction_source"] == "ground_truth"
    assert task.data["_skipme"] == ""
    assert task.data["additional_notes"]["SelectBestPrediction"] == "forced:ground_truth"


def test_forced_reference_preserves_an_intentionally_empty_reference() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "primary transcript",
            "original": "   ",
            "_skipme": "Hallucination",
        }
    )
    stage = SelectBestPredictionStage(reference_text_key="original", force_reference=True)

    stage.process(task)

    assert task.data["best_prediction"] == ""
    assert task.data["best_prediction_source"] == "ground_truth"
    assert task.data["_skipme"] == ""


def test_uses_reference_for_short_qwen_omni_audio() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "hallucinated transcript",
            "original": "dataset transcript",
            "duration": "0.25",
            "_skipme": "Hallucination",
        }
    )
    stage = SelectBestPredictionStage(reference_text_key="original", primary_model_type="qwen_omni")

    stage.process(task)

    assert task.data["best_prediction"] == "dataset transcript"
    assert task.data["best_prediction_source"] == "ground_truth"
    assert task.data["_skipme"] == ""
    assert task.data["additional_notes"]["SelectBestPrediction"] == "Ground Truth (short audio 0.25s < 1.0s)"


def test_short_audio_reference_requires_qwen_omni_and_valid_duration() -> None:
    cases = [
        ("parakeet", 0.25),
        ("qwen_omni", None),
        ("qwen_omni", "invalid"),
        ("qwen_omni", 0.0),
        ("qwen_omni", -0.25),
        ("qwen_omni", 1.0),
    ]

    for primary_model_type, duration in cases:
        task = AudioTask(
            data={
                "primary_model_prediction": "primary transcript",
                "original": "dataset transcript",
                "duration": duration,
            }
        )
        stage = SelectBestPredictionStage(
            reference_text_key="original",
            primary_model_type=primary_model_type,
        )

        stage.process(task)

        assert task.data["best_prediction"] == "primary transcript", (primary_model_type, duration)
        assert task.data["best_prediction_source"] == "primary", (primary_model_type, duration)


def test_short_audio_reference_can_be_disabled() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "primary transcript",
            "original": "dataset transcript",
            "duration": 0.25,
        }
    )
    stage = SelectBestPredictionStage(
        reference_text_key="original",
        primary_model_type="qwen_omni",
        use_ground_truth_for_short_audio=False,
    )

    stage.process(task)

    assert task.data["best_prediction"] == "primary transcript"
    assert task.data["best_prediction_source"] == "primary"


def test_cross_model_agreement_recovers_primary() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "Hello, world!",
            "fallback_model_prediction": "hello world",
            "_skipme": "Hallucination",
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == "Hello, world!"
    assert task.data["best_prediction_source"] == "primary"
    assert task.data["_skipme"] == ""
    assert task.data["primary_fallback_agreement_wer"] == 0.0
    assert task.data["primary_fallback_agreement_metric"] == "wer"


def test_cross_model_agreement_uses_reference_wer_rounding() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "one two three",
            "fallback_model_prediction": "one two",
            "_skipme": "Hallucination",
        }
    )
    stage = SelectBestPredictionStage(min_agreement_pct=66.67)

    stage.process(task)

    assert task.data["primary_fallback_agreement_wer"] == 33.33
    assert task.data["primary_fallback_agreement_metric"] == "wer"
    assert task.data["best_prediction"] == "one two three"
    assert task.data["_skipme"] == ""


def test_cross_model_disagreement_preserves_hallucination_skip() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "one two three",
            "fallback_model_prediction": "completely different transcript",
            "_skipme": "Hallucination",
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == "one two three"
    assert task.data["best_prediction_source"] == "primary"
    assert task.data["_skipme"] == "Hallucination"
    assert task.data["primary_fallback_agreement_wer"] > 20.0
    assert task.data["primary_fallback_agreement_metric"] == "wer"


def test_near_identical_japanese_predictions_are_recovered_with_cer() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": _JA_PRIMARY,
            "fallback_model_prediction": _JA_FALLBACK,
            "_skipme": "Hallucination:WhisperHallucination",
            "source_lang": "ja",
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == _JA_PRIMARY
    assert task.data["_skipme"] == ""
    assert task.data["primary_fallback_agreement_metric"] == "cer"
    assert task.data["primary_fallback_agreement_wer"] < 20.0


@pytest.mark.parametrize("language", ["ja", "zh", "th", "zh-TW", "yue"])
def test_no_space_languages_use_cer(language: str) -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "这是一个测试句子用来检查协议",
            "fallback_model_prediction": "这是一个测试句子用来检查协义",
            "_skipme": "Hallucination:WhisperHallucination",
            "source_lang": language,
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["primary_fallback_agreement_metric"] == "cer"
    assert task.data["_skipme"] == ""


@pytest.mark.parametrize("language", ["en", "de", "es", "ko", "vi"])
def test_space_separated_languages_use_wer(language: str) -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "the cat sat on the mat",
            "fallback_model_prediction": "the cat sat on the mat",
            "_skipme": "Hallucination:WhisperHallucination",
            "source_lang": language,
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["primary_fallback_agreement_metric"] == "wer"
    assert task.data["_skipme"] == ""


def test_japanese_genuine_disagreement_remains_flagged() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "あなたのおすすめの映画は何ですか",
            "fallback_model_prediction": "今日はとても良い天気ですね",
            "_skipme": "Hallucination:WhisperHallucination",
            "source_lang": "ja",
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["_skipme"].startswith("Hallucination")
    assert task.data["primary_fallback_agreement_metric"] == "cer"
    assert task.data["primary_fallback_agreement_wer"] > 20.0


def test_missing_language_defaults_to_wer() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "hello world",
            "fallback_model_prediction": "hello world",
            "_skipme": "Hallucination:WhisperHallucination",
        }
    )

    SelectBestPredictionStage().process(task)

    assert task.data["primary_fallback_agreement_metric"] == "wer"
    assert task.data["_skipme"] == ""


def test_custom_language_and_metric_keys_are_supported() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "日本語の文字列",
            "fallback_model_prediction": "日本語の文宇列",
            "_skipme": "Hallucination",
            "lang": "Japanese",
        }
    )

    SelectBestPredictionStage(language_key="lang", metric_key="agreement_metric").process(task)

    assert task.data["agreement_metric"] == "cer"
    assert "primary_fallback_agreement_metric" not in task.data


def test_scans_all_recovery_notes_like_reference() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "primary transcript",
            "fallback_model_prediction": "fallback transcript",
            "additional_notes": {
                "earlier_stage": "Recovered",
                "fallback_recheck": "passed",
            },
        }
    )
    SelectBestPredictionStage().process(task)

    assert task.data["best_prediction"] == "fallback transcript"
    assert task.data["best_prediction_source"] == "fallback"


def test_rerunning_cross_model_agreement_scans_the_prior_recovery_note() -> None:
    task = AudioTask(
        data={
            "primary_model_prediction": "Hello, world!",
            "fallback_model_prediction": "hello world",
            "_skipme": "Hallucination",
        }
    )
    stage = SelectBestPredictionStage()

    stage.process(task)
    stage.process(task)

    assert task.data["best_prediction"] == "hello world"
    assert task.data["best_prediction_source"] == "fallback"
    assert task.data["_skipme"] == ""
    assert "primary_fallback_agreement_wer" not in task.data
    assert "primary_fallback_agreement_metric" not in task.data
    assert task.data["additional_notes"]["SelectBestPrediction"] == "used fallback"


def test_declares_every_mutated_output_key() -> None:
    stage = SelectBestPredictionStage()

    assert stage.outputs() == (
        [],
        [
            "best_prediction",
            "best_prediction_source",
            "_skipme",
            "primary_fallback_agreement_wer",
            "primary_fallback_agreement_metric",
            "additional_notes",
        ],
    )


def test_keeps_primary_by_default() -> None:
    tasks = [
        AudioTask(data={"primary_model_prediction": "one"}),
        AudioTask(data={"primary_model_prediction": "two"}),
    ]

    SelectBestPredictionStage().process_batch(tasks)

    assert [task.data["best_prediction"] for task in tasks] == ["one", "two"]


def test_reference_decision_parity_precedence_and_fallthrough() -> None:
    """Golden decisions from reference selector commit 8469455fbd18de74928f21e1cc241358fc8d9e08."""
    cases = [
        (
            "short audio precedes force and recovery",
            {
                "primary_model_prediction": "primary transcript",
                "fallback_model_prediction": "fallback transcript",
                "reference": "ground truth",
                "duration": 0.25,
                "_skipme": "Hallucination",
                "additional_notes": {"fallback_check": "Recovered"},
            },
            {
                "reference_text_key": "reference",
                "primary_model_type": "qwen_omni",
                "force_reference": True,
            },
            (
                "ground truth",
                "ground_truth",
                "",
                None,
                "Ground Truth (short audio 0.25s < 1.0s)",
            ),
        ),
        (
            "ground truth precedes both-models-unsupported",
            {
                "primary_model_prediction": "",
                "fallback_model_prediction": "",
                "reference": "ground truth",
                "_skipme": "language_not_supported",
                "additional_notes": {
                    "primary_model_prediction": "lang_not_supported:xx",
                    "fallback_model_prediction": "lang_not_supported:xx",
                },
            },
            {"reference_text_key": "reference"},
            ("ground truth", "ground_truth", "", None, "Ground Truth"),
        ),
        (
            "recovery alias precedes reference and agreement",
            {
                "primary_model_prediction": "selected fallback",
                "fallback_model_prediction": "ignored fallback",
                "asr_prediction": "selected fallback",
                "reference": "ground truth",
                "_skipme": "Hallucination",
                "additional_notes": {"fallback_check": "Recovered:ASR"},
            },
            {
                "asr_text_key": "asr_prediction",
                "reference_text_key": "reference",
                "use_reference_on_hallucination": True,
            },
            ("selected fallback", "fallback", "Hallucination", None, "used fallback"),
        ),
        (
            "reference recovery precedes agreement",
            {
                "primary_model_prediction": "Hello, world!",
                "fallback_model_prediction": "hello world",
                "reference": "ground truth",
                "_skipme": "Hallucination",
            },
            {
                "reference_text_key": "reference",
                "use_reference_on_hallucination": True,
            },
            (
                "ground truth",
                "reference",
                "",
                None,
                "recovered:reference_text (hallucination_detected, fallback=reference)",
            ),
        ),
        (
            "empty reference falls through short and reference recovery",
            {
                "primary_model_prediction": "Hello, world!",
                "fallback_model_prediction": "hello world",
                "reference": "   ",
                "duration": 0.25,
                "_skipme": "Hallucination",
            },
            {
                "reference_text_key": "reference",
                "primary_model_type": "qwen_omni",
                "use_reference_on_hallucination": True,
            },
            (
                "Hello, world!",
                "primary",
                "",
                0.0,
                "recovered:cross_model_agreement (wer=0.0%)",
            ),
        ),
        (
            "fallback is ignored without a selection signal",
            {
                "primary_model_prediction": "primary transcript",
                "fallback_model_prediction": "fallback transcript",
            },
            {},
            ("primary transcript", "primary", None, None, "used primary"),
        ),
    ]

    for name, data, stage_kwargs, expected in cases:
        task = AudioTask(data=deepcopy(data))
        stage = SelectBestPredictionStage(
            agreement_wer_key="omni_asr_agreement_wer",
            **stage_kwargs,
        )

        stage.process(task)

        expected_text, expected_source, expected_skip, expected_wer, expected_note = expected
        assert task.data["best_prediction"] == expected_text, name
        assert task.data["best_prediction_source"] == expected_source, name
        if expected_skip is None:
            assert "_skipme" not in task.data, name
        else:
            assert task.data["_skipme"] == expected_skip, name
        if expected_wer is None:
            assert "omni_asr_agreement_wer" not in task.data, name
        else:
            assert task.data["omni_asr_agreement_wer"] == expected_wer, name
        assert task.data["additional_notes"]["SelectBestPrediction"] == expected_note, name
