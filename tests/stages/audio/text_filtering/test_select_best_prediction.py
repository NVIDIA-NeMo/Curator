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

from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.tasks import AudioTask


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
    assert task.data["_skipme"] == ""


def test_falls_back_when_primary_language_is_unsupported() -> None:
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


def test_marks_unsupported_primary_without_fallback_as_unusable() -> None:
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
    assert task.data["best_prediction_source"] == "none"
    assert task.data["_skipme"] == "not_supported"


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
    assert task.data["_skipme"] == ""


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


def test_recovery_note_key_ignores_unrelated_recovery_notes() -> None:
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
    stage = SelectBestPredictionStage(recovery_note_key="fallback_recheck")

    stage.process(task)

    assert task.data["best_prediction"] == "primary transcript"
    assert task.data["best_prediction_source"] == "primary"


def test_rerunning_cross_model_agreement_is_idempotent() -> None:
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

    assert task.data["best_prediction"] == "Hello, world!"
    assert task.data["best_prediction_source"] == "primary"
    assert task.data["_skipme"] == ""
    assert "primary_fallback_agreement_wer" not in task.data
    assert task.data["additional_notes"]["SelectBestPrediction"] == "used primary"


def test_declares_every_mutated_output_key() -> None:
    stage = SelectBestPredictionStage()

    assert stage.outputs() == (
        [],
        [
            "best_prediction",
            "best_prediction_source",
            "_skipme",
            "primary_fallback_agreement_wer",
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
