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


def test_fallback_uses_primary() -> None:
    stage = SelectBestPredictionStage()
    task = AudioTask(data={
        "qwen3_prediction_s1": "primary text",
        "qwen3_asr_prediction": "",
        "additional_notes": "",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["best_prediction"] == "primary text"


def test_asr_recovery() -> None:
    stage = SelectBestPredictionStage()
    task = AudioTask(data={
        "qwen3_prediction_s1": "primary text",
        "qwen3_asr_prediction": "asr text",
        "additional_notes": "Recovered:SomeReason",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["best_prediction"] == "asr text"


def test_cross_model_agreement_recovery() -> None:
    stage = SelectBestPredictionStage(min_agreement_pct=80.0)
    task = AudioTask(data={
        "qwen3_prediction_s1": "hello world",
        "qwen3_asr_prediction": "hello world",
        "additional_notes": "",
        "_skip_me": "Hallucination",
    })
    result = stage.process(task)
    assert result.data["best_prediction"] == "hello world"
    assert result.data["_skip_me"] == ""
    assert "CrossModelAgreement" in result.data["additional_notes"]
    assert result.data["omni_asr_agreement_wer"] == 0.0


def test_cross_model_disagreement_no_recovery() -> None:
    stage = SelectBestPredictionStage(min_agreement_pct=80.0)
    task = AudioTask(data={
        "qwen3_prediction_s1": "hello world foo bar",
        "qwen3_asr_prediction": "completely different text here",
        "additional_notes": "",
        "_skip_me": "Hallucination",
    })
    result = stage.process(task)
    assert result.data["best_prediction"] == "hello world foo bar"
    assert result.data["_skip_me"] == "Hallucination"


def test_no_asr_pred_uses_primary() -> None:
    stage = SelectBestPredictionStage()
    task = AudioTask(data={
        "qwen3_prediction_s1": "primary",
        "qwen3_asr_prediction": "",
        "additional_notes": "Recovered:Something",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["best_prediction"] == "primary"


def test_empty_batch() -> None:
    stage = SelectBestPredictionStage()
    assert stage.process_batch([]) == []


def test_process_batch() -> None:
    stage = SelectBestPredictionStage()
    tasks = [
        AudioTask(data={
            "qwen3_prediction_s1": "text1",
            "qwen3_asr_prediction": "",
            "additional_notes": "",
            "_skip_me": "",
        }),
        AudioTask(data={
            "qwen3_prediction_s1": "text2",
            "qwen3_asr_prediction": "asr2",
            "additional_notes": "Recovered:X",
            "_skip_me": "",
        }),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["best_prediction"] == "text1"
    assert results[1].data["best_prediction"] == "asr2"
