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

from nemo_curator.stages.audio.text_filtering.disfluency_wer_guard import DisfluencyWerGuardStage
from nemo_curator.tasks import AudioTask


def test_low_wer_passes_through() -> None:
    stage = DisfluencyWerGuardStage(max_wer_pct=50.0)
    task = AudioTask(data={
        "qwen3_prediction_s1": "hello world",
        "qwen3_prediction_s2": "hello world",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["qwen3_prediction_s2"] == "hello world"
    assert result.data["disfluency_wer"] == 0.0


def test_high_wer_reverts_to_ref() -> None:
    stage = DisfluencyWerGuardStage(max_wer_pct=50.0)
    task = AudioTask(data={
        "qwen3_prediction_s1": "the quick brown fox jumps over the lazy dog",
        "qwen3_prediction_s2": "completely different text here now",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["qwen3_prediction_s2"] == "the quick brown fox jumps over the lazy dog"
    assert result.data["disfluency_wer"] > 50.0


def test_skipped_task_gets_default_wer() -> None:
    stage = DisfluencyWerGuardStage()
    task = AudioTask(data={
        "qwen3_prediction_s1": "hello",
        "qwen3_prediction_s2": "world",
        "_skip_me": "Hallucination",
    })
    result = stage.process(task)
    assert result.data["disfluency_wer"] == -1.0


def test_empty_ref_gets_negative_wer() -> None:
    stage = DisfluencyWerGuardStage()
    task = AudioTask(data={
        "qwen3_prediction_s1": "",
        "qwen3_prediction_s2": "something",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["disfluency_wer"] == -1.0


def test_empty_hyp_gets_negative_wer() -> None:
    stage = DisfluencyWerGuardStage()
    task = AudioTask(data={
        "qwen3_prediction_s1": "something",
        "qwen3_prediction_s2": "",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["disfluency_wer"] == -1.0


def test_process_batch() -> None:
    stage = DisfluencyWerGuardStage(max_wer_pct=50.0)
    tasks = [
        AudioTask(data={"qwen3_prediction_s1": "abc", "qwen3_prediction_s2": "abc", "_skip_me": ""}),
        AudioTask(data={"qwen3_prediction_s1": "abc", "qwen3_prediction_s2": "xyz", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["disfluency_wer"] == 0.0
    assert results[1].data["qwen3_prediction_s2"] == "abc"


def test_empty_batch() -> None:
    stage = DisfluencyWerGuardStage()
    assert stage.process_batch([]) == []


def test_custom_keys() -> None:
    stage = DisfluencyWerGuardStage(
        ref_text_key="ref",
        hyp_text_key="hyp",
        wer_key="my_wer",
    )
    task = AudioTask(data={"ref": "hello world", "hyp": "hello world", "_skip_me": ""})
    result = stage.process(task)
    assert result.data["my_wer"] == 0.0
