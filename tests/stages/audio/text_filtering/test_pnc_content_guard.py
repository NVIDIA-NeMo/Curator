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

from nemo_curator.stages.audio.text_filtering.pnc_content_guard import PnCContentGuardStage
from nemo_curator.tasks import AudioTask


def test_matching_text_passes() -> None:
    stage = PnCContentGuardStage()
    task = AudioTask(data={
        "text": "hello world",
        "pnc_text": "Hello, World!",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["pnc_text"] == "Hello, World!"
    assert result.data["rejected_pnc_text"] == ""


def test_mismatched_text_reverts() -> None:
    stage = PnCContentGuardStage()
    task = AudioTask(data={
        "text": "hello world",
        "pnc_text": "Goodbye Universe!",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["pnc_text"] == "hello world"
    assert result.data["rejected_pnc_text"] == "Goodbye Universe!"


def test_skipped_task() -> None:
    stage = PnCContentGuardStage()
    task = AudioTask(data={
        "text": "hello",
        "pnc_text": "different",
        "_skip_me": "Hallucination",
    })
    result = stage.process(task)
    assert result.data["pnc_text"] == ""
    assert result.data["rejected_pnc_text"] == ""


def test_empty_original_no_revert() -> None:
    stage = PnCContentGuardStage()
    task = AudioTask(data={
        "text": "",
        "pnc_text": "Something",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["rejected_pnc_text"] == ""


def test_empty_pnc_no_revert() -> None:
    stage = PnCContentGuardStage()
    task = AudioTask(data={
        "text": "hello",
        "pnc_text": "",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["rejected_pnc_text"] == ""


def test_unicode_punctuation_ignored() -> None:
    """Unicode punctuation differences should not trigger a revert."""
    stage = PnCContentGuardStage()
    task = AudioTask(data={
        "text": "it's fine",
        "pnc_text": "It\u2019s fine.",
        "_skip_me": "",
    })
    result = stage.process(task)
    assert result.data["pnc_text"] == "It\u2019s fine."
    assert result.data["rejected_pnc_text"] == ""


def test_process_batch() -> None:
    stage = PnCContentGuardStage()
    tasks = [
        AudioTask(data={"text": "abc", "pnc_text": "Abc!", "_skip_me": ""}),
        AudioTask(data={"text": "abc", "pnc_text": "XYZ", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["rejected_pnc_text"] == ""
    assert results[1].data["rejected_pnc_text"] == "XYZ"


def test_empty_batch() -> None:
    stage = PnCContentGuardStage()
    assert stage.process_batch([]) == []


def test_custom_keys() -> None:
    stage = PnCContentGuardStage(
        text_key="orig",
        pnc_text_key="restored",
        rejected_text_key="bad",
    )
    task = AudioTask(data={"orig": "hello", "restored": "WRONG WORDS", "_skip_me": ""})
    result = stage.process(task)
    assert result.data["restored"] == "hello"
    assert result.data["bad"] == "WRONG WORDS"
