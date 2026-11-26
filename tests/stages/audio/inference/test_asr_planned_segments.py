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

"""``ASRStage(segment_bounds_key=...)``: transcribe a row's planned chunks and join them in place."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

_SR = 16000
_DUR = 20.0


def _stage(respond: Callable[[dict[str, Any]], ASRResult], **kw: object) -> ASRStage:
    stage = ASRStage(
        adapter_target="nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter",
        model_id="mock/qwen-omni",
        audio_filepath_key="audio_filepath",
        max_audio_sec_per_actor=60.0,
        max_inference_duration_s=10.0,
        segment_bounds_key="cuts",
        **kw,
    )
    stage._adapter = MagicMock()
    stage._adapter.transcribe_batch.side_effect = lambda items: [respond(it) for it in items]
    # waveform value = time in seconds, so a chunk's first sample tells which chunk it is
    wav = (np.arange(int(_DUR * _SR)) / _SR).astype(np.float32)
    stage._load_audio = MagicMock(return_value=(wav, _SR))  # type: ignore[method-assign]
    return stage


def _task(cuts: list | None, **extra: object) -> AudioTask:
    data = {"audio_filepath": "/rec.wav", "source_lang": "en", **extra}
    if cuts is not None:
        data["cuts"] = cuts
    return AudioTask(task_id="rec", dataset_name="test", data=data)


def _named(item: dict[str, Any]) -> ASRResult:
    start = round(float(item["waveform"][0]), 1)
    return ASRResult(text=f"t{start:g}", extras={"finish_reason": "stop", "output_tokens": 5})


CUTS = [[0.0, 6.0], [6.0, 12.0, "silence"], [12.0, 20.0]]


def test_planned_chunks_are_transcribed_and_joined_in_order() -> None:
    stage = _stage(_named)
    (task,) = stage.process_batch([_task(CUTS)])
    assert task.data["pred_text"] == "t0 t12"
    assert "_skipme" not in task.data
    assert "additional_notes" not in task.data
    items = stage._adapter.transcribe_batch.call_args.args[0]
    assert [round(it["audio_seconds"], 3) for it in items] == [6.0, 8.0]  # silence never sent


def test_extras_key_gets_per_chunk_token_counts() -> None:
    (task,) = _stage(_named, extras_key="asr_extras").process_batch([_task(CUTS)])
    assert task.data["asr_extras"] == {"chunk_output_tokens": [5, 5]}


def test_truncated_chunk_skips_whole_row() -> None:
    def respond(item: dict[str, Any]) -> ASRResult:
        r = _named(item)
        if r.text == "t12":
            r.extras["finish_reason"] = "length"
        return r

    (task,) = _stage(respond).process_batch([_task(CUTS)])
    assert task.data["_skipme"] == "asr_output_truncated"
    assert task.data["pred_text"] == ""
    assert "chunk 2/2" in task.data["additional_notes"]["ASR_inference"]


def test_truncation_check_can_be_disabled() -> None:
    def respond(item: dict[str, Any]) -> ASRResult:
        r = _named(item)
        r.extras["finish_reason"] = "length"
        return r

    (task,) = _stage(respond, skip_on_truncated_output=False).process_batch([_task(CUTS)])
    assert task.data["pred_text"] == "t0 t12"


def test_empty_chunk_output_is_noted_not_skipped() -> None:
    def respond(item: dict[str, Any]) -> ASRResult:
        r = _named(item)
        if r.text == "t0":
            return ASRResult(text="", skipped=True, extras={"finish_reason": "stop", "output_tokens": 1})
        return r

    (task,) = _stage(respond).process_batch([_task(CUTS)])
    assert task.data["pred_text"] == "t12"
    assert "_skipme" not in task.data
    assert "empty chunks [1]" in task.data["additional_notes"]["ASR_inference"]


def test_failed_chunk_skips_whole_row() -> None:
    def respond(item: dict[str, Any]) -> ASRResult:
        r = _named(item)
        return ASRResult(text="", skipped=True) if r.text == "t12" else r

    (task,) = _stage(respond).process_batch([_task(CUTS)])
    assert task.data["_skipme"]
    assert task.data["pred_text"] == ""


def test_high_token_rate_is_noted() -> None:
    def respond(item: dict[str, Any]) -> ASRResult:
        r = _named(item)
        r.extras["output_tokens"] = 100  # 100 tokens over 6-8 s
        return r

    (task,) = _stage(respond).process_batch([_task(CUTS)])
    assert task.data["pred_text"] == "t0 t12"
    assert "possible repetition" in task.data["additional_notes"]["ASR_inference"]


@pytest.mark.parametrize(
    ("cuts", "reason"),
    [
        (None, "no_cut_plan"),
        ([], "no_cut_plan"),
        ([[0.0, 20.0, "silence"]], "no_speech"),
        ([[0.0, 15.0], [15.0, 20.0]], "cut_exceeds_max_inference_duration"),
        ([[0.0, 5.0, "music"]], "invalid_cut_plan"),
    ],
)
def test_unusable_plan_marks_row(cuts: list | None, reason: str) -> None:
    stage = _stage(_named)
    (task,) = stage.process_batch([_task(cuts)])
    assert task.data["_skipme"] == reason
    stage._adapter.transcribe_batch.assert_not_called()


def test_existing_skip_reason_is_preserved_when_configured() -> None:
    (task,) = _stage(_named, preserve_existing_skip=True).process_batch(
        [_task([], _skipme="LongformCutPlanner: missing diarization")]
    )
    assert task.data["_skipme"] == "LongformCutPlanner: missing diarization"


def test_default_mode_is_unchanged_without_segment_bounds_key() -> None:
    stage = _stage(_named)
    stage.segment_bounds_key = None
    (task,) = stage.process_batch([_task(CUTS)])
    # blind 10 s splits: 0-10, 10-20
    assert task.data["pred_text"] == "t0 t10"


def test_contract_reads_cut_plan() -> None:
    contract = _stage(_named).describe()
    assert "cuts" in contract.reads.data_keys


def test_segment_bounds_key_must_not_be_an_output() -> None:
    with pytest.raises(ValueError, match="segment_bounds_key"):
        ASRStage(
            adapter_target="x.Y",
            model_id="m",
            max_audio_sec_per_actor=60.0,
            max_inference_duration_s=10.0,
            segment_bounds_key="pred_text",
        )
