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

"""ASRStage word timestamps and nested (split / segment) transcription."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import MergeAlignmentDiarizationStage
from nemo_curator.stages.audio.tagging.split import (
    JoinSplitAudioMetadataStage,
    SplitASRAlignJoinStage,
    SplitLongAudioStage,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

_SR = 16_000


class _WordAdapter:
    """Return one word spanning each input, plus the input's duration in the text."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        self.calls.append(items)
        results = []
        for item in items:
            seconds = round(len(item["waveform"]) / item["sample_rate"], 3)
            word = {"word": "hello", "start": 0.0, "end": seconds, "confidence": 0.9}
            results.append(ASRResult(text=f"hello {seconds}", extras={"words": [word]}))
        return results


def _stage(**kwargs: Any) -> ASRStage:  # noqa: ANN401
    options = {"max_audio_sec_per_actor": 100.0, "max_inference_duration_s": 100.0, **kwargs}
    stage = ASRStage(
        adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
        model_id="m",
        resources=Resources(cpus=1.0),
        **options,
    )
    stage._adapter = _WordAdapter()
    return stage


def _wav(path: Path, seconds: float) -> str:
    sf.write(path, np.zeros(int(seconds * _SR), dtype=np.float32), _SR)
    return str(path)


def test_words_are_written_and_long_audio_chunks_are_shifted() -> None:
    stage = _stage(
        waveform_key="waveform", words_key="words", max_audio_sec_per_actor=4.0, max_inference_duration_s=2.0
    )
    task = AudioTask(data={"waveform": np.zeros(3 * _SR, dtype=np.float32), "sampling_rate": _SR})

    stage.process_batch([task])

    assert task.data["pred_text"] == "hello 2.0 hello 1.0"
    assert [(w["start"], w["end"]) for w in task.data["words"]] == [(0.0, 2.0), (2.0, 3.0)]


def test_nested_file_mode_transcribes_each_split_then_joins(tmp_path: Path) -> None:
    stage = _stage(pred_text_key="text", words_key="alignment", nested_key="split_metadata")
    joiner, merger = JoinSplitAudioMetadataStage(), MergeAlignmentDiarizationStage()
    report = validate_pipeline(
        [SplitLongAudioStage(), stage, joiner, merger],
        initial_keys={"duration", "segments", "resampled_audio_filepath"},
        initial_task_type="AudioTask",
    )
    assert report.ok
    task = AudioTask(
        data={
            "duration": 3.0,
            "segments": [{"speaker": "s1", "start": 0.0, "end": 3.0}],
            "split_filepaths": ["a", "b"],
            "split_metadata": [
                {"resampled_audio_filepath": _wav(tmp_path / "a.wav", 2.0)},
                {"resampled_audio_filepath": _wav(tmp_path / "b.wav", 1.0)},
            ],
            "split_offsets": [0.0, 2.0],
        }
    )

    for processing_stage in (stage, joiner, merger):
        result = (
            processing_stage.process_batch([task]) if processing_stage is stage else processing_stage.process(task)
        )
        assert result

    assert task.data["text"] == "hello 2.0 hello 1.0"
    assert [(w["start"], w["end"]) for w in task.data["alignment"]] == [(0.0, 2.0), (2.0, 3.0)]
    assert "pred_text" not in task.data
    assert "split_metadata" not in task.data


def test_nested_slice_mode_skips_short_segments_and_offsets_words() -> None:
    stage = _stage(
        waveform_key="waveform",
        pred_text_key="text_2",
        words_key="words",
        nested_key="segments",
        nested_audio="slice",
        min_nested_duration_s=1.0,
    )
    segments = [{"start": 1.0, "end": 3.0}, {"start": 3.0, "end": 3.5}]
    task = AudioTask(
        data={"waveform": np.zeros(4 * _SR, dtype=np.float32), "sampling_rate": _SR, "segments": segments}
    )

    stage.process_batch([task])

    assert segments[0]["text_2"] == "hello 2.0"
    assert segments[0]["words"] == [{"word": "hello", "start": 1.0, "end": 3.0, "confidence": 0.9}]
    assert "text_2" not in segments[1]
    assert "text_2" not in task.data


def test_nested_contract_declares_iteration_and_segment_writes() -> None:
    contract = _stage(nested_key="segments", nested_audio="slice", pred_text_key="t", words_key="w").describe()

    assert contract.cardinality == "1:1 nested-list"
    assert contract.iteration_key == "segments"
    assert contract.reads.data_keys == ["segments", "resampled_audio_filepath"]
    assert contract.conditional_writes[0].writes.segment_data_keys == ["t", "w"]


def test_split_asr_align_join_decomposes_to_adapter_asr() -> None:
    _split, asr, _join = SplitASRAlignJoinStage(alignment_key="align").decompose()

    assert isinstance(asr, ASRStage)
    assert asr.nested_key == "split_metadata"
    assert asr.words_key == "align"
    assert asr.adapter_kwargs == {"enable_local_attention": True, "decoder_type": "rnnt", "timestamps": True}
