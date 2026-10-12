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
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nemo_curator.stages.audio.tts.diarization import (
    DIARIZATION_FAILED,
    SegmentDiarizationStage,
    parse_diarization_output,
    summarize_diarization,
    union_duration,
)
from nemo_curator.tasks import AudioTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments


class FakeSortformer:
    """Returns canned "begin end speaker" lines; records the inputs it received."""

    def __init__(self, outputs: list[list[str]] | None = None, fail: bool = False) -> None:
        self.sortformer_modules = SimpleNamespace(checked=False)
        self.sortformer_modules._check_streaming_parameters = self._check
        self.outputs = outputs or []
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def _check(self) -> None:
        self.sortformer_modules.checked = True

    def eval(self) -> None:
        pass

    def diarize(self, audio: list[Any], **kwargs: Any) -> list[list[str]]:  # noqa: ANN401
        self.calls.append({"audio": audio, **kwargs})
        if self.fail:
            msg = "boom"
            raise RuntimeError(msg)
        return [self.outputs[i % len(self.outputs)] if self.outputs else [] for i in range(len(audio))]


def test_parse_and_union_duration() -> None:
    parsed = parse_diarization_output(["0.0 1.0 speaker_0", "0.5 2.0 speaker_1", ("3.0", "3.5", "speaker_0")])
    assert parsed[0] == {"start_time": 0.0, "end_time": 1.0, "speaker": "speaker_0"}
    assert union_duration((s["start_time"], s["end_time"]) for s in parsed) == 2.5
    assert union_duration([(1.0, 1.0), (2.0, 1.5)]) == 0.0
    with pytest.raises(ValueError, match="Unexpected"):
        parse_diarization_output(["0.0 1.0"])


def test_summarize_diarization() -> None:
    segs = parse_diarization_output(["0.0 1.0 speaker_0", "0.5 2.0 speaker_0"])
    summary = summarize_diarization(segs, 4.0)
    assert summary == {
        "diarized_segments": segs,
        "num_speakers": 1,
        "diarized_speech_duration": 2.0,
        "diarized_speech_rate": 0.5,
        "error": None,
    }
    assert summarize_diarization([], 2.0)["num_speakers"] == 0
    assert summarize_diarization(segs, 0.0)["diarized_speech_rate"] is None
    assert summarize_diarization(segs, None)["diarized_speech_rate"] is None


@pytest.mark.parametrize("input_mode", ["numpy", "wav"])
def test_stage_annotates_segments(tmp_path: Path, input_mode: str) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 2.0, 22050), ("b.wav", 1.0, 16000)])
    segments[1]["GranaryHifi"]["AudioMetadata"]["tar_member_offset"] = None
    model = FakeSortformer([["0.00 1.50 speaker_0", "1.20 1.80 speaker_1"]])
    stage = SegmentDiarizationStage(diar_model=model, input_mode=input_mode)
    stage.setup()
    assert model.sortformer_modules.checked
    assert model.sortformer_modules.chunk_len == 340
    assert model.sortformer_modules.spkcache_len == 188
    assert not hasattr(model.sortformer_modules, "chunk_left_context")

    task = AudioTask(data={"orig_sample_id": "v", "segments": segments})
    out = stage.process(task)
    first, second = (s["diarization"] for s in out.data["segments"])
    assert first["num_speakers"] == 2
    assert first["diarized_speech_duration"] == 1.8
    assert first["diarized_speech_rate"] == 0.9
    assert first["error"] is None
    assert second == {
        "diarized_segments": None,
        "num_speakers": None,
        "diarized_speech_duration": None,
        "diarized_speech_rate": None,
        "error": "missing_tar_offsets",
    }
    assert len(model.calls) == 1
    if input_mode == "numpy":
        assert model.calls[0]["sample_rate"] == 16000
        wave = model.calls[0]["audio"][0]
        assert isinstance(wave, np.ndarray)
        assert abs(wave.shape[0] - 32000) <= 1
    else:
        assert isinstance(model.calls[0]["audio"][0], str)
        assert not os.path.exists(model.calls[0]["audio"][0])


def test_batched_numpy_and_flat_row(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 1.0, 16000), ("b.wav", 1.0, 16000), ("c.wav", 1.0, 16000)])
    model = FakeSortformer([["0.0 0.5 speaker_0"]])
    stage = SegmentDiarizationStage(
        diar_model=model, input_mode="numpy", inference_batch_size=2, output_key="GranaryHifi.Diar"
    )
    stage.setup()
    stage.process(AudioTask(data={"segments": segments}))
    assert [len(c["audio"]) for c in model.calls] == [2, 1]
    assert all(s["GranaryHifi"]["Diar"]["diarized_speech_rate"] == 0.5 for s in segments)

    flat = copy.deepcopy(segments[0])
    out = stage.process(AudioTask(data=flat))
    assert out.data["GranaryHifi"]["Diar"]["num_speakers"] == 1


def test_model_failure_is_annotated(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 1.0, 16000)])
    stage = SegmentDiarizationStage(diar_model=FakeSortformer(fail=True))
    stage.setup()
    stage.process(AudioTask(data={"segments": segments}))
    assert segments[0]["diarization"]["error"] == DIARIZATION_FAILED


def test_invalid_params() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        SegmentDiarizationStage(input_mode="mp3")
    with pytest.raises(ValueError, match="inference_batch_size"):
        SegmentDiarizationStage(inference_batch_size=0)


@pytest.mark.gpu
def test_real_sortformer_smoke(tmp_path: Path) -> None:
    """Real streaming Sortformer on a synthetic two-tone clip; numpy and temp-wav inputs must agree."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nemo.collections.asr")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        stage = SegmentDiarizationStage(model_path=os.environ.get("SORTFORMER_NEMO"))
        stage.setup()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Sortformer weights unavailable: {exc}")
    clip = os.environ.get("DIAR_SMOKE_WAV")
    segments = (
        [{"audio_filepath": clip, "duration": float(__import__("soundfile").info(clip).duration)}]
        if clip
        else make_tar_segments(tmp_path, [("a.wav", 4.0, 16000)])
    )
    numpy_ann = stage.annotate_segments(copy.deepcopy(segments))
    stage.input_mode = "wav"
    wav_ann = stage.annotate_segments(copy.deepcopy(segments))
    for annotation in numpy_ann + wav_ann:
        assert annotation["error"] is None
        assert annotation["num_speakers"] >= 0
    assert numpy_ann[0]["num_speakers"] == wav_ann[0]["num_speakers"]
    assert numpy_ann[0]["diarized_speech_rate"] == pytest.approx(wav_ann[0]["diarized_speech_rate"], abs=0.05)


def test_default_input_mode_is_wav() -> None:
    assert SegmentDiarizationStage().input_mode == "wav"
