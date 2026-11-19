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

"""InferenceSortformerStage driven by PyAnnoteDiarizationAdapter (fake pyannote pipeline)."""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003

import pytest

from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from tests.stages.audio.inference import review_helpers as rh

hf_token = os.getenv("HF_TOKEN")
_PYANNOTE = "nemo_curator.models.audio.speaker_diarization.pyannote.PyAnnoteDiarizationAdapter"


def test_resident_count_output_preserves_audio_for_asr(monkeypatch: pytest.MonkeyPatch) -> None:
    stage, _ = rh._make_stage("pyannote", monkeypatch, input_residency="waveform")
    waveform = rh.np.linspace(-0.2, 0.2, 10, dtype=rh.np.float32)[None, :]
    result = stage.process_batch([AudioTask(data={"waveform": waveform, "sample_rate": rh._SAMPLE_RATE})])[0]
    assert result.data["num_speakers"] == 1
    assert result.data["overlap_segments"] == []
    assert result.data["sample_rate"] == rh._SAMPLE_RATE
    rh.np.testing.assert_array_equal(result.data["waveform"], waveform)
    asr = rh.ASRStage(
        adapter_target=rh._ASR_TARGET,
        model_id="mock/model",
        max_audio_sec_per_actor=2400.0,
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        target_sample_rate=rh._SAMPLE_RATE,
    )
    asr._adapter = rh.MagicMock()
    asr._adapter.transcribe_batch.return_value = [rh.ASRResult(text="hello")]
    asr.process_batch([result])
    rh.np.testing.assert_array_equal(asr._adapter.transcribe_batch.call_args.args[0][0]["waveform"], waveform[0])


@pytest.mark.parametrize("residency", ["waveform", "auto"])
def test_non_speaker_fill_uses_duration_field_then_audio_length(
    residency: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "fallback.wav"
    rh._write_audio(audio_path, rh.np.zeros((1, 2), dtype=rh.np.float32))
    stage, _seen = rh._make_stage("pyannote", monkeypatch, input_residency=residency)
    stage.non_speaker_segment_max_length = 0.5
    common = {"audio_filepath": str(audio_path)} if residency == "auto" else {}
    waveform = rh.np.ones((1, 12), dtype=rh.np.float32)
    result = stage.process_batch([AudioTask(data={**common, "waveform": waveform, "sample_rate": rh._SAMPLE_RATE})])[0]
    segments = result.data["segments"]
    assert max(segment["end"] for segment in segments) == pytest.approx(1.2)
    fillers = [segment for segment in segments if segment["speaker"] == "no-speaker"]
    assert [(f["start"], f["end"]) for f in fillers] == [(0.0, 0.2), (0.6, 0.7), (1.0, 1.2)]
    assert result.data["num_speakers"] == 1
    with_duration = stage.process_batch(
        [AudioTask(data={**common, "waveform": waveform, "sample_rate": rh._SAMPLE_RATE, "duration": 2.0})]
    )[0]
    assert with_duration.data["segments"][-1]["end"] == pytest.approx(2.0)


def test_rttm_contains_speech_and_overlap_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pyannote.core import Segment

    stage, _seen = rh._make_stage("pyannote", monkeypatch, rttm_out_dir=str(tmp_path / "rttm"))
    stage.non_speaker_segment_max_length = 40.0
    annotation = rh._FakeAnnotation(
        [{"start": 0.1, "end": 0.4, "speaker": "A"}, {"start": 0.4, "end": 0.9, "speaker": "B"}],
        overlap_segments=[Segment(0.5, 0.6)],
    )
    stage._adapter._pipeline = lambda _payload, **_kwargs: annotation
    audio_path = tmp_path / "durable.wav"
    rh._write_audio(audio_path, rh.np.ones((1, 12), dtype=rh.np.float32))
    task = stage.process_batch([AudioTask(task_id="t1", data={"audio_filepath": str(audio_path)})])[0]
    lines = (tmp_path / "rttm" / task.data["rttm_filepath"]).read_text().splitlines()
    assert [line.split()[7] for line in lines] == ["durable_A", "durable_B"]
    assert task.data["overlap_segments"] == [{"start": 0.4, "end": 0.9, "speaker": "durable_B"}]
    assert {segment["speaker"] for segment in task.data["segments"]} == {"durable_A", "no-speaker"}
    assert task.data["num_speakers"] == 2


def test_fanout_excludes_silence_and_includes_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pyannote.core import Segment

    audio_path = tmp_path / "source.wav"
    rh._write_audio(audio_path, rh.np.arange(50, dtype=rh.np.float32)[None, :])
    stage, _seen = rh._make_stage("pyannote", monkeypatch, fanout=True)
    stage.non_speaker_segment_max_length = 40.0
    annotation = rh._FakeAnnotation(
        [{"start": 1.0, "end": 2.0, "speaker": "speaker_0"}, {"start": 2.0, "end": 3.0, "speaker": "speaker_1"}],
        overlap_segments=[Segment(2.0, 3.0)],
    )
    stage._adapter._pipeline = lambda _payload, **_kwargs: annotation
    children = stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path)})])
    assert [(child.data["start"], child.data["end"], child.data["is_overlap"]) for child in children] == [
        (1.0, 2.0, False),
        (2.0, 3.0, True),
    ]
    assert all(child.data["speaker"] != "no-speaker" for child in children)
    assert all("overlap_segments" not in child.data for child in children)


@pytest.mark.gpu
@pytest.mark.skipif(not hf_token, reason="HF_TOKEN not set")
def test_process(wav_filepath: Path) -> None:
    """Diarize a real file with the tagging-tutorial configuration."""
    stage = InferenceSortformerStage(
        adapter_target=_PYANNOTE,
        model_id="pyannote/speaker-diarization-3.1",
        audio_filepath_key="resampled_audio_filepath",
        diar_segments_key="segments",
        overlap_segments_key="overlap_segments",
        num_speakers_key=None,
        global_speaker_ids=True,
        non_speaker_segment_max_length=40.0,
        resources=Resources(gpus=1),
        adapter_kwargs={"hf_token": hf_token},
    )
    stage.setup_on_node()
    stage.setup()
    task = AudioTask(data={"resampled_audio_filepath": str(wav_filepath), "audio_item_id": "id_1", "duration": 60.0})
    result = stage.process_batch([task])[0]
    segments = result.data["segments"]
    assert segments
    for segment in segments:
        assert segment["start"] < segment["end"]
        assert 0 <= segment["start"] <= 60.0
        assert 0 <= segment["end"] <= 60.0
