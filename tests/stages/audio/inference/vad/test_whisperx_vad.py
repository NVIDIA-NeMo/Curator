# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from tests.stages.audio.inference import review_helpers as rh


class TestWhisperXVADStage:
    @pytest.mark.gpu
    def test_process(self, wav_filepath: Path) -> None:
        stage = WhisperXVADStage(
            min_length=0.5,
            max_length=40.0,
            segments_key="vad_segments",
            resources=Resources(gpus=1),
        )
        stage.setup()

        entry = {
            "resampled_audio_filepath": str(wav_filepath),
            "duration": 60.0,
        }
        task = AudioTask(data=entry)
        result = stage.process(task)
        out = result.data
        assert "vad_segments" in out
        assert isinstance(out["vad_segments"], list)
        assert len(out["vad_segments"]) == 2


@rh.pytest.mark.parametrize("residency", ["waveform", "auto"])
def test_whisperx_resident_duration_controls_short_input_decisions(
    residency: str, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "fallback.wav"
    rh._write_audio(audio_path, rh.np.zeros((1, 2), dtype=rh.np.float32))
    stage, seen = rh._make_stage("whisperx", monkeypatch, input_residency=residency)
    stage.min_length = 0.5
    common = {"audio_filepath": str(audio_path)} if residency == "auto" else {}
    long_task = rh.AudioTask(
        data={
            **common,
            "waveform": rh.np.ones((1, 12), dtype=rh.np.float32),
            "sample_rate": rh._SAMPLE_RATE,
            "duration": 0.1,
        }
    )
    long_result = stage.process_batch([long_task])
    assert long_result == [long_task]
    assert long_task.data["vad_segments"]
    assert seen == [12]
    short_task = rh.AudioTask(
        data={
            **common,
            "waveform": rh.np.ones((1, 2), dtype=rh.np.float32),
            "sample_rate": rh._SAMPLE_RATE,
            "duration": 99.0,
        }
    )
    short_result = stage.process_batch([short_task])
    assert short_result == [short_task]
    assert short_task.data["vad_segments"] == []
    assert seen == [12], "the VAD model must not run for the selected 0.2-second waveform"


def test_whisperx_file_mode_keeps_manifest_duration_behavior(
    tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "long.wav"
    rh._write_audio(audio_path, rh.np.ones((1, 12), dtype=rh.np.float32))
    stage, seen = rh._make_stage("whisperx", monkeypatch, input_residency="file")
    stage.min_length = 0.5
    task = rh.AudioTask(data={"audio_filepath": str(audio_path), "duration": 0.1})
    result = stage.process_batch([task])
    assert result == [task]
    assert task.data["vad_segments"] == []
    assert seen == []
