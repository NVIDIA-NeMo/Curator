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

import subprocess
from collections.abc import Callable
from math import isclose
from pathlib import Path

import pytest
import torch

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io.audio_file_reader import AudioFileReaderStage
from nemo_curator.tasks import AudioTask


def test_audio_file_reader_process(audio_task: Callable[..., AudioTask], audio_filepath: Path) -> None:
    stage = AudioFileReaderStage(target_sample_rate=16000, target_nchannels=1)
    stage.setup_on_node()
    task = audio_task(
        audio_filepath=str(audio_filepath),
        audio_item_id="id_1",
    )

    result = stage.process(task)

    out = result.data
    assert out.get("audio_filepath") == str(audio_filepath)
    assert out.get("sample_rate") == 16000
    assert out.get("is_mono") is True
    assert out["waveform"].shape[0] == 1
    assert out.get("num_samples") == out["waveform"].shape[-1]
    assert isclose(out.get("duration"), 60.0)


def test_audio_file_reader_process_batch_validates_audio_filepath() -> None:
    stage = AudioFileReaderStage(target_sample_rate=16000, target_nchannels=1)

    assert stage.inputs() == ([], ["audio_filepath"])
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"duration": 1.0})])


def test_audio_file_reader_skip_on_read_error(
    audio_task: Callable[..., AudioTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = AudioFileReaderStage(target_sample_rate=16000, target_nchannels=1)
    monkeypatch.setattr(
        stage,
        "_load_waveform",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode lost")),
    )
    task = audio_task(audio_filepath="/local/audio/missing.opus")

    result = stage.process(task)

    out = result.data
    assert out["_skip_me"] == "audio_read_error"
    assert "decode lost" in out["audio_read_error"]
    assert out["waveform"].shape == (1, 0)
    assert out["sample_rate"] == 16000
    assert out["duration"] == 0.0


def test_audio_file_reader_respects_custom_output_keys(
    audio_task: Callable[..., AudioTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = AudioFileReaderStage(
        waveform_key="custom_waveform",
        sample_rate_key="custom_sample_rate",
        num_samples_key="custom_num_samples",
    )
    monkeypatch.setattr(stage, "_load_waveform", lambda *_args, **_kwargs: (torch.zeros(1, 123), 16000))
    task = audio_task(audio_filepath="/local/audio/example.opus")

    result = stage.process(task)

    assert result.data["custom_waveform"].shape == (1, 123)
    assert result.data["custom_sample_rate"] == 16000
    assert result.data["custom_num_samples"] == 123
    assert "waveform" not in result.data
    assert "sample_rate" not in result.data
    assert "num_samples" not in result.data


def test_audio_file_reader_worker_specs() -> None:
    stage = AudioFileReaderStage(ray_num_workers=2)

    assert stage.num_workers() == 2
    assert stage.ray_stage_spec()[RayStageSpecKeys.IS_ACTOR_STAGE] is True

    stage = AudioFileReaderStage(xenna_num_workers_per_node=1)
    assert stage.num_workers() is None
    assert stage.xenna_stage_spec() == {"num_workers_per_node": 1}


def test_audio_file_reader_rejects_conflicting_xenna_worker_specs() -> None:
    with pytest.raises(ValueError, match="set at most one"):
        AudioFileReaderStage(xenna_num_workers=2, xenna_num_workers_per_node=1)


def test_audio_file_reader_rejects_remote_paths(
    audio_task: Callable[..., AudioTask],
) -> None:
    stage = AudioFileReaderStage(skip_on_read_error=True)
    task = audio_task(audio_filepath="s3://bucket/path/audio.opus")

    with pytest.raises(ValueError, match="only accepts local audio paths"):
        stage.process(task)


def test_audio_file_reader_uses_ffmpeg_seek_for_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    seen_cmd: list[str] = []

    def fake_run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
        seen_cmd[:] = cmd
        samples = torch.zeros(16000, dtype=torch.float32).numpy().tobytes()
        return subprocess.CompletedProcess(cmd, 0, stdout=samples, stderr=b"")

    stage = AudioFileReaderStage(target_sample_rate=16000, target_nchannels=1)
    monkeypatch.setattr(stage, "_run_ffmpeg", fake_run_ffmpeg)

    waveform, sample_rate = stage._load_waveform(
        str(audio_path),
        segment_start_s=12.5,
        segment_duration_s=30.0,
    )

    assert waveform.shape == (1, 16000)
    assert sample_rate == 16000
    assert seen_cmd[:6] == ["ffmpeg", "-v", "error", "-ss", "12.5", "-i"]
    assert seen_cmd[6] == str(audio_path)
    assert "-t" in seen_cmd
    assert seen_cmd[seen_cmd.index("-t") + 1] == "30"
