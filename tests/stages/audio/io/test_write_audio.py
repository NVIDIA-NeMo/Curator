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

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.io.write_audio import AudioSegmentWriterStage
from nemo_curator.tasks import AudioTask


def _segment(source: str = "/recordings/a.wav") -> AudioTask:
    waveform = torch.sin(torch.arange(16000) * (2 * np.pi * 440 / 16000)).unsqueeze(0) * 0.1
    return AudioTask(
        data={
            "audio_filepath": source,
            "original_file": source,
            "audio_item_id": "recording",
            "waveform": waveform,
            "sample_rate": 16000,
            "start_ms": 2000,
            "end_ms": 3000,
            "duration": 100.0,
            "resampled_audio_filepath": "/temporary/resampled.wav",
            "num_samples": 1600000,
            "is_mono": True,
            "diar_segments": [{"start": 0, "end": 100, "speaker": "speaker_0"}],
            "_sed_framewise": np.ones((2, 3)),
            "num_speakers": 1,
            "sed_events": [{"label": "Speech", "start_time": 0, "end_time": 1}],
            "source_lang": "skipped",
            "_skipme": "skipped due to disagreement between langID models.",
            "additional_notes": {"source_lid_confidence": 0.0},
        }
    )


def test_opus_round_trip_and_metadata(tmp_path: Path) -> None:
    stage = AudioSegmentWriterStage(output_dir=str(tmp_path))
    task = _segment()
    original = task.data["waveform"].numpy().flatten()
    output = stage.process(task)
    waveform, sample_rate = sf.read(output.data["audio_filepath"])
    assert sample_rate == 16000
    assert waveform.shape == (16000,)
    assert np.sqrt(np.mean((waveform - original) ** 2)) < 0.01
    assert output.data["duration"] == 1.0
    assert output.data["audio_item_id"] != "recording"
    assert output.data["original_file"] == "/recordings/a.wav"
    assert output.data["start_ms"] == 2000
    assert output.data["end_ms"] == 3000
    assert output.data["num_speakers"] == 1
    assert output.data["source_lang"] == "skipped"
    assert output.data["additional_notes"] == {"source_lid_confidence": 0.0}
    assert output.data["sed_events"][0]["label"] == "Speech"
    assert "_skipme" in output.data
    assert not set(stage._removed_keys()).intersection(output.data)
    json.dumps(output.data, allow_nan=False)
    assert_agent_ready(stage, fixture_factory=_segment)


def test_same_basename_and_retries_have_stable_distinct_paths(tmp_path: Path) -> None:
    stage = AudioSegmentWriterStage(output_dir=str(tmp_path))
    first = stage.process(_segment("/first/a.wav")).data["audio_filepath"]
    other = stage.process(_segment("/second/a.wav")).data["audio_filepath"]
    retry = stage.process(_segment("/first/a.wav")).data["audio_filepath"]
    assert first == retry
    assert first != other
    assert len(list((tmp_path / "audio").glob("*.opus"))) == 2


def test_sibling_segments_have_distinct_ids_stable_across_retries(tmp_path: Path) -> None:
    stage = AudioSegmentWriterStage(output_dir=str(tmp_path))
    first = stage.process(_segment()).data
    sibling_task = _segment()
    sibling_task.data.update(start_ms=3000, end_ms=4000)
    sibling = stage.process(sibling_task).data
    retry = stage.process(_segment()).data
    assert first["audio_item_id"] != sibling["audio_item_id"]
    assert first["audio_item_id"] == retry["audio_item_id"]
    assert first["original_file"] == sibling["original_file"] == "/recordings/a.wav"


def test_empty_vad_keeps_source_audit_without_audio(tmp_path: Path) -> None:
    stage = AudioSegmentWriterStage(output_dir=str(tmp_path))

    def fixture() -> AudioTask:
        return AudioTask(
            data={"audio_filepath": "/source.wav", "vad_empty": True, "duration": 10, "_skipme": "vad_empty"}
        )

    assert_agent_ready(stage, fixture_factory=fixture)
    output = stage.process(fixture())
    assert output.data["audio_filepath"] == "/source.wav"
    assert output.data["original_file"] == "/source.wav"
    assert output.data["duration"] == 10
    assert output.data["_skipme"] == "vad_empty"
    assert not list(tmp_path.rglob("*.opus"))


@pytest.mark.parametrize("failure", ["missing", "empty", "stereo", "nonfinite", "sample_rate", "read_error"])
def test_bad_audio_does_not_publish_output(tmp_path: Path, failure: str) -> None:
    stage = AudioSegmentWriterStage(output_dir=str(tmp_path))
    task = _segment()
    if failure == "missing":
        task.data.pop("waveform")
    elif failure == "empty":
        task.data["waveform"] = torch.zeros(0)
    elif failure == "stereo":
        task.data["waveform"] = torch.zeros(2, 16000)
    elif failure == "nonfinite":
        task.data["waveform"][0, 0] = float("nan")
    elif failure == "sample_rate":
        task.data["sample_rate"] = 48000
    else:
        task.data["read_error"] = True
    with pytest.raises((KeyError, ValueError)):
        stage.process(task)
    assert task.data["audio_filepath"] == "/recordings/a.wav"
    assert not list(tmp_path.rglob("*.opus"))


def test_encoder_failure_is_atomic(tmp_path: Path) -> None:
    stage = AudioSegmentWriterStage(output_dir=str(tmp_path))
    task = _segment()

    def failing_encoder(path: Path, *_args: object, **_kwargs: object) -> None:
        path.write_bytes(b"partial")
        msg = "encoder failed"
        raise RuntimeError(msg)

    with (
        patch("nemo_curator.stages.audio.io.write_audio.sf.write", side_effect=failing_encoder),
        pytest.raises(RuntimeError, match="encoder failed"),
    ):
        stage.process(task)
    assert task.data["audio_filepath"] == "/recordings/a.wav"
    assert "waveform" in task.data
    assert not list(tmp_path.rglob("*.opus"))


def test_unknown_tensor_metadata_is_rejected_before_encoding(tmp_path: Path) -> None:
    task = _segment()
    task.data["extra"] = torch.zeros(1)
    with pytest.raises(TypeError, match="not JSON serializable"):
        AudioSegmentWriterStage(output_dir=str(tmp_path)).process(task)
    assert not list(tmp_path.rglob("*.opus"))


def test_remapped_keys(tmp_path: Path) -> None:
    stage = AudioSegmentWriterStage(
        output_dir=str(tmp_path),
        waveform_key="samples",
        sample_rate_key="sr",
        audio_filepath_key="path",
        original_file_key="source",
        duration_key="seconds",
        start_ms_key="begin",
        end_ms_key="end",
    )

    def fixture() -> AudioTask:
        return AudioTask(
            data={"samples": torch.zeros(1600), "sr": 16000, "path": "/input.wav", "begin": 0, "end": 100}
        )

    assert_agent_ready(stage, fixture_factory=fixture)
    output = stage.process(fixture())
    assert output.data["source"] == "/input.wav"
    assert output.data["seconds"] == 0.1
    assert "samples" not in output.data


def test_key_aliases_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        AudioSegmentWriterStage(output_dir=str(tmp_path), waveform_key="audio_filepath")
