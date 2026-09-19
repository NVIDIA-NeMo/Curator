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

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage, has_overlap
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from tests.stages.audio.inference import review_helpers as rh

if TYPE_CHECKING:
    from typing import Any

hf_token = os.getenv("HF_TOKEN")


class TestPyannoteHasOverlap:
    """Tests for has_overlap helper."""

    def test_turn_overlaps_with_segment(self) -> None:
        """Turn that overlaps an overlap segment returns True."""

        class Turn:
            start = 0.0
            end = 2.0

        class Overlap:
            start = 1.0
            end = 1.5

        turn = Turn()
        overlaps = [Overlap()]
        assert has_overlap(turn, overlaps) is True

    def test_turn_after_overlap_returns_false(self) -> None:
        """Turn entirely after overlap returns False."""

        class Turn:
            start = 3.0
            end = 4.0

        class Overlap:
            start = 1.0
            end = 2.0

        turn = Turn()
        overlaps = [Overlap()]
        assert has_overlap(turn, overlaps) is False

    def test_turn_before_overlap_returns_false(self) -> None:
        """Turn entirely before overlap returns False."""

        class Turn:
            start = 0.0
            end = 0.5

        class Overlap:
            start = 1.0
            end = 2.0

        turn = Turn()
        overlaps = [Overlap()]
        assert has_overlap(turn, overlaps) is False

    def test_empty_overlaps_returns_false(self) -> None:
        """Empty overlaps list returns False."""

        class Turn:
            start = 0.0
            end = 1.0

        turn = Turn()
        assert has_overlap(turn, []) is False


class TestPyAnnoteDiarizationStage:
    """Tests for PyAnnoteDiarizationStage."""

    def test_xenna_num_workers_routes_through_generic_num_workers(self) -> None:
        stage = PyAnnoteDiarizationStage(hf_token=self.__class__.__name__, xenna_num_workers=4)

        assert stage.num_workers() == 4
        assert stage.xenna_stage_spec() == {}

    @pytest.mark.gpu
    @pytest.mark.skipif(not hf_token, reason="HF_TOKEN not set")
    def test_process(self, wav_filepath: Path) -> None:
        """Process a single entry for diarization."""
        stage = PyAnnoteDiarizationStage(hf_token=hf_token, resources=Resources(gpus=1))
        stage.setup_on_node()
        stage.setup()
        data_entry = {
            "resampled_audio_filepath": str(wav_filepath),
            "audio_item_id": "id_1",
            "duration": 60.0,
        }
        task = AudioTask(data=data_entry)
        result = stage.process(task)
        assert result.data["resampled_audio_filepath"] == str(wav_filepath)
        segments = result.data["segments"]
        assert len(segments) == 33
        # assert len(segments) < 100, "Sanity check: too many segments suggests an issue"
        for segment in segments:
            assert "start" in segment, "Segment should have start time"
            assert "end" in segment, "Segment should have end time"
            assert segment["start"] < segment["end"], "Start should be before end"
            assert 0 <= segment["start"] <= 60.0, "Start within audio duration"
            assert 0 <= segment["end"] <= 60.0, "End within audio duration"


@rh.pytest.mark.parametrize("residency", ["waveform", "auto"])
def test_pyannote_non_speaker_bounds_use_selected_resident_duration(
    residency: str, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "fallback.wav"
    rh._write_audio(audio_path, rh.np.zeros((1, 2), dtype=rh.np.float32))
    stage, _seen = rh._make_stage("pyannote", monkeypatch, input_residency=residency)
    monkeypatch.setattr(rh.pyannote_module, "add_non_speaker_segments", rh.add_non_speaker_segments)
    common = {"audio_filepath": str(audio_path)} if residency == "auto" else {}
    task = rh.AudioTask(
        data={
            **common,
            "waveform": rh.np.ones((1, 12), dtype=rh.np.float32),
            "sample_rate": rh._SAMPLE_RATE,
            "duration": 99.0,
        }
    )
    result = stage.process_batch([task])[0]
    assert max(segment["end"] for segment in result.data["segments"]) == rh.pytest.approx(1.2)


@rh.pytest.mark.parametrize("fail_after_rttm", [False, True], ids=["success", "failure"])
def test_pyannote_cleans_temporary_rttm_siblings(
    fail_after_rttm: bool, tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    stage, _seen = rh._make_stage(
        "pyannote", monkeypatch, input_residency="waveform", write_rttm=True, fail_after_rttm=fail_after_rttm
    )

    def temp_resolver(item: dict[str, Any], **kwargs: object) -> str | None:
        return rh.resolve_audio_path(item, temp_dir=str(tmp_path), **kwargs)

    monkeypatch.setattr(rh.pyannote_module, "resolve_audio_path", temp_resolver)
    task = rh.AudioTask(data={"waveform": rh.np.ones((1, 12), dtype=rh.np.float32), "sample_rate": rh._SAMPLE_RATE})
    if fail_after_rttm:
        with rh.pytest.raises(RuntimeError, match="failure after RTTM"):
            stage.process(task)
    else:
        stage.process(task)
    assert list(tmp_path.glob("*.wav")) == []
    assert list(tmp_path.glob("*.rttm")) == []


def test_pyannote_preserves_durable_file_rttm(tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch) -> None:
    stage, _seen = rh._make_stage("pyannote", monkeypatch, write_rttm=True)
    audio_path = tmp_path / "durable.wav"
    rh._write_audio(audio_path, rh.np.ones((1, 12), dtype=rh.np.float32))
    stage.process(rh.AudioTask(data={"audio_filepath": str(audio_path)}))
    assert audio_path.with_suffix(".rttm").exists()


def test_pyannote_fanout_excludes_silence_and_includes_overlap(
    tmp_path: Path, monkeypatch: rh.pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "source.wav"
    rh._write_audio(audio_path, rh.np.arange(50, dtype=rh.np.float32)[None, :])
    annotation = rh._FakeAnnotation(
        [{"start": 1.0, "end": 2.0, "speaker": "speaker_0"}, {"start": 2.0, "end": 3.0, "speaker": "speaker_1"}],
        overlap_segments=[rh.pyannote_module.Segment(2.0, 3.0)],
    )
    monkeypatch.setattr(rh.pyannote_module, "ProgressHook", rh._ProgressHook)
    stage = rh.PyAnnoteDiarizationStage(
        fanout=True, min_length=0.0, max_length=40.0, write_rttm=False, resources=rh.Resources(gpus=0)
    )
    stage._pipeline = lambda _payload, hook=None: (hook, annotation)[1]
    stage._vad_model = rh.MagicMock()
    children = stage.process(rh.AudioTask(data={"resampled_audio_filepath": str(audio_path)}))
    assert [(child.data["start"], child.data["end"], child.data["is_overlap"]) for child in children] == [
        (1.0, 2.0, False),
        (2.0, 3.0, True),
    ]
    assert all(child.data.get("speaker") != "no-speaker" for child in children)
