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

"""Tests for MergeConversationSDPStage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.merging.merge_conversation import (
    MergeConversationSDPStage,
)
from nemo_curator.tasks import AudioTask

SR = 16000


def _write_wav(path: Path, duration_sec: float = 1.0, sr: int = SR) -> str:
    """Write a sine-wave WAV and return its path as a string."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return str(path)


def _write_rttm(path: Path, file_id: str, speaker: str,
                segments: list[tuple[float, float]]) -> str:
    """Write an RTTM file and return its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(
            f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>\n"
            for start, dur in segments
        )
    return str(path)


def _write_ctm(path: Path, file_id: str,
               words: list[tuple[float, float, str]]) -> str:
    """Write a CTM file and return its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(
            f"{file_id} 1 {start:.3f} {dur:.3f} {word}\n"
            for start, dur, word in words
        )
    return str(path)


def _make_task(data: dict, task_id: str = "t1") -> AudioTask:
    return AudioTask(data=data, task_id=task_id, dataset_name="test")


def _build_stage(tmp_path: Path, **overrides) -> MergeConversationSDPStage:
    defaults = {"output_conversations_dir": str(tmp_path / "conversations")}
    defaults.update(overrides)
    return MergeConversationSDPStage(**defaults)


class TestConstruction:
    def test_defaults(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)
        assert stage.max_pause_duration == 2.0
        assert stage.max_intra_turn_pause == 1.0
        assert stage.randomize_pauses is False
        assert stage.seglst_offset == 0.1

    def test_custom_params(self, tmp_path: Path) -> None:
        stage = _build_stage(
            tmp_path,
            max_pause_duration=3.0,
            max_intra_turn_pause=0.5,
            randomize_pauses=True,
            seglst_offset=0.2,
        )
        assert stage.max_pause_duration == 3.0
        assert stage.max_intra_turn_pause == 0.5
        assert stage.randomize_pauses is True
        assert stage.seglst_offset == 0.2

    def test_process_raises(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)
        with pytest.raises(NotImplementedError, match="only supports process_batch"):
            stage.process(_make_task({"audio_filepath": "/a.wav"}))


class TestEmptyBatch:
    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)
        assert stage.process_batch([]) == []


class TestParseRTTM:
    def test_valid_rttm(self, tmp_path: Path) -> None:
        rttm = _write_rttm(
            tmp_path / "test.rttm", "file1", "spk",
            [(0.0, 0.5), (1.0, 0.3)],
        )
        result = MergeConversationSDPStage._parse_rttm_timestamps(rttm)
        assert len(result) == 2
        assert result[0] == (0.0, 0.5)
        assert result[1] == (1.0, 0.3)

    def test_missing_file(self):
        assert MergeConversationSDPStage._parse_rttm_timestamps("/nonexistent") == []

    def test_empty_path(self):
        assert MergeConversationSDPStage._parse_rttm_timestamps("") == []


class TestParseCTM:
    def test_valid_ctm(self, tmp_path: Path) -> None:
        ctm = _write_ctm(
            tmp_path / "test.ctm", "file1",
            [(0.1, 0.2, "hello"), (0.5, 0.3, "world")],
        )
        result = MergeConversationSDPStage._parse_ctm_words(ctm)
        assert len(result) == 2
        assert result[0][2] == "hello"
        assert result[1][2] == "world"

    def test_missing_file(self):
        assert MergeConversationSDPStage._parse_ctm_words("/nonexistent") == []


class TestFindNearestSegment:
    def test_within_segment(self):
        segments = [(0.0, 0.5), (1.0, 0.5)]
        assert MergeConversationSDPStage._find_nearest_segment(0.3, segments) == 0
        assert MergeConversationSDPStage._find_nearest_segment(1.2, segments) == 1

    def test_between_segments(self):
        segments = [(0.0, 0.3), (2.0, 0.3)]
        assert MergeConversationSDPStage._find_nearest_segment(0.5, segments) == 0
        assert MergeConversationSDPStage._find_nearest_segment(1.8, segments) == 1


class TestExtractSpeakingSegments:
    def test_extracts_speech(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "audio.wav"
        _write_wav(wav_path, duration_sec=3.0)
        rttm = _write_rttm(
            tmp_path / "audio.rttm", "audio", "spk",
            [(0.0, 0.5), (1.0, 0.5)],
        )

        stage = _build_stage(tmp_path)
        audio, sr = stage.extract_speaking_segments(str(wav_path), rttm)
        assert audio is not None
        assert sr == SR
        assert len(audio) < 3 * SR

    def test_no_rttm_returns_none(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "audio.wav"
        _write_wav(wav_path, duration_sec=1.0)

        stage = _build_stage(tmp_path)
        audio, sr = stage.extract_speaking_segments(str(wav_path), "")
        assert audio is None
        assert sr is None


class TestComputeTimeline:
    def test_single_turn(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)
        rttm = _write_rttm(
            tmp_path / "t1.rttm", "t1", "spk", [(0.0, 1.0)],
        )
        turns = [{"rttm_filepath": rttm, "overlap": 0}]
        timeline = stage._compute_timeline(turns, [0.0])
        assert len(timeline) == 1
        assert len(timeline[0].adjusted_segments) == 1
        assert timeline[0].adjusted_segments[0][0] == pytest.approx(0.0)

    def test_two_turns_no_overlap(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])
        rttm2 = _write_rttm(tmp_path / "t2.rttm", "t2", "B", [(0.0, 0.5)])
        turns = [
            {"rttm_filepath": rttm1, "overlap": 0},
            {"rttm_filepath": rttm2, "overlap": 0},
        ]
        timeline = stage._compute_timeline(turns, [0.0, 0.0])
        assert len(timeline) == 2
        assert timeline[1].time_offset >= timeline[0].local_offset


class TestProcessBatch:
    def test_two_turn_conversation(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        wav2 = _write_wav(tmp_path / "t2.wav", 0.8)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "Alice", [(0.0, 1.0)])
        rttm2 = _write_rttm(tmp_path / "t2.rttm", "t2", "Bob", [(0.0, 0.8)])

        tasks = [
            _make_task({
                "audio_filepath": wav1,
                "rttm_filepath": rttm1,
                "speaker": "Alice",
                "conversation_id": "conv001",
                "turn_index": 0,
                "overlap": 0,
                "utterance": "Hello Bob",
            }, task_id="t1"),
            _make_task({
                "audio_filepath": wav2,
                "rttm_filepath": rttm2,
                "speaker": "Bob",
                "conversation_id": "conv001",
                "turn_index": 1,
                "overlap": 0,
                "utterance": "Hi Alice",
            }, task_id="t2"),
        ]

        results = stage.process_batch(tasks)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, AudioTask)
        assert result.data["conversation_id"] == "conv001"
        assert result.data["num_speakers"] == 2
        assert result.data["duration"] > 0
        assert os.path.exists(result.data["audio_filepath"])
        assert "mixed.wav" in result.data["audio_filepath"]

        conv_dir = Path(result.data["audio_filepath"]).parent
        assert (conv_dir / "Alice.wav").exists()
        assert (conv_dir / "Bob.wav").exists()
        assert (conv_dir / "multichannel.wav").exists()

    def test_missing_audio_skips_turn(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "Alice", [(0.0, 1.0)])

        tasks = [
            _make_task({
                "audio_filepath": wav1,
                "rttm_filepath": rttm1,
                "speaker": "Alice",
                "conversation_id": "conv002",
                "turn_index": 0,
                "overlap": 0,
            }, task_id="t1"),
            _make_task({
                "audio_filepath": "/nonexistent.wav",
                "speaker": "Bob",
                "conversation_id": "conv002",
                "turn_index": 1,
                "overlap": 0,
            }, task_id="t2"),
        ]

        results = stage.process_batch(tasks)
        assert len(results) == 1
        assert results[0].data["num_speakers"] == 1

    def test_all_missing_audio_returns_empty(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)
        tasks = [
            _make_task({
                "audio_filepath": "/nonexistent.wav",
                "speaker": "Alice",
                "conversation_id": "conv003",
                "turn_index": 0,
            }),
        ]
        results = stage.process_batch(tasks)
        assert results == []


class TestRTTMOutput:
    def test_rttm_files_created(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        wav2 = _write_wav(tmp_path / "t2.wav", 0.5)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])
        rttm2 = _write_rttm(tmp_path / "t2.rttm", "t2", "B", [(0.0, 0.5)])

        tasks = [
            _make_task({
                "audio_filepath": wav1, "rttm_filepath": rttm1,
                "speaker": "A", "conversation_id": "c1",
                "turn_index": 0, "overlap": 0,
            }),
            _make_task({
                "audio_filepath": wav2, "rttm_filepath": rttm2,
                "speaker": "B", "conversation_id": "c1",
                "turn_index": 1, "overlap": 0,
            }),
        ]

        results = stage.process_batch(tasks)
        assert len(results) == 1

        conv_dir = Path(results[0].data["audio_filepath"]).parent
        assert (conv_dir / "A.rttm").exists()
        assert (conv_dir / "B.rttm").exists()
        assert (conv_dir / "all.rttm").exists()

        all_rttm = (conv_dir / "all.rttm").read_text()
        assert "SPEAKER" in all_rttm


class TestCTMOutput:
    def test_ctm_files_created(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])
        ctm1 = _write_ctm(
            tmp_path / "t1.ctm", "t1",
            [(0.1, 0.2, "hello"), (0.4, 0.3, "world")],
        )

        tasks = [
            _make_task({
                "audio_filepath": wav1,
                "rttm_filepath": rttm1,
                "ctm_filepath": ctm1,
                "speaker": "A",
                "conversation_id": "c2",
                "turn_index": 0,
                "overlap": 0,
            }),
        ]

        results = stage.process_batch(tasks)
        conv_dir = Path(results[0].data["audio_filepath"]).parent
        assert (conv_dir / "A.ctm").exists()
        assert (conv_dir / "all.ctm").exists()

        ctm_content = (conv_dir / "all.ctm").read_text()
        assert "hello" in ctm_content
        assert "world" in ctm_content


class TestSeglst:
    def test_seglst_generated(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])

        tasks = [
            _make_task({
                "audio_filepath": wav1,
                "rttm_filepath": rttm1,
                "speaker": "A",
                "conversation_id": "c3",
                "turn_index": 0,
                "overlap": 0,
                "utterance": "Hello world",
            }),
        ]

        results = stage.process_batch(tasks)
        conv_dir = Path(results[0].data["audio_filepath"]).parent
        seglst_path = conv_dir / "segments.seglst.json"
        assert seglst_path.exists()

        seglst = json.loads(seglst_path.read_text())
        assert len(seglst) == 1
        assert seglst[0]["speaker"] == "A"
        assert seglst[0]["words"] == "Hello world"
        assert seglst[0]["start_time"] >= 0
        assert seglst[0]["end_time"] > 0


class TestOverlap:
    def test_negative_overlap_adds_pause(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        wav2 = _write_wav(tmp_path / "t2.wav", 1.0)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])
        rttm2 = _write_rttm(tmp_path / "t2.rttm", "t2", "B", [(0.0, 1.0)])

        tasks = [
            _make_task({
                "audio_filepath": wav1, "rttm_filepath": rttm1,
                "speaker": "A", "conversation_id": "c4",
                "turn_index": 0, "overlap": 0,
            }),
            _make_task({
                "audio_filepath": wav2, "rttm_filepath": rttm2,
                "speaker": "B", "conversation_id": "c4",
                "turn_index": 1, "overlap": -0.5,
            }),
        ]

        results = stage.process_batch(tasks)
        assert results[0].data["duration"] > 2.0


class TestMFAFallback:
    def test_mfa_fallback_flag(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])

        tasks = [
            _make_task({
                "audio_filepath": wav1, "rttm_filepath": rttm1,
                "speaker": "A", "conversation_id": "c5",
                "turn_index": 0, "overlap": 0,
                "mfa_skipped": True,
            }),
        ]

        results = stage.process_batch(tasks)
        assert results[0].data["mfa_fallback"] is True

    def test_no_mfa_fallback(self, tmp_path: Path) -> None:
        stage = _build_stage(tmp_path)

        wav1 = _write_wav(tmp_path / "t1.wav", 1.0)
        rttm1 = _write_rttm(tmp_path / "t1.rttm", "t1", "A", [(0.0, 1.0)])

        tasks = [
            _make_task({
                "audio_filepath": wav1, "rttm_filepath": rttm1,
                "speaker": "A", "conversation_id": "c6",
                "turn_index": 0, "overlap": 0,
            }),
        ]

        results = stage.process_batch(tasks)
        assert results[0].data["mfa_fallback"] is False
