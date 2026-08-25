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

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.tagging.split import (
    JoinSplitAudioMetadataStage,
    SplitASRAlignJoinStage,
    SplitLongAudioStage,
)
from nemo_curator.tasks import AudioTask


class _FakeWaveform:
    """Minimal waveform stand-in for split path tests."""

    def __getitem__(self, key: object) -> object:
        if key == 0:
            return [0.0] * 80
        return self


def _patch_audio_io(monkeypatch: pytest.MonkeyPatch, saved_paths: list[str]) -> None:
    def fake_load(_path: str) -> tuple[_FakeWaveform, int]:
        return _FakeWaveform(), 10

    def fake_save(path: str, _waveform: object, _sample_rate: int) -> None:
        saved_paths.append(path)
        Path(path).touch()

    monkeypatch.setattr("nemo_curator.stages.audio.tagging.split.torchaudio.load", fake_load)
    monkeypatch.setattr("nemo_curator.stages.audio.tagging.split.torchaudio.save", fake_save)


def test_additive_output_dir_preserves_legacy_positional_arguments() -> None:
    splitter = SplitLongAudioStage(120.0, 2.0, "legacy_duration")
    composite = SplitASRAlignJoinStage(120.0, 2.0, "legacy/model")

    assert splitter.duration_key == "legacy_duration"
    assert splitter.output_dir is None
    assert composite.model_name == "legacy/model"
    assert composite.output_dir is None


class TestSplitLongAudioStageGetSplitPoints:
    """Tests for SplitLongAudioStage.get_split_points."""

    def test_no_splits_when_segments_short(self) -> None:
        """No split points when total duration under suggested_max_len."""
        stage = SplitLongAudioStage(suggested_max_len=3600.0)
        metadata = {
            "segments": [
                {"start": 0.0, "end": 100.0},
                {"start": 100.0, "end": 200.0},
            ]
        }
        splits = stage.get_split_points(metadata)
        assert splits == []

    def test_split_point_when_exceeds_max_len(self) -> None:
        """Split point added when segment span exceeds suggested_max_len."""
        stage = SplitLongAudioStage(suggested_max_len=40)
        metadata = {
            "segments": [
                {"start": 0.0, "end": 20.0},
                {"start": 20.0, "end": 40.0},
                {"start": 40.0, "end": 60.0},
                {"start": 60.0, "end": 90.0},
            ]
        }
        splits = stage.get_split_points(metadata)
        assert len(splits) == 2
        assert 40.0 in splits
        assert 60.0 in splits

    def test_empty_segments_returns_empty_splits(self) -> None:
        """Empty segments list returns no split points."""
        stage = SplitLongAudioStage(suggested_max_len=100.0)
        metadata = {"segments": []}
        splits = stage.get_split_points(metadata)
        assert splits == []


class TestSplitLongAudioStageProcessDatasetEntry:
    """Tests for SplitLongAudioStage.process."""

    def test_short_audio_passthrough(self, audio_task: Callable[..., AudioTask]) -> None:
        """When duration < suggested_max_len, entry returned with split_filepaths wrapping the filepath."""
        stage = SplitLongAudioStage(suggested_max_len=3600.0)
        task = audio_task(
            duration=100.0,
            audio_item_id="test_1",
            resampled_audio_filepath="test_1_resampled.wav",
        )
        result = stage.process(task)
        out = result.data
        assert out["split_filepaths"] == ["test_1_resampled.wav"]

    def test_long_audio_round_trip_with_torchaudio(
        self,
        tmp_path: Path,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        sample_rate = 8000
        audio_path = tmp_path / "long.wav"
        sf.write(audio_path, np.zeros(sample_rate * 3, dtype=np.float32), sample_rate)
        stage = SplitLongAudioStage(suggested_max_len=1.5, min_len=0.5)
        task = audio_task(
            duration=3.0,
            segments=[
                {"start": 0.0, "end": 1.0},
                {"start": 1.0, "end": 2.0},
                {"start": 2.0, "end": 3.0},
            ],
            audio_item_id="long",
            resampled_audio_filepath=str(audio_path),
        )

        result = stage.process(task)

        assert len(result.data["split_filepaths"]) == 3
        assert result.data["split_offsets"] == [0.0, 1.0, 2.0]
        assert all(sf.info(path).frames == sample_rate for path in result.data["split_filepaths"])

    def test_default_output_paths_remain_source_adjacent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        """The default keeps the exact sibling path format used before output_dir existed."""
        saved_paths: list[str] = []
        _patch_audio_io(monkeypatch, saved_paths)
        source_path = tmp_path / "recording.flac"
        stage = SplitLongAudioStage(suggested_max_len=5.0, min_len=0.5)
        task = audio_task(
            duration=8.0,
            audio_item_id="sample",
            resampled_audio_filepath=str(source_path),
            segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
        )

        result = stage.process(task)

        expected_paths = [
            str(tmp_path / "recording.1_of_2.wav"),
            str(tmp_path / "recording.2_of_2.wav"),
        ]
        assert saved_paths == expected_paths
        assert result.data["split_filepaths"] == expected_paths
        assert [entry["resampled_audio_filepath"] for entry in result.data["split_metadata"]] == expected_paths

    def test_output_dir_redirects_written_and_returned_split_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        """An explicit output_dir contains every written and returned chunk."""
        saved_paths: list[str] = []
        _patch_audio_io(monkeypatch, saved_paths)
        source_dir = tmp_path / "input"
        source_dir.mkdir()
        source_path = source_dir / "recording.flac"
        output_dir = tmp_path / "smoke-chunks"
        stage = SplitLongAudioStage(
            suggested_max_len=5.0,
            min_len=0.5,
            output_dir=str(output_dir),
        )
        task = audio_task(
            duration=8.0,
            audio_item_id="sample",
            resampled_audio_filepath=str(source_path),
            segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
        )

        result = stage.process(task)

        # Under output_dir the stem carries a path hash; see the collision test below.
        stem = f"recording_{hashlib.sha256(str(source_path).encode()).hexdigest()[:8]}"
        expected_paths = [
            str(output_dir / f"{stem}.1_of_2.wav"),
            str(output_dir / f"{stem}.2_of_2.wav"),
        ]
        assert output_dir.is_dir()
        assert saved_paths == expected_paths
        assert result.data["split_filepaths"] == expected_paths
        assert [entry["resampled_audio_filepath"] for entry in result.data["split_metadata"]] == expected_paths
        assert not list(source_dir.glob("recording.*_of_2.wav"))

    def test_two_recordings_sharing_a_basename_do_not_overwrite_each_other(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        saved_paths: list[str] = []
        _patch_audio_io(monkeypatch, saved_paths)
        output_dir = tmp_path / "chunks"

        for speaker in ("spk1", "spk2"):
            source_dir = tmp_path / speaker
            source_dir.mkdir()
            stage = SplitLongAudioStage(suggested_max_len=5.0, min_len=0.5, output_dir=str(output_dir))
            stage.process(
                audio_task(
                    duration=8.0,
                    audio_item_id="utt1",
                    resampled_audio_filepath=str(source_dir / "utt1.wav"),
                    segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
                )
            )

        assert len(saved_paths) == len(set(saved_paths)), f"one speaker overwrote the other: {saved_paths}"


def test_split_asr_align_join_forwards_output_dir(tmp_path: Path) -> None:
    """Composite construction forwards both redirected and legacy defaults."""
    output_dir = str(tmp_path / "smoke-chunks")
    redirected_splitter = SplitASRAlignJoinStage(output_dir=output_dir).decompose()[0]
    default_splitter = SplitASRAlignJoinStage().decompose()[0]

    assert isinstance(redirected_splitter, SplitLongAudioStage)
    assert redirected_splitter.output_dir == output_dir
    assert isinstance(default_splitter, SplitLongAudioStage)
    assert default_splitter.output_dir is None


class TestJoinSplitAudioMetadataStage:
    """Tests for JoinSplitAudioMetadataStage."""

    def test_no_split_passthrough(self, audio_task: Callable[..., AudioTask]) -> None:
        """Entry with split_filepaths=None (no split occurred) returns entry without key."""
        stage = JoinSplitAudioMetadataStage()
        task = audio_task(
            audio_item_id="x",
            split_filepaths=None,
            text="hello",
        )
        result = stage.process(task)
        out = result.data
        assert "split_filepaths" not in out
        assert out["text"] == "hello"

    def test_join_split_metadata_concatenates_text_and_alignments(self, audio_task: Callable[..., AudioTask]) -> None:
        """Meta-entry with split_metadata joins text and adjusts alignment timestamps."""
        stage = JoinSplitAudioMetadataStage()
        task = audio_task(
            audio_item_id="parent",
            split_filepaths=["/path/a.wav", "/path/b.wav"],
            split_metadata=[
                {
                    "text": "first part",
                    "alignment": [
                        {"word": "first", "start": 0.0, "end": 0.5},
                        {"word": "part", "start": 0.5, "end": 1.0},
                    ],
                },
                {
                    "text": "second part",
                    "alignment": [
                        {"word": "second", "start": 0.0, "end": 0.5},
                        {"word": "part", "start": 0.5, "end": 1.0},
                    ],
                },
            ],
            split_offsets=[0.0, 5.0],
        )
        result = stage.process(task)
        out = result.data
        assert out["text"] == "first part second part"
        assert "split_filepaths" not in out
        assert "split_metadata" not in out
        align = out["alignment"]
        assert len(align) == 4
        assert align[0]["word"] == "first"
        assert align[0]["start"] == 0.0
        assert align[0]["end"] == 0.5
        assert align[2]["word"] == "second"
        assert align[2]["start"] == 5.0
        assert align[2]["end"] == 5.5
