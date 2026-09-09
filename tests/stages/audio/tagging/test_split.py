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
from fsspec.core import url_to_fs

from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio.tagging.split import (
    JoinSplitAudioMetadataStage,
    SplitASRAlignJoinStage,
    SplitLongAudioStage,
)
from nemo_curator.tasks import AudioTask


def _patch_audio_io(monkeypatch: pytest.MonkeyPatch, saved_paths: list[str]) -> None:
    def fake_load(_path: str) -> tuple[np.ndarray, int]:
        return np.linspace(-0.5, 0.5, 80, dtype=np.float32)[None, :], 10

    def fake_save(path: str, waveform: np.ndarray, sample_rate: int) -> None:
        saved_paths.append(path)
        sf.write(path, waveform.T, sample_rate)

    monkeypatch.setattr("nemo_curator.stages.audio.tagging.split.torchaudio.load", fake_load)
    monkeypatch.setattr("nemo_curator.stages.audio.tagging.split.torchaudio.save", fake_save)


def test_additive_fields_preserve_legacy_positional_arguments() -> None:
    splitter = SplitLongAudioStage(120.0, 2.0, "custom-split")
    joiner = JoinSplitAudioMetadataStage("transcript", "custom-join")
    composite = SplitASRAlignJoinStage(120.0, 2.0, "legacy/model")

    assert splitter.name == "custom-split"
    assert splitter.duration_key == "duration"
    assert splitter.output_dir is None
    assert joiner.text_key == "transcript"
    assert joiner.name == "custom-join"
    assert joiner.split_filepaths_key == "split_filepaths"
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

        stem = stage._shared_output_stem("recording", str(source_path), [4.0], 10, 80)
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

    def test_same_source_with_different_split_plans_gets_distinct_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        saved_paths: list[str] = []
        _patch_audio_io(monkeypatch, saved_paths)
        source_path = tmp_path / "same.wav"
        output_dir = tmp_path / "chunks"
        plans = [
            [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
            [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 8.0}],
        ]
        emitted: list[list[str]] = []

        for segments in plans:
            result = SplitLongAudioStage(
                suggested_max_len=5.0,
                min_len=0.5,
                output_dir=str(output_dir),
            ).process(
                audio_task(
                    duration=8.0,
                    audio_item_id="same",
                    resampled_audio_filepath=str(source_path),
                    segments=segments,
                )
            )
            emitted.append(result.data["split_filepaths"])

        assert set(emitted[0]).isdisjoint(emitted[1])
        assert len(saved_paths) == len(set(saved_paths)) == 4
        assert sorted(sf.info(path).frames for path in saved_paths) == [30, 40, 40, 50]

    def test_known_short_source_hash_collision_gets_distinct_shared_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        first = "/dataset/spk25433/utt.wav"
        second = "/dataset/spk158142/utt.wav"
        assert hashlib.sha256(first.encode()).hexdigest()[:8] == hashlib.sha256(second.encode()).hexdigest()[:8]
        saved_paths: list[str] = []
        _patch_audio_io(monkeypatch, saved_paths)

        for source in (first, second):
            SplitLongAudioStage(
                suggested_max_len=5.0,
                min_len=0.5,
                output_dir=str(tmp_path / "chunks"),
            ).process(
                audio_task(
                    duration=8.0,
                    audio_item_id="utt",
                    resampled_audio_filepath=source,
                    segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
                )
            )

        assert len(saved_paths) == len(set(saved_paths)) == 4

    def test_remote_output_paths_are_written_through_fsspec(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        local_writes: list[str] = []
        _patch_audio_io(monkeypatch, local_writes)
        output_dir = f"memory://split-{tmp_path.name}"
        result = SplitLongAudioStage(
            suggested_max_len=5.0,
            min_len=0.5,
            output_dir=output_dir,
        ).process(
            audio_task(
                duration=8.0,
                audio_item_id="remote",
                resampled_audio_filepath=str(tmp_path / "source.wav"),
                segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
            )
        )

        assert all(path.startswith(f"{output_dir}/") for path in result.data["split_filepaths"])
        for advertised in result.data["split_filepaths"]:
            fs, path = url_to_fs(advertised)
            assert fs.exists(path)
        assert local_writes
        assert all(not Path(path).exists() for path in local_writes)

    def test_remote_output_cleans_local_temp_when_save_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        local_writes: list[str] = []
        _patch_audio_io(monkeypatch, local_writes)

        def fail_save(path: str, waveform: np.ndarray, sample_rate: int) -> None:
            local_writes.append(path)
            sf.write(path, waveform.T, sample_rate)
            message = "local encoding failed"
            raise OSError(message)

        monkeypatch.setattr("nemo_curator.stages.audio.tagging.split.torchaudio.save", fail_save)
        stage = SplitLongAudioStage(
            suggested_max_len=5.0,
            min_len=0.5,
            output_dir=f"memory://split-failure-{tmp_path.name}",
        )

        with pytest.raises(OSError, match="local encoding failed"):
            stage.process(
                audio_task(
                    duration=8.0,
                    audio_item_id="remote",
                    resampled_audio_filepath=str(tmp_path / "source.wav"),
                    segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
                )
            )

        assert len(local_writes) == 1
        assert not Path(local_writes[0]).exists()

    def test_static_contract_exposes_conservative_split_gates(self, tmp_path: Path) -> None:
        static = static_contract(SplitLongAudioStage)
        configured_default = build_contract(SplitLongAudioStage())
        configured_shared = build_contract(SplitLongAudioStage(output_dir=str(tmp_path)))

        assert static.gates.writes_to_disk is True
        assert static.gates.output_path_params == ["output_dir"]
        assert static.gates.per_row_independent is False
        assert configured_default.gates.per_row_independent is True
        assert configured_shared.gates.per_row_independent is False


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
