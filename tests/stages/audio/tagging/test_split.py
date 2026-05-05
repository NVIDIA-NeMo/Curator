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

from collections.abc import Callable

from nemo_curator.stages.audio.tagging.split import (
    JoinSplitAudioMetadataStage,
    SplitLongAudioStage,
)
from nemo_curator.tasks import AudioTask


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
    """Tests for SplitLongAudioStage.process (no actual audio I/O)."""

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
