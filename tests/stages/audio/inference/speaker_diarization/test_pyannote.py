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

import os
from pathlib import Path

import pytest

from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage, has_overlap
from nemo_curator.tasks import AudioTask

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

    @pytest.mark.skipif(not hf_token, reason="HF_TOKEN not set")
    def test_process(self, wav_filepath: Path) -> None:
        """Process a single entry for diarization."""
        stage = PyAnnoteDiarizationStage(hf_token=hf_token)
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
        assert len(segments) > 0, "Should produce at least one segment"
        assert len(segments) < 100, "Sanity check: too many segments suggests an issue"
        for segment in segments:
            assert "start" in segment, "Segment should have start time"
            assert "end" in segment, "Segment should have end time"
            assert segment["start"] < segment["end"], "Start should be before end"
            assert 0 <= segment["start"] <= 60.0, "Start within audio duration"
            assert 0 <= segment["end"] <= 60.0, "End within audio duration"
