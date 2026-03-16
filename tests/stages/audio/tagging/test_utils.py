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


from nemo_curator.stages.audio.tagging.utils import add_non_speaker_segments


class TestAddNonSpeakerSegments:
    """Tests for add_non_speaker_segments utility."""

    def test_adds_gap_before_first_segment(self) -> None:
        """Non-speaker segment is added from 0 to first segment start."""
        segments = [{"speaker": "s1", "start": 2.0, "end": 5.0}]
        add_non_speaker_segments(segments, audio_duration=10.0)
        assert len(segments) == 3
        assert segments[0]["speaker"] == "no-speaker"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.0
        assert segments[1]["speaker"] == "s1"
        assert segments[1]["start"] == 2.0
        assert segments[1]["end"] == 5.0
        assert segments[2]["speaker"] == "no-speaker"
        assert segments[2]["start"] == 5.0
        assert segments[2]["end"] == 10.0

    def test_adds_gap_after_last_segment(self) -> None:
        """Non-speaker segment is added from last segment end to audio_duration."""
        segments = [{"speaker": "s1", "start": 0.0, "end": 3.0}]
        add_non_speaker_segments(segments, audio_duration=10.0)
        assert len(segments) == 2
        assert segments[1]["speaker"] == "no-speaker"
        assert segments[1]["start"] == 3.0
        assert segments[1]["end"] == 10.0

    def test_adds_gap_between_segments(self) -> None:
        """Gap between two speaker segments becomes no-speaker."""
        segments = [
            {"speaker": "s1", "start": 0.0, "end": 2.0},
            {"speaker": "s2", "start": 5.0, "end": 8.0},
        ]
        add_non_speaker_segments(segments, audio_duration=10.0)
        assert len(segments) == 4
        # Sorted by start
        assert segments[0]["speaker"] == "s1"
        assert segments[0]["start"] == 0.0
        assert segments[1]["speaker"] == "no-speaker"
        assert segments[1]["start"] == 2.0
        assert segments[1]["end"] == 5.0
        assert segments[2]["speaker"] == "s2"
        assert segments[2]["start"] == 5.0
        assert segments[3]["speaker"] == "no-speaker"
        assert segments[3]["start"] == 8.0
        assert segments[3]["end"] == 10.0

    def test_max_length_splits_non_speaker_segments(self) -> None:
        """When max_length is set, long no-speaker regions are split."""
        segments = [
            {"speaker": "s1", "start": 0.0, "end": 1.0},
            {"speaker": "s2", "start": 6.0, "end": 7.0},
        ]
        add_non_speaker_segments(segments, audio_duration=10.0, max_length=2.0)
        # Gap 1-6 should be split into chunks of max 2s: 1-3, 3-5, 5-6; gap 7-10 into 7-9, 9-10
        no_speaker = [s for s in segments if s["speaker"] == "no-speaker"]
        assert len(no_speaker) >= 2
        for seg in no_speaker:
            assert seg["end"] - seg["start"] <= 2.0
