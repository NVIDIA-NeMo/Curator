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

from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.audio.tagging.utils import (
    add_non_speaker_segments,
    load_vocab_file,
    validate_tagging_outputs,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _valid_prepared_segment(start: object = 0.0, end: object = 2.0, **overrides: object) -> dict[str, object]:
    segment: dict[str, object] = {
        "speaker": "speaker_0",
        "start": start,
        "end": end,
        "text": "hello",
        "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
        "metrics": {"bandwidth": [8000]},
    }
    segment.update(overrides)
    return segment


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


class TestLoadVocabFile:
    def test_single_line(self, tmp_path: Path) -> None:
        p = tmp_path / "v.txt"
        p.write_text("abc xyz")
        result = load_vocab_file(str(p))
        assert "a" in result
        assert " " in result

    def test_multi_line(self, tmp_path: Path) -> None:
        p = tmp_path / "v.txt"
        p.write_text("a\nb\nc\n")
        result = load_vocab_file(str(p))
        assert "a" in result
        assert "b" in result
        assert "c" in result


class TestValidateTaggingOutputs:
    @staticmethod
    def _task(duration: object = 10.0, segments: object = None) -> AudioTask:
        if segments is None:
            segments = [_valid_prepared_segment()]
        return AudioTask(data={"duration": duration, "segments": segments})

    def test_returns_output_metrics(self) -> None:
        metrics = validate_tagging_outputs(
            [
                self._task(
                    duration=10.0,
                    segments=[
                        _valid_prepared_segment(start=0.0, end=2.0),
                        _valid_prepared_segment(start=2.0, end=4.0),
                    ],
                ),
                self._task(duration=20.0, segments=[]),
            ]
        )

        assert metrics == {
            "num_tasks_processed": 2,
            "num_tasks_with_segments": 1,
            "num_segments_processed": 2,
            "segment_task_coverage_ratio": 0.5,
            "total_audio_duration_hours": 30.0 / 3600,
            "tagged_audio_duration_hours": 4.0 / 3600,
        }

    def test_rejects_empty_outputs(self) -> None:
        with pytest.raises(RuntimeError, match="no output tasks"):
            validate_tagging_outputs([])

    @pytest.mark.parametrize("duration", [None, 0, -1, float("nan")])
    def test_rejects_invalid_duration(self, duration: object) -> None:
        with pytest.raises(RuntimeError, match="duration"):
            validate_tagging_outputs([self._task(duration=duration)])

    def test_rejects_missing_segments(self) -> None:
        with pytest.raises(TypeError, match="segments list"):
            validate_tagging_outputs([AudioTask(data={"duration": 10.0})])

    def test_rejects_zero_segments(self) -> None:
        with pytest.raises(RuntimeError, match="no tagged segments"):
            validate_tagging_outputs([self._task(segments=[])])

    @pytest.mark.parametrize(
        ("segment", "error_match"),
        [
            pytest.param(None, "must be a mapping", id="null"),
            pytest.param("not-a-segment", "must be a mapping", id="non-mapping"),
            pytest.param({}, "missing required fields", id="missing-fields"),
            pytest.param(_valid_prepared_segment(start=float("nan")), "finite number", id="non-finite-start"),
            pytest.param(_valid_prepared_segment(end=float("inf")), "finite number", id="non-finite-end"),
            pytest.param(_valid_prepared_segment(start=2.0, end=2.0), "start < end", id="empty-range"),
            pytest.param(_valid_prepared_segment(start=3.0, end=2.0), "start < end", id="reversed-range"),
        ],
    )
    def test_rejects_malformed_segments(self, segment: object, error_match: str) -> None:
        with pytest.raises((RuntimeError, TypeError), match=error_match):
            validate_tagging_outputs([self._task(segments=[segment])])

    @pytest.mark.parametrize(
        "field",
        ["speaker", "text", "words", "metrics"],
    )
    def test_rejects_missing_prepared_segment_fields(self, field: str) -> None:
        segment = _valid_prepared_segment()
        segment.pop(field)

        with pytest.raises(RuntimeError, match=field):
            validate_tagging_outputs([self._task(segments=[segment])])

    def test_rejects_malformed_word_entries(self) -> None:
        segment = _valid_prepared_segment(words=[None])

        with pytest.raises(TypeError, match="word 0 must be a mapping"):
            validate_tagging_outputs([self._task(segments=[segment])])
