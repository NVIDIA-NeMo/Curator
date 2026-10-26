# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest
from textgrid import IntervalTier, TextGrid

from tutorials.audio.vad_refinement.filter_segments_by_vad_agreement import (
    FilterConfig,
    bad_ratio_distribution,
    reference_speech_intervals,
    score_manifest,
    score_segment_frames,
)


def _write_textgrid(path: Path, intervals: list[tuple[float, float, str]], duration: float = 1.0) -> None:
    textgrid = TextGrid(minTime=0.0, maxTime=duration)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=duration)
    for start, end, word in intervals:
        tier.add(start, end, word)
    textgrid.append(tier)
    textgrid.write(str(path))


def test_bad_frame_requires_both_vads_to_disagree_with_mfa() -> None:
    metrics = score_segment_frames(
        {"offset": 0.0, "duration": 1.0},
        [(0.0, 0.5)],
        [(0.0, 0.25), (0.5, 0.75)],
        [(0.0, 0.25), (0.5, 0.75)],
        frame_duration=0.25,
    )

    assert metrics["total_frames"] == 4
    assert metrics["good_frames"] == 2
    assert metrics["bad_frames"] == 2
    assert metrics["good_ratio"] == pytest.approx(0.5)
    assert metrics["bad_ratio"] == pytest.approx(0.5)
    assert metrics["both_miss_speech_frames"] == 1
    assert metrics["both_false_alarm_frames"] == 1
    assert metrics["both_agree_frames"] == 2


def test_frame_is_good_when_either_vad_matches_mfa() -> None:
    metrics = score_segment_frames(
        {"offset": 0.0, "duration": 1.0},
        [(0.0, 0.5)],
        [(0.0, 0.5)],
        [(0.5, 1.0)],
        frame_duration=0.25,
    )

    assert metrics["good_frames"] == 4
    assert metrics["bad_frames"] == 0
    assert metrics["silero_only_agrees_frames"] == 4
    assert metrics["good_ratio"] == pytest.approx(1.0)
    assert metrics["bad_to_good_ratio"] == pytest.approx(0.0)
    assert metrics["good_to_bad_ratio"] is None


def test_event_intervals_are_set_to_silence_in_all_three_streams() -> None:
    metrics = score_segment_frames(
        {"offset": 0.0, "duration": 1.0},
        [(0.0, 1.0)],
        [],
        [],
        frame_duration=0.25,
        event_intervals=[(0.0, 0.5)],
    )

    assert metrics["event_excluded_frames"] == 2
    assert metrics["good_frames"] == 2
    assert metrics["bad_frames"] == 2
    assert metrics["both_miss_speech_frames"] == 2


def test_embedded_global_word_alignment_can_supply_reference() -> None:
    intervals = reference_speech_intervals(
        {
            "offset": 10.0,
            "duration": 2.0,
            "alignment": {
                "word": [
                    ["hello", 10.25, 0.5, None],
                    ["[Timestamp alignment GT labels]", 10.75, 1.0, None],
                ]
            },
        }
    )

    assert len(intervals) == 1
    assert intervals[0] == pytest.approx((0.25, 0.75))


def test_refined_textgrid_is_the_default_reference(tmp_path: Path) -> None:
    mfa_path = tmp_path / "mfa.TextGrid"
    refined_path = tmp_path / "refined.TextGrid"
    _write_textgrid(mfa_path, [(0.0, 0.5, "mfa")])
    _write_textgrid(refined_path, [(0.5, 1.0, "refined")])
    row = {
        "offset": 0.0,
        "duration": 1.0,
        "mfa_textgrid": str(mfa_path),
        "vad_refined_textgrid": str(refined_path),
    }

    refined = reference_speech_intervals(row)
    mfa = reference_speech_intervals(row, "mfa")
    assert len(refined) == 1
    assert len(mfa) == 1
    assert refined[0] == pytest.approx((0.5, 1.0))
    assert mfa[0] == pytest.approx((0.0, 0.5))


def test_bad_ratio_distribution_writes_bins_and_quantiles() -> None:
    distribution = bad_ratio_distribution(
        [
            {"status": "scored", "bad_ratio": 0.0},
            {"status": "scored", "bad_ratio": 0.05},
            {"status": "scored", "bad_ratio": 0.1},
            {"status": "scored", "bad_ratio": 1.0},
            {"status": "skipped_reference_unavailable"},
        ],
        bin_width=0.1,
    )

    assert distribution["count"] == 4
    assert distribution["quantiles"]["p50"] == pytest.approx(0.075)
    assert distribution["bins"][0]["count"] == 2
    assert distribution["bins"][1]["count"] == 1
    assert distribution["bins"][-1]["count"] == 1


def test_manifest_scoring_reads_legacy_audit_and_filters_by_bad_ratio(tmp_path: Path) -> None:
    textgrid_path = tmp_path / "segment.TextGrid"
    _write_textgrid(textgrid_path, [(0.0, 0.5, "hello")])
    manifest_path = tmp_path / "segments.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": "meeting-001",
                "speaker_id": "speaker-00",
                "segment_index": 0,
                "language": "en",
                "audio_filepath": "speaker.wav",
                "offset": 0.0,
                "duration": 1.0,
                "text": "hello",
                "mfa_textgrid": str(textgrid_path),
                "vad_refined_textgrid": str(textgrid_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "vad_boundaries.json"
    audit_path.write_text(
        json.dumps(
            {
                "speakers": {
                    "speaker-00": {
                        "silero_vad_raw_intervals": [[0.0, 0.25], [0.5, 0.75]],
                        "sortformer_vad_raw_intervals": [[0.0, 0.25], [0.5, 0.75]],
                        "sound_expert_event_intervals": [[0.25, 0.5]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    score = score_manifest(
        manifest_path,
        audit_path,
        FilterConfig(
            frame_duration=0.25,
            vad_variant="raw",
            max_bad_ratio=0.5,
            missing_reference_policy="error",
        ),
    )

    assert score.counts["scored_segments"] == 1
    assert score.counts["selected_segments"] == 1
    assert score.report_rows[0]["bad_frames"] == 2
    assert score.report_rows[0]["bad_ratio"] == pytest.approx(0.5)
    assert score.report_rows[0]["silero_interval_source"] == "silero_vad_raw_intervals"
    assert score.report_rows[0]["sortformer_interval_source"] == "sortformer_vad_raw_intervals"
    assert score.selected_rows[0]["vad_frame_agreement"]["bad_ratio"] == pytest.approx(0.5)

    event_score = score_manifest(
        manifest_path,
        audit_path,
        FilterConfig(
            frame_duration=0.25,
            vad_variant="raw_event_excluded",
            missing_reference_policy="error",
        ),
    )
    assert event_score.report_rows[0]["event_excluded_frames"] == 1
    assert event_score.report_rows[0]["bad_frames"] == 1
    assert event_score.report_rows[0]["event_interval_source"] == "sound_expert_event_intervals"
