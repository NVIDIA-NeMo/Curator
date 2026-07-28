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

"""Tests for the pure-numpy SED postprocessing helpers."""

import numpy as np
import pytest

from nemo_curator.stages.audio.postprocessing.sed_utils import (
    AUDIOSET_CLASS_NAMES,
    SPEECH_CLASS_INDICES,
    SUPERCLASS_GROUPS,
    _hysteresis_threshold,
    _mask_to_segments,
    _merge_segments,
    aggregate_speech_probs,
    framewise_to_events,
)

_FPS = 50.0


def _curve(*spans: tuple[int, int], length: int = 100, high: float = 0.9, low: float = 0.1) -> np.ndarray:
    """A probability curve that is `high` inside each [start, stop) span."""
    probs = np.full(length, low, dtype=np.float32)
    for start, stop in spans:
        probs[start:stop] = high
    return probs


# ----------------------------------------------------------------------
# Mask -> segments
# ----------------------------------------------------------------------


def test_mask_to_segments_finds_each_contiguous_run() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert _mask_to_segments(mask) == [(1, 3), (5, 6)]


def test_mask_to_segments_handles_a_run_touching_both_edges() -> None:
    assert _mask_to_segments(np.ones(4, dtype=bool)) == [(0, 4)]


def test_mask_to_segments_returns_nothing_for_an_all_false_mask() -> None:
    assert _mask_to_segments(np.zeros(4, dtype=bool)) == []


# ----------------------------------------------------------------------
# Segment merging
# ----------------------------------------------------------------------


def test_merge_segments_joins_runs_closer_than_the_gap() -> None:
    assert _merge_segments([(0, 10), (12, 20)], max_gap=5) == [(0, 20)]


def test_merge_segments_keeps_runs_further_apart_than_the_gap() -> None:
    assert _merge_segments([(0, 10), (30, 40)], max_gap=5) == [(0, 10), (30, 40)]


def test_merge_segments_is_a_noop_on_an_empty_list() -> None:
    assert _merge_segments([], max_gap=5) == []


# ----------------------------------------------------------------------
# Hysteresis
# ----------------------------------------------------------------------


def test_hysteresis_needs_the_high_threshold_to_open_a_segment() -> None:
    """A curve that never reaches `high` produces nothing, even above `low`."""
    probs = np.full(10, 0.5, dtype=np.float32)
    assert not _hysteresis_threshold(probs, low=0.3, high=0.8).any()


def test_hysteresis_holds_a_segment_open_through_a_dip_above_low() -> None:
    probs = np.array([0.9, 0.5, 0.9], dtype=np.float32)
    assert _hysteresis_threshold(probs, low=0.3, high=0.8).tolist() == [True, True, True]


def test_hysteresis_closes_a_segment_below_low() -> None:
    probs = np.array([0.9, 0.1, 0.9], dtype=np.float32)
    assert _hysteresis_threshold(probs, low=0.3, high=0.8).tolist() == [True, False, True]


# ----------------------------------------------------------------------
# Aggregation across classes
# ----------------------------------------------------------------------


def test_noisy_or_exceeds_every_individual_probability() -> None:
    framewise = np.array([[0.5, 0.5]], dtype=np.float32)
    # 1 - (0.5 * 0.5) = 0.75
    assert aggregate_speech_probs(framewise, [0, 1], mode="noisy_or") == pytest.approx([0.75])


def test_max_aggregation_takes_the_strongest_class() -> None:
    framewise = np.array([[0.2, 0.7]], dtype=np.float32)
    assert aggregate_speech_probs(framewise, [0, 1], mode="max") == pytest.approx([0.7])


def test_mean_aggregation_averages_the_classes() -> None:
    framewise = np.array([[0.2, 0.8]], dtype=np.float32)
    assert aggregate_speech_probs(framewise, [0, 1], mode="mean") == pytest.approx([0.5])


def test_aggregation_defaults_to_the_speech_class_indices() -> None:
    framewise = np.zeros((3, 527), dtype=np.float32)
    framewise[:, SPEECH_CLASS_INDICES[0]] = 1.0
    assert aggregate_speech_probs(framewise) == pytest.approx([1.0, 1.0, 1.0])


# ----------------------------------------------------------------------
# Framewise -> events
# ----------------------------------------------------------------------


def test_events_carry_frame_boundaries_converted_to_seconds() -> None:
    events = framewise_to_events(_curve((50, 75)), fps=_FPS, threshold=0.5)
    assert len(events) == 1
    assert events[0]["start_time"] == pytest.approx(1.0)
    assert events[0]["end_time"] == pytest.approx(1.5)


def test_events_report_mean_and_max_confidence_over_the_span() -> None:
    probs = np.array([0.1, 0.6, 1.0, 0.8, 0.1], dtype=np.float32)
    (event,) = framewise_to_events(probs, fps=_FPS, threshold=0.5)
    assert event["max_confidence"] == pytest.approx(1.0)
    assert event["mean_confidence"] == pytest.approx(0.8, abs=1e-6)


def test_a_flat_curve_below_threshold_yields_no_events() -> None:
    assert framewise_to_events(np.full(100, 0.1, dtype=np.float32), fps=_FPS, threshold=0.5) == []


def test_min_duration_drops_events_that_are_too_short() -> None:
    # 5 frames at 50 fps = 0.1 s, under the 0.3 s floor.
    assert framewise_to_events(_curve((10, 15)), fps=_FPS, threshold=0.5, min_duration_sec=0.3) == []


def test_min_duration_keeps_events_at_or_over_the_floor() -> None:
    assert len(framewise_to_events(_curve((10, 40)), fps=_FPS, threshold=0.5, min_duration_sec=0.3)) == 1


def test_two_separated_spans_produce_two_events() -> None:
    assert len(framewise_to_events(_curve((10, 30), (60, 80)), fps=_FPS, threshold=0.5)) == 2


def test_merge_gap_fuses_two_nearby_spans_into_one_event() -> None:
    events = framewise_to_events(_curve((10, 30), (35, 60)), fps=_FPS, threshold=0.5, merge_gap_sec=0.2)
    assert len(events) == 1
    assert events[0]["start_time"] == pytest.approx(0.2)
    assert events[0]["end_time"] == pytest.approx(1.2)


def test_hysteresis_thresholds_take_priority_over_the_simple_threshold() -> None:
    """Only the span that crosses `high` opens, even though both clear `threshold`."""
    probs = _curve((10, 20), high=0.6)
    probs[40:50] = 0.95
    events = framewise_to_events(probs, fps=_FPS, threshold=0.5, hysteresis_low=0.7, hysteresis_high=0.9)
    assert len(events) == 1
    assert events[0]["start_time"] == pytest.approx(0.8)


def test_a_lone_high_threshold_is_ignored_without_its_low_partner() -> None:
    """Hysteresis needs both bounds; one alone falls back to the simple threshold."""
    events = framewise_to_events(_curve((10, 30), high=0.6), fps=_FPS, threshold=0.5, hysteresis_high=0.9)
    assert len(events) == 1


def test_median_smoothing_removes_an_isolated_spike() -> None:
    pytest.importorskip("scipy")
    probs = np.full(100, 0.1, dtype=np.float32)
    probs[50] = 0.99
    assert framewise_to_events(probs, fps=_FPS, threshold=0.5, smoothing_window_frames=5) == []


def test_median_smoothing_preserves_a_genuine_event() -> None:
    pytest.importorskip("scipy")
    events = framewise_to_events(_curve((30, 70)), fps=_FPS, threshold=0.5, smoothing_window_frames=5)
    assert len(events) == 1


# ----------------------------------------------------------------------
# Class metadata
# ----------------------------------------------------------------------


def test_superclass_groups_are_non_empty() -> None:
    assert SUPERCLASS_GROUPS
    assert all(indices for indices in SUPERCLASS_GROUPS.values())


def test_every_grouped_class_index_is_within_the_audioset_label_space() -> None:
    for indices in SUPERCLASS_GROUPS.values():
        assert all(0 <= idx < 527 for idx in indices)


def test_grouped_class_indices_have_readable_names() -> None:
    """Names are what land on each event, so a missing one is user-visible."""
    for indices in SUPERCLASS_GROUPS.values():
        for idx in indices:
            assert idx in AUDIOSET_CLASS_NAMES, f"class index {idx} has no name"


def test_no_class_index_is_claimed_by_two_superclasses() -> None:
    """Overlap would emit duplicate events for one detection."""
    seen: dict[int, str] = {}
    for superclass, indices in SUPERCLASS_GROUPS.items():
        for idx in indices:
            assert idx not in seen, f"index {idx} in both {seen.get(idx)!r} and {superclass!r}"
            seen[idx] = superclass
