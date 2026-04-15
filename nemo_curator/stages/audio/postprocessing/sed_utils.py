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

"""Stateless utilities for SED postprocessing: framewise probabilities -> events.

Ported from ameister's tts_granary ``sound_event_detection/postprocessing/``.
All functions are pure numpy — no model or GPU dependencies.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# AudioSet speech-related class indices (superclass groups)
# ---------------------------------------------------------------------------

SPEECH_CLASS_INDICES: list[int] = [0, 1, 2, 3, 4, 5, 6]
"""AudioSet classes 0-6 correspond to Speech, Male/Female speech, Child speech, Conversation, Narration, Babbling."""

SUPERCLASS_GROUPS: dict[str, list[int]] = {
    "speech": [0, 1, 2, 3, 4, 5, 6],
    "music": [137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
    "natural_noise": [288, 289, 290, 291, 292, 293, 294, 295, 296, 297],
    "urban_noise": [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310],
    "animal": [73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
    "impulse_noise": [420, 421, 422, 423, 424, 425, 426],
    "indoor_noise": [350, 351, 352, 353, 354, 355],
    "background_noise": [493, 494, 495, 496, 497, 498],
    "vocal_nonverbal": [27, 28, 29, 30, 31, 32, 33, 34, 35],
}


# ---------------------------------------------------------------------------
# Speech probability aggregation
# ---------------------------------------------------------------------------


def aggregate_speech_probs(
    framewise: np.ndarray,
    speech_indices: list[int] | None = None,
    mode: str = "noisy_or",
) -> np.ndarray:
    """Aggregate per-frame probabilities across speech-related AudioSet classes.

    Args:
        framewise: (T, C) probability matrix.
        speech_indices: Class indices to aggregate. Defaults to SPEECH_CLASS_INDICES.
        mode: ``"noisy_or"`` (default), ``"max"``, or ``"mean"``.

    Returns:
        (T,) aggregated speech probability per frame.
    """
    if speech_indices is None:
        speech_indices = SPEECH_CLASS_INDICES
    probs = framewise[:, speech_indices]
    if mode == "max":
        return probs.max(axis=1)
    if mode == "mean":
        return probs.mean(axis=1)
    # noisy-or: P(at least one) = 1 - prod(1 - p_i)
    return 1.0 - np.prod(1.0 - probs, axis=1)


# ---------------------------------------------------------------------------
# Framewise -> events conversion
# ---------------------------------------------------------------------------


def framewise_to_events(
    probs: np.ndarray,
    fps: float,
    threshold: float = 0.5,
    min_duration_sec: float = 0.0,
    smoothing_window_frames: int = 0,
    hysteresis_low: float | None = None,
    hysteresis_high: float | None = None,
    merge_gap_sec: float = 0.0,
) -> list[dict]:
    """Convert a 1-D probability curve into a list of events.

    Args:
        probs: (T,) per-frame probability.
        fps: Frames per second (used to convert frame indices to seconds).
        threshold: Simple threshold when hysteresis is disabled.
        min_duration_sec: Drop events shorter than this.
        smoothing_window_frames: Median-filter window (0 = disabled).
        hysteresis_low: Low threshold for hysteresis (None = disabled).
        hysteresis_high: High threshold for hysteresis (None = disabled).
        merge_gap_sec: Merge events with gaps smaller than this (0 = disabled).

    Returns:
        List of dicts ``{start_time, end_time, mean_confidence, max_confidence}``.
    """
    if smoothing_window_frames > 0:
        try:
            from scipy.ndimage import median_filter
        except ImportError:
            pass
        else:
            probs = median_filter(probs, size=smoothing_window_frames)

    if hysteresis_low is not None and hysteresis_high is not None:
        mask = _hysteresis_threshold(probs, hysteresis_low, hysteresis_high)
    else:
        mask = probs >= threshold

    segments = _mask_to_segments(mask)

    if merge_gap_sec > 0:
        merge_gap_frames = int(merge_gap_sec * fps)
        segments = _merge_segments(segments, merge_gap_frames)

    events: list[dict] = []
    for start_frame, end_frame in segments:
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        duration = end_sec - start_sec
        if duration < min_duration_sec:
            continue
        seg_probs = probs[start_frame:end_frame]
        events.append({
            "start_time": round(start_sec, 4),
            "end_time": round(end_sec, 4),
            "mean_confidence": float(np.mean(seg_probs)),
            "max_confidence": float(np.max(seg_probs)),
        })
    return events


def _hysteresis_threshold(probs: np.ndarray, low: float, high: float) -> np.ndarray:
    """Dual-threshold hysteresis: enter above ``high``, exit below ``low``."""
    mask = np.zeros(len(probs), dtype=bool)
    active = False
    for i, p in enumerate(probs):
        if active:
            if p < low:
                active = False
            else:
                mask[i] = True
        elif p >= high:
            active = True
            mask[i] = True
    return mask


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert boolean mask to list of (start_frame, end_frame) pairs."""
    segments: list[tuple[int, int]] = []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        segments.append((int(s), int(e)))
    return segments


def _merge_segments(segments: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge consecutive segments separated by fewer than ``max_gap`` frames."""
    if not segments:
        return segments
    merged: list[tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
