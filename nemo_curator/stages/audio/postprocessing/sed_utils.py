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

Ported from ameister's hifi_granary ``sound_event_detection/postprocessing/``.
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

AUDIOSET_CLASS_NAMES: dict[int, str] = {
    # speech
    0: "speech",
    1: "male_speech",
    2: "female_speech",
    3: "child_speech",
    4: "conversation",
    5: "narration",
    6: "babbling",
    # vocal_nonverbal
    27: "laughter",
    28: "baby_laughter",
    29: "giggle",
    30: "snicker",
    31: "belly_laugh",
    32: "chuckle",
    33: "crying",
    34: "baby_cry",
    35: "whimper",
    # animal
    73: "animal",
    74: "domestic_animals",
    75: "dog",
    76: "bark",
    77: "howl",
    78: "bow_wow",
    79: "growling",
    80: "whimper_dog",
    81: "cat",
    82: "purr",
    # music
    137: "music",
    138: "musical_instrument",
    139: "plucked_string_instrument",
    140: "guitar",
    141: "electric_guitar",
    142: "bass_guitar",
    143: "acoustic_guitar",
    144: "steel_guitar",
    145: "tapping_guitar",
    146: "strum",
    147: "banjo",
    148: "sitar",
    149: "mandolin",
    150: "zither",
    # natural_noise
    288: "rain",
    289: "raindrop",
    290: "rain_on_surface",
    291: "thunder",
    292: "thunderstorm",
    293: "wind",
    294: "rustling_leaves",
    295: "wind_noise_microphone",
    296: "gusts",
    297: "waves_surf",
    # urban_noise
    300: "vehicle",
    301: "boat",
    302: "sailboat",
    303: "rowboat_canoe_kayak",
    304: "motorboat_speedboat",
    305: "ship",
    306: "motor_vehicle_road",
    307: "car",
    308: "vehicle_horn",
    309: "toot",
    310: "car_alarm",
    # indoor_noise
    350: "dishes_pots_pans",
    351: "cutlery_silverware",
    352: "chopping_food",
    353: "frying_food",
    354: "microwave_oven",
    355: "blender",
    # impulse_noise
    420: "gunshot",
    421: "machine_gun",
    422: "fusillade",
    423: "artillery_fire",
    424: "cap_gun",
    425: "fireworks",
    426: "firecracker",
    # background_noise
    493: "static",
    494: "mains_hum",
    495: "radio_channel_noise",
    496: "pink_noise",
    497: "tick",
    498: "pop",
}
"""Human-readable names for every AudioSet class index used in SUPERCLASS_GROUPS."""


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