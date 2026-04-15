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

"""Minimal tests for SED postprocessing utilities.

Tests framewise -> events conversion with synthetic NPZ data.
Verifies: event detection at correct timestamps, threshold behaviour, merging.

Run: pytest tests/stages/audio/postprocessing/test_sed_postprocessing.py -v --noconftest
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Direct import (bypass nemo_curator __init__ chain for Py3.9)
# ---------------------------------------------------------------------------
def _import_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_utils_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "nemo_curator", "stages", "audio", "postprocessing", "sed_utils.py",
)
_sed_utils = _import_from_path("_test_sed_utils", os.path.abspath(_utils_path))

aggregate_speech_probs = _sed_utils.aggregate_speech_probs
framewise_to_events = _sed_utils.framewise_to_events
SPEECH_CLASS_INDICES = _sed_utils.SPEECH_CLASS_INDICES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS = 50.0  # 16kHz / 320 hop
CLASSES = 527


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_npz(tmp_path: Path, speech_region: tuple[int, int] = (100, 200), total_frames: int = 500) -> str:
    """Create a synthetic NPZ with one known speech blob.

    Speech classes (0-6) have prob 0.9 in the given frame range, 0.01 elsewhere.
    """
    fw = np.full((total_frames, CLASSES), 0.01, dtype=np.float32)
    start, end = speech_region
    fw[start:end, :7] = 0.9  # speech classes high
    npz_path = str(tmp_path / "test.npz")
    np.savez_compressed(
        npz_path,
        framewise=fw.astype(np.float16),
        fps=np.float32(FPS),
        valid_frames=np.int32(total_frames),
        audio_filepath="test.wav",
        was_padded=np.bool_(False),
    )
    return npz_path


# ---------------------------------------------------------------------------
# Tests: aggregate_speech_probs
# ---------------------------------------------------------------------------


class TestAggregateSpeechProbs:
    def test_noisy_or_high_for_speech(self) -> None:
        fw = np.zeros((10, CLASSES), dtype=np.float32)
        fw[:, :7] = 0.9
        probs = aggregate_speech_probs(fw, mode="noisy_or")
        assert probs.shape == (10,)
        assert probs.min() > 0.99  # noisy-or of 7 x 0.9 ~ 1.0

    def test_noisy_or_low_for_silence(self) -> None:
        fw = np.full((10, CLASSES), 0.01, dtype=np.float32)
        probs = aggregate_speech_probs(fw, mode="noisy_or")
        assert probs.max() < 0.1

    def test_max_mode(self) -> None:
        fw = np.zeros((5, CLASSES), dtype=np.float32)
        fw[2, 0] = 0.8
        probs = aggregate_speech_probs(fw, mode="max")
        assert probs[2] == pytest.approx(0.8)
        assert probs[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: framewise_to_events
# ---------------------------------------------------------------------------


class TestFramewiseToEvents:
    def test_one_event_at_correct_timestamps(self) -> None:
        """Synthetic NPZ with speech at frames 100-200 -> 1 event at 2.0-4.0s."""
        probs = np.zeros(500, dtype=np.float32)
        probs[100:200] = 0.9  # speech from frame 100 to 200
        events = framewise_to_events(probs, fps=FPS, threshold=0.5)
        assert len(events) == 1
        evt = events[0]
        assert evt["start_time"] == pytest.approx(100 / FPS, abs=0.02)
        assert evt["end_time"] == pytest.approx(200 / FPS, abs=0.02)
        assert evt["mean_confidence"] > 0.8

    def test_no_events_below_threshold(self) -> None:
        probs = np.full(100, 0.3, dtype=np.float32)
        events = framewise_to_events(probs, fps=FPS, threshold=0.5)
        assert len(events) == 0

    def test_min_duration_filters_short(self) -> None:
        probs = np.zeros(100, dtype=np.float32)
        probs[10:12] = 0.9  # only 2 frames = 0.04s
        events = framewise_to_events(probs, fps=FPS, threshold=0.5, min_duration_sec=0.1)
        assert len(events) == 0

    def test_merge_gap(self) -> None:
        probs = np.zeros(200, dtype=np.float32)
        probs[10:20] = 0.9
        probs[22:32] = 0.9  # gap of 2 frames
        events_no_merge = framewise_to_events(probs, fps=FPS, threshold=0.5, merge_gap_sec=0.0)
        events_merged = framewise_to_events(probs, fps=FPS, threshold=0.5, merge_gap_sec=0.1)
        assert len(events_no_merge) == 2
        assert len(events_merged) == 1

    def test_all_speech(self) -> None:
        probs = np.full(100, 0.9, dtype=np.float32)
        events = framewise_to_events(probs, fps=FPS, threshold=0.5)
        assert len(events) == 1
        assert events[0]["start_time"] == pytest.approx(0.0, abs=0.02)
        assert events[0]["end_time"] == pytest.approx(100 / FPS, abs=0.02)
