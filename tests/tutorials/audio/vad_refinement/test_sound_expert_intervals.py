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

import pytest

from tutorials.audio.vad_refinement.sound_expert_intervals import (
    SoundExpertConfig,
    event_probabilities_to_prediction,
    subtract_intervals,
)


def test_event_probabilities_become_padded_intervals() -> None:
    config = SoundExpertConfig(
        model="example-model",
        threshold=0.6,
        frame_pad_sec=0.05,
        merge_gap_sec=0.0,
        event_tokens=("<ev:breath>", "<ev:laughter>"),
    )

    prediction = event_probabilities_to_prediction(
        [
            [0.1, 0.1],
            [0.7, 0.2],
            [0.8, 0.1],
            [0.1, 0.9],
            [0.1, 0.1],
        ],
        config.event_tokens,
        frame_duration=0.1,
        audio_duration=0.5,
        config=config,
    )

    assert prediction.by_event is not None
    assert prediction.by_event["<ev:breath>"][0] == pytest.approx((0.05, 0.35))
    assert prediction.by_event["<ev:laughter>"][0] == pytest.approx((0.25, 0.45))
    assert len(prediction.intervals) == 1
    assert prediction.intervals[0] == pytest.approx((0.05, 0.45))


def test_subtract_intervals_splits_speech_and_drops_short_residuals() -> None:
    result = subtract_intervals(
        [(0.0, 3.0)],
        [(0.8, 1.2), (2.95, 3.0)],
        minimum_duration=0.1,
    )

    assert result == pytest.approx([(0.0, 0.8), (1.2, 2.95)])
