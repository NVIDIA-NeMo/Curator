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

import pytest

from nemo_curator.stages.audio.model_input_segmentation import (
    duration_to_num_samples,
    plan_audio_segments,
    resolve_max_model_input_duration,
)


def test_resolve_max_model_input_duration_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="max_inference_duration_s must be > 0"):
        resolve_max_model_input_duration(max_duration_s=0, owner="test")


def test_duration_to_num_samples_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate must be > 0"):
        duration_to_num_samples(10.0, 0)


def test_plan_audio_segments_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate must be > 0"):
        plan_audio_segments(num_samples=100, sample_rate=0, max_duration_s=10.0, owner="test")


def test_plan_audio_segments_keeps_zero_sample_inputs_representable() -> None:
    segments = plan_audio_segments(num_samples=0, sample_rate=16000, max_duration_s=30.0, owner="test")

    assert len(segments) == 1
    assert segments[0].index == 0
    assert segments[0].count == 1
    assert segments[0].start_sample == 0
    assert segments[0].stop_sample == 0
    assert segments[0].duration_s == 0.0


def test_plan_audio_segments_handles_non_divisible_final_segment() -> None:
    segments = plan_audio_segments(num_samples=95, sample_rate=10, max_duration_s=3.0, owner="test")

    assert [(segment.index, segment.count, segment.start_sample, segment.stop_sample) for segment in segments] == [
        (0, 4, 0, 30),
        (1, 4, 30, 60),
        (2, 4, 60, 90),
        (3, 4, 90, 95),
    ]
    assert [segment.duration_s for segment in segments] == [3.0, 3.0, 3.0, 0.5]


def test_plan_audio_segments_exact_boundary_has_no_empty_tail() -> None:
    segments = plan_audio_segments(num_samples=60, sample_rate=10, max_duration_s=3.0, owner="test")

    assert [(segment.index, segment.count, segment.start_sample, segment.stop_sample) for segment in segments] == [
        (0, 2, 0, 30),
        (1, 2, 30, 60),
    ]
    assert [segment.duration_s for segment in segments] == [3.0, 3.0]


def test_plan_audio_segments_qwen_2400s_boundary_at_16khz() -> None:
    sample_rate = 16000
    max_duration_s = 2400.0
    boundary_samples = int(sample_rate * max_duration_s)

    exact = plan_audio_segments(
        num_samples=boundary_samples,
        sample_rate=sample_rate,
        max_duration_s=max_duration_s,
        owner="ASRStage",
    )
    just_over = plan_audio_segments(
        num_samples=boundary_samples + 1,
        sample_rate=sample_rate,
        max_duration_s=max_duration_s,
        owner="ASRStage",
    )

    assert [(segment.index, segment.count, segment.start_sample, segment.stop_sample) for segment in exact] == [
        (0, 1, 0, boundary_samples),
    ]
    assert exact[0].duration_s == max_duration_s
    assert [(segment.index, segment.count, segment.start_sample, segment.stop_sample) for segment in just_over] == [
        (0, 2, 0, boundary_samples),
        (1, 2, boundary_samples, boundary_samples + 1),
    ]
    assert just_over[0].duration_s == max_duration_s
    assert just_over[1].duration_s == 1.0 / sample_rate
