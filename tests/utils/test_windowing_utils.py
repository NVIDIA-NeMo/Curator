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

from unittest.mock import patch

import torch

from nemo_curator.utils import windowing_utils


def test_split_video_into_windows_preserves_all_sampled_frames_per_window() -> None:
    video = torch.arange(12).reshape(12, 1, 1, 1)

    with (
        patch("nemo_curator.utils.windowing_utils.get_frame_count", return_value=12),
        patch("nemo_curator.utils.windowing_utils.fetch_video", return_value=(video, [5, 5, 2])),
    ):
        _, window_frames, windows = windowing_utils.split_video_into_windows(
            b"video-bytes",
            window_size=5,
            remainder_threshold=1,
            return_bytes=False,
        )

    assert [(window.start, window.end) for window in windows] == [(0, 4), (5, 9), (10, 11)]
    assert [frames.flatten().tolist() for frames in window_frames] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11],
    ]


def test_read_video_cpu_samples_inclusive_window_end() -> None:
    captured_indices = []

    def decode_video_cpu_frame_ids(_video_path: str, indices: list[int]) -> list[list[list[list[int]]]]:
        captured_indices.extend(indices)
        return [[[[idx]]] for idx in indices]

    with (
        patch("nemo_curator.utils.windowing_utils.get_avg_frame_rate", return_value=2.0),
        patch("nemo_curator.utils.windowing_utils.decode_video_cpu_frame_ids", decode_video_cpu_frame_ids),
    ):
        video, frame_counts = windowing_utils.read_video_cpu(
            "video.mp4",
            fps=2.0,
            num_frames_to_use=0,
            window_range=[windowing_utils.WindowFrameInfo(0, 3)],
        )

    assert frame_counts == [4]
    assert captured_indices == [0, 1, 2, 3]
    assert video.flatten().tolist() == [0, 1, 2, 3]


def test_read_video_cpu_respects_num_frames_to_use() -> None:
    captured_indices = []

    def decode_video_cpu_frame_ids(_video_path: str, indices: list[int]) -> list[list[list[list[int]]]]:
        captured_indices.extend(indices)
        return [[[[idx]]] for idx in indices]

    with (
        patch("nemo_curator.utils.windowing_utils.get_avg_frame_rate", return_value=2.0),
        patch("nemo_curator.utils.windowing_utils.decode_video_cpu_frame_ids", decode_video_cpu_frame_ids),
    ):
        video, frame_counts = windowing_utils.read_video_cpu(
            "video.mp4",
            fps=2.0,
            num_frames_to_use=4,
            window_range=[windowing_utils.WindowFrameInfo(10, 19)],
        )

    assert frame_counts == [4]
    assert captured_indices == [10, 11, 12, 13]
    assert video.flatten().tolist() == [10, 11, 12, 13]


def test_fetch_video_flips_and_returns_passthrough_dtype() -> None:
    video = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.uint8)

    with patch("nemo_curator.utils.windowing_utils.read_video_cpu", return_value=(video, [1])):
        result, frame_counts = windowing_utils.fetch_video(
            "video.mp4",
            preprocess_dtype="passthrough",
            flip_input=True,
        )

    expected = torch.tensor([[[[6, 5, 4], [3, 2, 1]]]], dtype=torch.uint8)
    assert frame_counts == [1]
    assert torch.equal(result, expected)
