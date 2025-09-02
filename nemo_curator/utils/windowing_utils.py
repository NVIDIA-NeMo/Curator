# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
import subprocess
from dataclasses import dataclass

import torch
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import InterpolationMode, v2  # type: ignore[import-untyped]

from nemo_curator.utils.decoder_utils import decode_video_cpu_frame_ids, get_avg_frame_rate, get_frame_count
from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file


@dataclass
class WindowFrameInfo:
    """Container for frame window information, storing start and end frame indices.

    This class represents a window of frames in a video, defined by its start and end frame positions.
    """

    start: int
    end: int


WINDOW_MIN_FRAMES = 4


def compute_windows(total_frames: int, window_size: int = 128, remainder_threshold: int = 64) -> list[WindowFrameInfo]:
    """Generate windows by splitting the video into segments of the specified size.

    Args:
        total_frames: total frames
        window_size: The size of each window in number of frames.
        remainder_threshold: The minimum number of frames required to create a new window from the remainder.

    Yields:
        Tuple of (start_frame, end_frame) representing each window.

    """
    if not total_frames or total_frames < WINDOW_MIN_FRAMES:
        return []
    if total_frames <= window_size:
        return [WindowFrameInfo(0, total_frames - 1)]
    # Calculate the number of full window_size windows
    num_full_windows = total_frames // window_size

    # Calculate the remainder frames after filling in window_size windows
    remainder = total_frames % window_size

    out: list[WindowFrameInfo] = []
    # Yield each full window
    for i in range(num_full_windows):
        start_frame = i * window_size
        end_frame = start_frame + window_size - 1
        out.append(WindowFrameInfo(start_frame, end_frame))

    # Handle the remainder
    if remainder >= remainder_threshold:
        out.append(WindowFrameInfo(total_frames - remainder, total_frames - 1))
    elif remainder > 0 and num_full_windows > 0:
        # Expand the last window with the remainder if it exists
        out[-1] = WindowFrameInfo(out[-1].start, total_frames - 1)
    return out


def split_video_into_windows(  # noqa: PLR0913
    mp4_bytes: bytes,
    window_size: int = 256,
    remainder_threshold: int = 128,
    sampling_fps: float = 2.0,
    *,
    model_does_preprocess: bool = False,
    preprocess_dtype: str = "uint8",
    flip_input: bool = False,
    num_frames_to_use: int = 0,
    return_bytes: bool = False,
    return_video_frames: bool = True,
    num_threads: int = 1,
) -> tuple[list[bytes], list[torch.Tensor | None], list[WindowFrameInfo]]:
    """Calculate windows and return video inputs for language model from input clips.

    Processes video to determine the windows for a clip, decode in one shot and return processed frames
    for each window in a format suitable for consumption by the Qwen model.

    Args:
        mp4_bytes: input video in bytes
        fps: Frames per second of the input video.
        preprocess_dtype: Data type to use for preprocessing the video/image inputs.
        num_frames_to_use: Number of frames to extract from the video. If 0, uses all frames.
        flip_input: Whether to flip the input video/image horizontally.
        return_bytes: Whether to extract mp4 bytes for each window for use by PreviewStage
        model_does_preprocess: if the model does preprocessing
        num_threads: number of threads
        remainder_threshold: threshold for remainder
        return_video_frames: whether to return video frames
        sampling_fps: sampling fps
        window_size: window size

    Returns:
        Tuple containing:
            - "window_mp4_bytes": mp4 bytes corresponding to each window - only used when Preview stage is enabled
            - "window_frames": Decoded and per-window processed frames ready for use by Qwen model
            - "window info": start and end frame indices for each window in a clip

    """
    with make_pipeline_named_temporary_file(sub_dir="windowing") as input_file:
        input_file.write_bytes(mp4_bytes)
        total_frames = get_frame_count(mp4_bytes)
        windows = compute_windows(total_frames, window_size, remainder_threshold)
        video_frames: list[torch.Tensor | None] = []
        mp4_bytes_list: list[bytes] = []

        if not windows:
            return mp4_bytes_list, video_frames, windows

        if return_video_frames:
            video, frame_counts = fetch_video(
                str(input_file),
                sampling_fps=sampling_fps,
                window_range=windows,
                do_preprocess=not model_does_preprocess,
                preprocess_dtype=preprocess_dtype,
                num_frames_to_use=num_frames_to_use,
                flip_input=flip_input,
            )

            index = 0
            for count in frame_counts:
                video_frames.append(video[index : index + count - 1])
                index = count

        if return_bytes:
            if len(windows) == 1:
                return [mp4_bytes], video_frames, windows

            for window in windows:
                with make_pipeline_named_temporary_file(sub_dir="windowing") as tmp_file:
                    # Use ffmpeg to split the file on the frames.
                    command = [
                        "ffmpeg",
                        "-threads",
                        str(num_threads),
                        "-y",
                        "-i",
                        str(input_file),
                        "-loglevel",
                        "error",
                        "-vf",
                        f"select='between(n\\,{window.start}\\,{window.end})',setpts=PTS-STARTPTS",
                        "-threads",
                        str(num_threads),
                        "-f",
                        "mp4",
                        "-an",
                        str(tmp_file),
                    ]
                    subprocess.check_call(command)  # noqa: S603
                    mp4_bytes_list.append(tmp_file.read_bytes())
        return mp4_bytes_list, video_frames, windows


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def round_by_factor(number: float, factor: int) -> int:
    """Return the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Return the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Return the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Rescales the image so that the following conditions are met.

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        error_msg = (
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
        raise ValueError(error_msg)
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_nframes(
    fps: float,
    total_frames: int,
    video_fps: float,
) -> int:
    """Calculate the number of frames for video used for model inputs."""
    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * fps
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = round_by_factor(nframes, FRAME_FACTOR)

    if not (nframes >= FRAME_FACTOR and nframes <= total_frames):
        error_msg = f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        raise ValueError(error_msg)
    return nframes


def read_video_cpu(
    video_path: str,
    fps: float,
    num_frames_to_use: int,
    window_range: list[WindowFrameInfo],
) -> tuple[torch.Tensor, list[int]]:
    """Read video using PyAv.

    Args:
        video_path: path to the video support "file://", "http://", "https://" and local path.
        fps: frames per second
        num_frames_to_use: number of frames to use
        window_range: window range

    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).

    """
    video_fps = get_avg_frame_rate(video_path)
    idx_list = []
    frame_counts = []
    for window_frame_info in window_range:
        total_frames = window_frame_info.end - window_frame_info.start
        if num_frames_to_use > 0 and num_frames_to_use < total_frames:
            total_frames = num_frames_to_use
            end_frame_idx = window_frame_info.start + num_frames_to_use
        else:
            end_frame_idx = window_frame_info.end
        nframes = smart_nframes(fps, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(window_frame_info.start, end_frame_idx - 1, nframes).round().long().tolist()
        idx_list.extend(idx)
        frame_counts.append(nframes)

    video = decode_video_cpu_frame_ids(video_path, idx_list)
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video, frame_counts


def fetch_video(  # noqa: C901, PLR0911, PLR0913
    video_path: str,
    sampling_fps: float = 2.0,
    window_range: list[WindowFrameInfo] | None = None,
    *,
    do_preprocess: bool = False,
    preprocess_dtype: str = "float32",
    num_frames_to_use: int = 0,
    flip_input: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Load and preprocess video frames from a file.

    Args:
        video_path: Path to the video file.
        sampling_fps: Target frames per second for sampling.
        window_range: List of frame windows to extract.
        do_preprocess: Whether to preprocess the frames.
        preprocess_dtype: Data type for preprocessing.
        num_frames_to_use: Number of frames to extract (0 for all).
        flip_input: Whether to flip frames horizontally.

    Returns:
        Tuple of (processed frames tensor, frame indices).

    """
    if window_range is None:
        window_range = []
    video, frame_counts = read_video_cpu(
        video_path,
        sampling_fps,
        num_frames_to_use,
        window_range,
    )
    nframes, _, height, width = video.shape

    max_pixels = max(
        min(VIDEO_MAX_PIXELS, int(VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR)),
        int(VIDEO_MIN_PIXELS * 1.05),
    )
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )

    if do_preprocess:
        try:
            desired_dtype = getattr(torch, preprocess_dtype)
        except AttributeError:
            desired_dtype = torch.float32  # Fallback to default dtype

        if preprocess_dtype == "uint8":
            desired_dtype = torch.bfloat16

        resize_norm_transform = v2.Compose(
            [
                v2.Resize(
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.ToDtype(desired_dtype, scale=True),
                v2.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ],
        )
        video = resize_norm_transform(video)
        if flip_input:
            # Flip along the width and height dims
            # TODO: is this a bug?
            video = [torch.flip(frame, dims=[1, 2]) for frame in video]  # type: ignore[assignment]
        if preprocess_dtype == "uint8":
            return video.to(torch.uint8), frame_counts
        return video, frame_counts
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    if flip_input:
        video = [torch.flip(frame, dims=[1, 2]) for frame in video]  # type: ignore[assignment]
    if preprocess_dtype == "float32":
        return video.float(), frame_counts
    if preprocess_dtype == "float16":
        return video.half(), frame_counts
    if preprocess_dtype == "bfloat16":
        return video.to(torch.bfloat16), frame_counts
    if preprocess_dtype == "uint8":
        return video.to(torch.uint8), frame_counts
    return video, frame_counts  # is video a list ? TODO: XXX
