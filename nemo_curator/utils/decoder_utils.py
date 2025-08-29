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

import enum
import io
import json
import subprocess
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    NamedTuple,
    cast,
)

import av
import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from av.container import InputContainer

from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file


class Resolution(NamedTuple):
    """Container for video frame dimensions.

    This class stores the height and width of video frames as a named tuple.
    """

    height: int
    width: int


@dataclass
class VideoMetadata:
    """Metadata for video content including dimensions, timing, and codec information.

    This class stores essential video properties such as resolution, frame rate,
    duration, and encoding details.
    """

    height: int = None
    width: int = None
    fps: float = None
    num_frames: int = None
    video_codec: str = None
    pixel_format: str = None
    video_duration: float = None
    audio_codec: str = None
    bit_rate_k: int = None


class FrameExtractionPolicy(enum.Enum):
    """Policy for extracting frames from video content.

    This enum defines different strategies for selecting frames from a video,
    including first frame, middle frame, last frame, or a sequence of frames.
    """

    first = 0
    middle = 1
    last = 2
    sequence = 3


class FramePurpose(enum.Enum):
    """Purpose for extracting frames from video content.

    This enum defines different purposes for extracting frames from a video,
    including aesthetics and embeddings.
    """

    AESTHETICS = 1
    EMBEDDINGS = 2


@dataclass
class FrameExtractionSignature:
    """Configuration for frame extraction parameters.

    This class combines extraction policy and target frame rate into a single signature
    that can be used to identify and reproduce frame extraction settings.
    """

    extraction_policy: FrameExtractionPolicy
    target_fps: float

    def to_str(self) -> str:
        """Convert frame extraction signature to string format.

        Returns:
            String representation of extraction policy and target FPS.

        """
        return f"{self.extraction_policy!s}-{int(self.target_fps * 1000)}"


def extract_video_metadata(video: str | bytes) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe.

    Args:
        video: Path to video file or video data as bytes.

    Returns:
        VideoMetadata object containing video properties.

    """
    inp = None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
    ]
    with make_pipeline_named_temporary_file(sub_dir="extract_video_metadata") as video_path:
        if isinstance(video, bytes):
            video_path.write_bytes(video)
            real_video_path = video_path
        else:
            real_video_path = Path(str(video))
        if not real_video_path.exists():
            error_msg = f"{real_video_path} not found!"
            raise FileNotFoundError(error_msg)
        cmd.append(real_video_path.as_posix())
        result = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)  # noqa: UP022, S603
        video_info = json.loads(result.stdout)

    video_stream, audio_codec = None, None
    for stream in video_info["streams"]:
        if stream["codec_type"] == "video":
            video_stream = stream
        elif stream["codec_type"] == "audio":
            audio_codec = stream["codec_name"]
    if not video_stream:
        error_msg = "No video stream found!"
        raise ValueError(error_msg)

    # Convert avg_frame_rate to float
    num, denom = map(int, video_stream["avg_frame_rate"].split("/"))
    fps = num / denom

    # not all formats store duration at stream level, so fallback to format container
    if "duration" in video_stream:
        video_duration = float(video_stream["duration"])
    elif "format" in video_info and "duration" in video_info["format"]:
        video_duration = float(video_info["format"]["duration"])
    else:
        error_msg = "Could not find `duration` in video metadata."
        raise KeyError(error_msg)
    num_frames = int(video_duration * fps)

    # store bit_rate if available
    bit_rate_k = 2000  # default to 2000K (2M) bit rate
    if "bit_rate" in video_stream:
        bit_rate_k = int(int(video_stream["bit_rate"]) / 1024)

    return VideoMetadata(
        height=video_stream["height"],
        width=video_stream["width"],
        fps=fps,
        num_frames=num_frames,
        video_codec=video_stream["codec_name"],
        pixel_format=video_stream["pix_fmt"],
        audio_codec=audio_codec,
        video_duration=video_duration,
        bit_rate_k=bit_rate_k,
    )


def _make_video_stream(data: Path | str | BinaryIO | bytes | io.BytesIO | io.BufferedReader) -> BinaryIO:
    """Convert various input types into a binary stream for video processing.

    This function handles different input types that could represent video
    data and converts them into a consistent BinaryIO interface that can be
    used for video processing operations.

    Args:
        data: The input video data, which can be one of:
            - Path: A path to a video file
            - bytes: Raw video data in bytes
            - io.BytesIO: An in-memory binary stream
            - io.BufferedReader: A buffered binary file reader
            - BinaryIO: Any binary stream

    Returns:
        BinaryIO: A binary stream containing the video data

    Raises:
        ValueError: If the input type is not one of the supported types

    """
    if isinstance(data, str):
        data = Path(data)

    if isinstance(data, Path):
        return data.open("rb")
    if isinstance(data, bytes):
        return io.BytesIO(data)
    if isinstance(data, (io.BytesIO, io.BufferedReader, BinaryIO)):
        return data

    error_msg = f"Invalid video type: {type(data)}"  # type: ignore[unreachable]
    raise ValueError(error_msg)


@contextmanager
def save_stream_position(stream: BinaryIO) -> Generator[BinaryIO, None, None]:
    """Context manager that saves and restores stream position."""
    pos = stream.tell()
    try:
        yield stream
    finally:
        stream.seek(pos)


def get_video_timestamps(
    data: Path | str | BinaryIO | bytes,
    stream_idx: int = 0,
    video_format: str | None = None,
) -> npt.NDArray[np.float32]:
    """Get timestamps for all frames in a video stream.

    The file position will be moved as needed to get the timestamps.

    Note: the order that frames appear in a video stream is not necessarily
    the order that the frames will be displayed. This means that timestamps
    are not monotonically increasing within a video stream. This can happen
    when B-frames are present

    This function will return presentation timestamps in monotonically
    increasing order.

    Args:
        data: An open file, io.BytesIO, or bytes object with the video data.
        stream_idx: PyAv index of the video stream to decode, usually 0.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best

    Returns:
        A numpy array of monotonically increasing timestamps.

    """
    stream = _make_video_stream(data)

    timestamps: list[float] = []
    with av.open(stream, format=video_format) as container:
        video_stream = container.streams.video[stream_idx]

        time_base: float = 0.0
        if video_stream.time_base is not None:
            time_base = float(video_stream.time_base)
        else:
            error_msg = f"Time base is None for video stream {stream_idx}"
            raise ValueError(error_msg)

        # cast is safe because av.open returns an InputContainer
        for packet in cast("InputContainer", container).demux(video=stream_idx):
            if packet.pts is None:
                continue

            ts = float(packet.pts) * time_base
            timestamps.append(ts)

    return np.sort(np.array(timestamps, dtype=np.float32))


def find_closest_indices(src: npt.NDArray[np.float32], dst: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
    """Find the closest indices in src to each element in dst.

    If an element in dst is equidistant from two elements in src, the left
    index in src is used.

    Args:
        src: Monotonically increasing array of numbers to match dst against
        dst: Monotonically increasing array of numbers to search for in src

    Returns:
        Array of closest indices in src for each element in dst

    """
    # Rightmost indices are the insertion points into sorted array
    right_idx = np.searchsorted(src, dst)
    right_idx = np.clip(right_idx, 1, len(src) - 1)

    # leftmost elements now, becomes closest index later
    closest_idx = right_idx - 1

    # Compare distances to left and right neighbors
    left = src[closest_idx]
    right = src[right_idx]
    right_closest = np.abs(dst - right) < np.abs(dst - left)
    closest_idx[right_closest] = right_idx[right_closest]

    # Handle edge case for values beyond the last timestamp
    beyond_end = dst >= src[-1]
    closest_idx[beyond_end] = len(src) - 1

    return closest_idx


def sample_closest(  # noqa: PLR0913
    src: npt.NDArray[np.float32],
    sample_rate: float,
    start: float | None = None,
    stop: float | None = None,
    endpoint: bool = True,
    dedup: bool = True,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Sample `src` at `sample_rate` rate and return the closest indices.

    This function is meant to be used for sampling monotonically increasing
    numbers, like timestamps. This function can be used for synchronizing
    sensors, like multiple cameras, or synchronizing video with GPS and LIDAR.

    The first element sampled with either or src[0] or the element closest
    to `start`

    The last element sampled will either be src[-1] or the element closest
    to `stop`. The last element is only included if it both fits into the
    sampling rate and if endpoint=True

    This function intentionally has no policy about distance from the closest
    elements in src to the sample elements. It will return the index of the
    closest element to the sample. It is up to the caller to define policy,
    which is why sample_elements is returned.

    Args:
        src: Monotonically increasing array of elements
        sample_rate: Sampling rate
        start: Start element (defaults to first element)
        stop: End element (defaults to last element)
        endpoint: If True, `stop` can be the last sample, if it fits into
            the sample rate. If False, `stop` is not included in the output.
        dedup: Whether to deduplicate indices. Repeated indices will be
            reflected in the returned counts array.

    Returns:
        Tuple of (indices, counts) where counts[i] is the number of times
        indices[i] was sampled. The sample elements are also returned

    """
    if sample_rate <= 0:
        error_msg = f"Sample rate must be greater than 0, got {sample_rate=}"
        raise ValueError(error_msg)

    sample_interval = 1.0 / sample_rate
    _start = start if start is not None else src[0]
    _stop = stop if stop is not None else src[-1]

    if endpoint:
        # Add a small epsilon to the end element to ensure the last element
        # can be included, if it fits into the sampling scheme. Too large of an
        # epsilon will expand the element range too far, so use half of the
        # sample interval.
        _stop += sample_interval * 0.5

    sample_elements = np.arange(_start, _stop, sample_interval, dtype=np.float32)
    indices = find_closest_indices(src, sample_elements)

    if not endpoint and np.isclose(sample_elements[-1], _stop):
        indices = indices[:-1]
        sample_elements = sample_elements[:-1]

    if dedup:
        indices, counts = np.unique(indices, return_counts=True)
    else:
        counts = np.ones_like(indices)

    return indices, counts, sample_elements


def decode_video_cpu_frame_ids(  # noqa: PLR0913
    data: Path | str | BinaryIO | bytes,
    frame_ids: npt.NDArray[np.int32],
    counts: npt.NDArray[np.int32] | None = None,
    stream_idx: int = 0,
    video_format: str | None = None,
    num_threads: int = 1,
) -> npt.NDArray[np.uint8]:
    """Decode video using PyAV frame ids.

    It is not recommended to use this function directly. Instead, use
    `decode_video_cpu`, which is timestamp-based. Timestamps are necessary for
    synchronizing sensors, like multiple cameras, or synchronizing video with
    GPS and LIDAR.

    Args:
        data: An open file, io.BytesIO, or bytes object with the video data.
        frame_ids: List of frame ids to decode.
        counts: List of counts for each frame id. It is possible that a frame id
            is repeated during supersampling, which can happen in videos with
            frame drops, or just due to clock drift between sensors.
        stream_idx: PyAv index of the video stream to decode, usually 0.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best
        num_threads: Number of threads to use for decoding.

    Returns:
        A numpy array of shape (frame_count, height, width, channels) containing
        the decoded frames.

    """
    stream = _make_video_stream(data)

    _counts = counts
    if _counts is None:
        _counts = np.ones_like(frame_ids, dtype=np.int32)

    with av.open(stream, format=video_format) as container:
        container = cast("InputContainer", container)
        frame_iterator: Iterator[av.VideoFrame] = container.decode(video=stream_idx)

        video_stream = container.streams.video[stream_idx]
        video_stream.thread_type = 3
        video_stream.thread_count = num_threads

        width = video_stream.width
        height = video_stream.height
        channels = 3
        dest_format = "rgb24"

        # preallocate output frames tensor
        frame_count = _counts.sum()
        frames_np = np.empty((frame_count, height, width, channels), dtype=np.uint8)

        frame_id_iter = zip(frame_ids, list(_counts), strict=True)
        next_frame_id, count = next(frame_id_iter)

        frames_np_idx = 0

        for frame_id, frame in enumerate(frame_iterator):
            if frame_id == next_frame_id:
                frames_np[frames_np_idx : frames_np_idx + count, :, :, :] = np.broadcast_to(
                    frame.to_ndarray(format=dest_format),
                    (count, height, width, channels),
                )
                frames_np_idx += count

                try:
                    next_frame_id, count = next(frame_id_iter)
                except StopIteration:
                    # Stop after the last desired frame has been decoded.
                    break

    return frames_np


def get_avg_frame_rate(
    data: Path | str | BinaryIO | bytes,
    stream_idx: int = 0,
    video_format: str | None = None,
) -> float:
    """Get the average frame rate of a video.

    Args:
        data: An open file, io.BytesIO, or bytes object with the video data.
        stream_idx: Index of the video stream to decode, usually 0.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best

    Returns:
        The average frame rate of the video.

    """
    stream = _make_video_stream(data)
    with save_stream_position(stream), av.open(stream, format=video_format) as container:
        video_stream = container.streams.video[stream_idx]

        if video_stream.average_rate is not None:
            avg_frame_rate = float(video_stream.average_rate)
        else:
            # Fall back to calculating the average frame rate from the timestamps
            ts = get_video_timestamps(stream, stream_idx, video_format)
            num_frames = len(ts)

            if num_frames <= 1:
                error_msg = f"Not enough frames to get average frame rate {num_frames=}"
                raise ValueError(error_msg)

            if ts[-1] - ts[0] <= 0:
                error_msg = f"Invalid timestamps: {ts[-1]} - {ts[0]} <= 0"
                raise ValueError(error_msg)

            avg_frame_rate = (ts[-1] - ts[0]) / num_frames

    return avg_frame_rate


def decode_video_cpu(  # noqa: PLR0913
    data: Path | str | BinaryIO | bytes,
    sample_rate_fps: float,
    timestamps: npt.NDArray[np.float32] | None = None,
    start: float | None = None,
    stop: float | None = None,
    endpoint: bool = True,
    stream_idx: int = 0,
    video_format: str | None = None,
    num_threads: int = 1,
) -> npt.NDArray[np.uint8]:
    """Decode video frames from a binary stream using PyAV with configurable frame rate sampling.

    This function decodes video frames from a binary stream at a specified
    frame rate. The frame rate does not need to match the input video's frame
    rate. It is possible to supersample a video as well as undersample.

    Args:
        data: An open file, io.BytesIO, or bytes object with the video data.
        sample_rate_fps: Frame rate for sampling the video
        timestamps: Optional array of presentation timestamps for each frame
            in the video. If supplied, this array *must* be monotonically
            increasing. If not supplied, timestamps will be extracted from the
            video stream.
        start: Optional start timestamp for frame extraction. If None, the
            first frame timestamp is used.
        stop: Optional end timestamp for frame extraction. If None, the last
            frame timestamp is used.
        endpoint: If True, stop is the last sample. Otherwise, it is not included.
            Default is True.
        stream_idx: PyAv index of the video stream to decode, usually 0.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best
        num_threads: Number of threads to use for decoding.

    Returns:
        A numpy array of shape (num_frames, height, width, channels) containing the decoded
        frames in RGB24 format

    Raises:
        ValueError: If the sampled timestamps differ from source timestamps by more than
            the specified tolerance

    """
    stream = _make_video_stream(data)
    _timestamps = timestamps
    if _timestamps is None:
        with save_stream_position(stream):
            _timestamps = get_video_timestamps(stream, stream_idx, video_format)

    _start = _timestamps[0] if start is None else start
    _stop = _timestamps[-1] if stop is None else stop

    frame_ids, counts, _ = sample_closest(
        _timestamps,
        sample_rate=sample_rate_fps,
        start=_start,
        stop=_stop,
        endpoint=endpoint,
        dedup=True,
    )

    with save_stream_position(stream):
        return decode_video_cpu_frame_ids(
            stream,
            frame_ids,
            counts,
            stream_idx,
            num_threads=num_threads,
            video_format=video_format,
        )


def get_frame_count(data: Path | str | BinaryIO | bytes, stream_idx: int = 0, video_format: str | None = None) -> int:
    """Get the total number of frames in a video file or stream.

    Args:
        data: An open file, io.BytesIO, or bytes object with the video data.
        stream_idx: Index of the video stream to read from. Defaults to 0,
            which is typically the main video stream.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best

    Returns:
        The total number of frames in the video stream.

    """
    stream = _make_video_stream(data)
    with save_stream_position(stream), av.open(stream, format=video_format) as container:
        frame_count = container.streams.video[stream_idx].frames
        if frame_count is None:
            timestamps = get_video_timestamps(stream, stream_idx, video_format)  # type: ignore[unreachable]
            frame_count = len(timestamps)

    return frame_count


def extract_frames(  # noqa: PLR0913
    video: Path | str | BinaryIO | bytes,
    extraction_policy: FrameExtractionPolicy,
    sample_rate_fps: float = 1.0,
    target_res: tuple[int, int] = (-1, -1),
    num_threads: int = 1,
    stream_idx: int = 0,
    video_format: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Extract frames from a video into a numpy array.

    Args:
        video: An open file, io.BytesIO, or bytes object with the video data.
        extraction_policy: The policy for extracting frames.
        sample_rate_fps: Frame rate for sampling the video
        target_res: The target resolution for the frames.
        stream_idx: PyAv index of the video stream to decode, usually 0.
        video_format: Format of the video stream, like "mp4", "mkv", etc.
            None is probably best
        num_threads: Number of threads to use for decoding.

    Returns:
        A numpy array of shape (num_frames, height, width, 3) containing the decoded
        frames in RGB24 format

    """
    stream = _make_video_stream(video)

    with save_stream_position(stream):
        all_timestamps = get_video_timestamps(stream, stream_idx, video_format)

    if len(all_timestamps) == 0:
        error_msg = "Can't extract frames from empty video"
        raise ValueError(error_msg)

    if extraction_policy == FrameExtractionPolicy.sequence or len(all_timestamps) == 1:
        timestamps = all_timestamps
    elif extraction_policy == FrameExtractionPolicy.middle:
        num_ts = len(all_timestamps)
        idx = num_ts // 2 - 1 if num_ts % 2 == 0 else num_ts // 2
        timestamps = all_timestamps[idx : idx + 1]
    else:
        error_msg = "Extraction policies apart from Sequence and Middle not available yet"
        raise NotImplementedError(error_msg)

    frames = decode_video_cpu(
        stream,
        sample_rate_fps,
        timestamps=timestamps,
        endpoint=True,
        num_threads=num_threads,
        stream_idx=stream_idx,
        video_format=video_format,
    )

    if target_res[0] > 0 and target_res[1] > 0:
        interpolation = cv2.INTER_CUBIC
        frames = np.array(
            [cv2.resize(frame, (target_res[1], target_res[0]), interpolation=interpolation) for frame in frames]
        )

    return frames
