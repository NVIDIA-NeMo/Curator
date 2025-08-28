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

import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np
import numpy.typing as npt

from nemo_curator.utils.decoder_utils import extract_video_metadata

from .tasks import Task


@dataclass
class _Window:
    """Container for video window data including metadata, frames, and processing results.

    This class stores information about a video window, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.
    """

    # Start frame number of this window
    start_frame: int
    # End frame number of this window
    end_frame: int
    # MP4 bytes for this window
    mp4_bytes: bytes | None = None
    # Qwen LLM input for this window
    qwen_llm_input: dict[str, Any] | None = None
    # X1 model input for this window
    x1_input: Any | None = None
    # `caption: {model_name: caption}`
    caption: dict[str, str] = field(default_factory=dict)
    enhanced_caption: dict[str, str] = field(default_factory=dict)
    # webp preview
    webp_bytes: bytes | None = None

    def get_major_size(self) -> int:
        """Calculate total memory size of the window.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        total_size += len(self.mp4_bytes) if self.mp4_bytes else 0
        # TODO: this is probably inaccurate
        total_size += sys.getsizeof(self.qwen_llm_input) if self.qwen_llm_input else 0
        total_size += sys.getsizeof(self.caption)
        total_size += sys.getsizeof(self.enhanced_caption)
        total_size += len(self.webp_bytes) if self.webp_bytes else 0
        return total_size


@dataclass
class Clip:
    """Container for video clip data including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.
    """

    uuid: UUID
    source_video: str
    span: tuple[float, float]
    buffer: bytes | None = None
    extracted_frames: dict[str, npt.NDArray[np.uint8]] = field(default_factory=dict)
    # motion
    decoded_motion_data: None = None  # TODO: Add motion data type in the motion filter PR
    motion_score_global_mean: float | None = None
    motion_score_per_patch_min_256: float | None = None
    # aesthetic
    aesthetic_score: float | None = None
    # embedding
    cosmos_embed1_frames: npt.NDArray[np.float32] | None = None
    cosmos_embed1_embedding: npt.NDArray[np.float32] | None = None
    intern_video_2_frames: npt.NDArray[np.float32] | None = None
    intern_video_2_embedding: npt.NDArray[np.float32] | None = None
    # captioning
    windows: list[_Window] = field(default_factory=list)
    # egomotion
    egomotion: dict[str, bytes] = field(default_factory=dict)
    # for testing
    cosmos_embed1_text_match: tuple[str, float] | None = None
    intern_video_2_text_match: tuple[str, float] | None = None
    # for debugging
    errors: dict[str, str] = field(default_factory=dict)

    def extract_metadata(self) -> dict[str, Any] | None:
        """Extract metadata from the clip's buffer.

        Returns:
            A dictionary containing the extracted metadata (width, height, framerate,
            num_frames, video_codec, num_bytes) if buffer exists, None otherwise.

        Raises:
            Exception: Any exception from extract_video_metadata is propagated.

        """
        if self.buffer is None:
            return None

        metadata = extract_video_metadata(self.buffer)

        return {
            "width": metadata.width,
            "height": metadata.height,
            "framerate": metadata.fps,
            "num_frames": metadata.num_frames,
            "video_codec": metadata.video_codec,
            "num_bytes": len(self.buffer),
        }

    @property
    def duration(self) -> float:
        """Calculate the duration of the clip.

        Returns:
            Duration of the clip in seconds.

        """
        return self.span[1] - self.span[0]

    def get_major_size(self) -> int:
        """Calculate total memory size of the clip.

        Returns:
            Total size in bytes.

        """
        total_size = len(self.uuid.bytes)
        if self.buffer:
            total_size += len(self.buffer)
        if self.extracted_frames:
            for x in self.extracted_frames.values():
                total_size += x.nbytes
        if self.decoded_motion_data is not None:
            total_size += self.decoded_motion_data.get_major_size()
        if self.intern_video_2_frames is not None:
            total_size += self.intern_video_2_frames.nbytes
        if self.intern_video_2_embedding is not None:
            total_size += self.intern_video_2_embedding.nbytes
        for window in self.windows:
            total_size += window.get_major_size()
        return total_size


@dataclass
class ClipStats:
    """Statistics for video clips including filtering, transcoding, and captioning results.

    This class accumulates statistics about the number of clips processed through
    different stages of the video processing pipeline, including motion filtering,
    aesthetic filtering, and captioning.
    """

    num_filtered_by_motion: int = 0
    num_filtered_by_aesthetic: int = 0
    num_passed: int = 0
    num_transcoded: int = 0
    num_with_embeddings: int = 0
    num_with_caption: int = 0
    num_with_webp: int = 0
    total_clip_duration: float = 0.0
    max_clip_duration: float = 0.0

    def combine(self, other: "ClipStats") -> None:
        """Combine two ClipStats objects.

        Args:
            other: ClipStats object to combine with.

        """
        self.num_filtered_by_motion += other.num_filtered_by_motion
        self.num_filtered_by_aesthetic += other.num_filtered_by_aesthetic
        self.num_passed += other.num_passed
        self.num_transcoded += other.num_transcoded
        self.num_with_embeddings += other.num_with_embeddings
        self.num_with_caption += other.num_with_caption
        self.num_with_webp += other.num_with_webp
        self.total_clip_duration += other.total_clip_duration
        self.max_clip_duration = max(self.max_clip_duration, other.max_clip_duration)


@dataclass
class VideoMetadata:
    """Metadata for video content including dimensions, timing, and codec information.

    This class stores essential video properties such as resolution, frame rate,
    duration, and encoding details.
    """

    size: int | None = None
    height: int | None = None
    width: int | None = None
    framerate: float | None = None
    num_frames: int | None = None
    duration: float | None = None
    video_codec: str | None = None
    pixel_format: str | None = None
    audio_codec: str | None = None
    bit_rate_k: int | None = None


@dataclass
class Video:
    """
    Container for video content including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.
    """

    input_video: pathlib.Path
    source_bytes: bytes | None = None
    # video metadata
    metadata: VideoMetadata = field(default_factory=VideoMetadata)
    # transnetv2 decoded input
    frame_array: npt.NDArray[np.uint8] | None = None
    # clips
    clips: list[Clip] = field(default_factory=list)
    filtered_clips: list[Clip] = field(default_factory=list)
    # for cracking
    num_total_clips: int = 0
    num_clip_chunks: int = 0
    clip_chunk_index: int = 0
    # for last writer stage
    clip_stats: ClipStats = field(default_factory=ClipStats)
    # for debugging
    errors: dict[str, str] = field(default_factory=dict)

    def populate_metadata(self) -> None:
        """Extract and assign video metadata from source_bytes.

        This method extracts metadata from the video data in source_bytes and
        assigns it to self.metadata.

        Raises:
            ValueError: If source_bytes is None.
            Exception: Any exception from extract_video_metadata is propagated.

        """
        if self.source_bytes is None:
            error_msg = "No video data available: source_bytes is None"
            raise ValueError(error_msg)

        # Extract metadata using the existing function
        extracted_metadata = extract_video_metadata(self.source_bytes)

        # Set the size from source_bytes
        self.metadata.size = len(self.source_bytes)

        # Map the extracted metadata to our metadata object
        self.metadata.height = extracted_metadata.height
        self.metadata.width = extracted_metadata.width
        self.metadata.framerate = extracted_metadata.fps
        self.metadata.num_frames = extracted_metadata.num_frames
        self.metadata.duration = extracted_metadata.video_duration
        self.metadata.video_codec = extracted_metadata.video_codec
        self.metadata.pixel_format = extracted_metadata.pixel_format
        self.metadata.audio_codec = extracted_metadata.audio_codec
        self.metadata.bit_rate_k = extracted_metadata.bit_rate_k

    @property
    def fraction(self) -> float:
        """Calculate the fraction of processed clips.

        Returns:
            Fraction of processed clips.

        """
        if self.num_total_clips == 0:
            return 1.0
        return (len(self.clips) + len(self.filtered_clips)) / self.num_total_clips

    @property
    def weight(self) -> float:
        """Calculate the weight of the video.

        Returns:
            Weight of the video.

        """
        if self.metadata.size is None:
            return 0
        # normalize to 5 min
        if self.metadata.duration is None:
            msg = "metadata.duration is None"
            raise ValueError(msg)
        weight = self.metadata.duration / 300
        # when clips are further chunked
        return weight * self.fraction

    def get_major_size(self) -> int:
        """Calculate total memory size of the video.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        total_size += len(self.source_bytes) if self.source_bytes else 0
        total_size += sys.getsizeof(self.frame_array)
        for clip in self.clips:
            total_size += clip.get_major_size()
        total_size += self.frame_array.nbytes if self.frame_array is not None else 0
        return total_size

    def has_metadata(self) -> bool:
        """Check if all metadata fields are present.

        Returns:
            True if all metadata fields are present, False otherwise.

        """
        return all(
            [
                self.metadata.height,
                self.metadata.width,
                self.metadata.duration,
                self.metadata.framerate,
                self.metadata.num_frames,
                self.metadata.video_codec,
            ],
        )

    def is_10_bit_color(self) -> bool | None:
        """Heuristic function to determine if the input video has 10-bit color."""
        if self.metadata.pixel_format is None:
            return None
        return "10le" in self.metadata.pixel_format or "10be" in self.metadata.pixel_format

    @property
    def input_path(self) -> str:
        """Get the input path of the video.

        Returns:
            Input path of the video.

        """
        # TODO: Support StorageClient type input
        return self.input_video.as_posix()


@dataclass
class VideoTask(Task[Video]):
    """
    Task for processing a single video.
    """

    data: Video = field(default_factory=Video)

    def validate(self) -> bool:
        """Validate the task data."""
        if isinstance(self.data.input_video, pathlib.Path) and not os.path.exists(self.data.input_video):
            print(f"Video {self.data.input_video} does not exist")
            return False
        return True

    @property
    def num_items(self) -> int:
        """Get the number of items in this task."""
        return 1
