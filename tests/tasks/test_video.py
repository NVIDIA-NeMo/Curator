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

import pathlib
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from nemo_curator.tasks.video import (
    Clip,
    ClipStats,
    Video,
    VideoMetadata,
    VideoTask,
    _Window,
)
from nemo_curator.utils.decoder_utils import VideoMetadata as DecoderVideoMetadata

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestWindow:
    """Test suite for _Window class."""

    def test_window_initialization(self) -> None:
        """Test _Window initialization with default values."""
        window = _Window(start_frame=0, end_frame=10)

        assert window.start_frame == 0
        assert window.end_frame == 10
        assert window.mp4_bytes is None
        assert window.qwen_llm_input is None
        assert window.x1_input is None
        assert window.caption == {}
        assert window.enhanced_caption == {}
        assert window.webp_bytes is None

    def test_window_initialization_with_data(self) -> None:
        """Test _Window initialization with data."""
        mp4_data = b"fake_mp4_data"
        qwen_input = {"key": "value"}
        webp_data = b"fake_webp_data"
        caption = {"model1": "test caption"}
        enhanced_caption = {"model1": "enhanced caption"}

        window = _Window(
            start_frame=5,
            end_frame=15,
            mp4_bytes=mp4_data,
            qwen_llm_input=qwen_input,
            webp_bytes=webp_data,
            caption=caption,
            enhanced_caption=enhanced_caption,
        )

        assert window.start_frame == 5
        assert window.end_frame == 15
        assert window.mp4_bytes == mp4_data
        assert window.qwen_llm_input == qwen_input
        assert window.webp_bytes == webp_data
        assert window.caption == caption
        assert window.enhanced_caption == enhanced_caption

    def test_get_major_size_empty(self) -> None:
        """Test get_major_size with empty window."""
        window = _Window(start_frame=0, end_frame=10)
        size = window.get_major_size()

        # Should be just the size of the dictionaries (empty)
        assert size >= 0

    def test_get_major_size_with_data(self) -> None:
        """Test get_major_size with data."""
        mp4_data = b"fake_mp4_data"
        webp_data = b"fake_webp_data"
        qwen_input = {"key": "value"}
        caption = {"model1": "test"}
        enhanced_caption = {"model1": "enhanced"}

        window = _Window(
            start_frame=0,
            end_frame=10,
            mp4_bytes=mp4_data,
            qwen_llm_input=qwen_input,
            webp_bytes=webp_data,
            caption=caption,
            enhanced_caption=enhanced_caption,
        )

        size = window.get_major_size()

        # Should include sizes of all data
        assert size >= len(mp4_data) + len(webp_data)


class TestClip:
    """Test suite for Clip class."""

    def test_clip_initialization(self) -> None:
        """Test Clip initialization with default values."""
        clip_uuid = uuid4()
        clip = Clip(
            uuid=clip_uuid,
            source_video="test_video.mp4",
            span=(0.0, 10.0),
        )

        assert clip.uuid == clip_uuid
        assert clip.source_video == "test_video.mp4"
        assert clip.span == (0.0, 10.0)
        assert clip.buffer is None
        assert clip.extracted_frames == {}
        assert clip.motion_score_global_mean is None
        assert clip.aesthetic_score is None
        assert clip.windows == []

    def test_clip_duration_property(self) -> None:
        """Test clip duration property."""
        clip = Clip(
            uuid=uuid4(),
            source_video="test.mp4",
            span=(5.0, 15.0),
        )

        assert clip.duration == 10.0

    @patch("nemo_curator.tasks.video.extract_video_metadata")
    def test_extract_metadata_success(self, mock_extract: "MagicMock") -> None:
        """Test successful metadata extraction."""
        # Mock the metadata extraction
        mock_metadata = DecoderVideoMetadata(
            height=1080,
            width=1920,
            fps=30.0,
            num_frames=300,
            video_codec="h264",
            pixel_format="yuv420p",
            video_duration=10.0,
            bit_rate_k=5000,
        )
        mock_extract.return_value = mock_metadata

        buffer_data = b"fake_video_data"
        clip = Clip(
            uuid=uuid4(),
            source_video="test.mp4",
            span=(0.0, 10.0),
            buffer=buffer_data,
        )

        metadata = clip.extract_metadata()

        assert metadata is not None
        assert metadata["width"] == 1920
        assert metadata["height"] == 1080
        assert metadata["framerate"] == 30.0
        assert metadata["num_frames"] == 300
        assert metadata["video_codec"] == "h264"
        assert metadata["num_bytes"] == len(buffer_data)

        mock_extract.assert_called_once_with(buffer_data)

    def test_extract_metadata_no_buffer(self) -> None:
        """Test metadata extraction with no buffer."""
        clip = Clip(
            uuid=uuid4(),
            source_video="test.mp4",
            span=(0.0, 10.0),
        )

        metadata = clip.extract_metadata()
        assert metadata is None

    def test_get_major_size_empty(self) -> None:
        """Test get_major_size with empty clip."""
        clip = Clip(
            uuid=uuid4(),
            source_video="test.mp4",
            span=(0.0, 10.0),
        )

        size = clip.get_major_size()

        # Should at least include UUID bytes
        assert size >= 16  # UUID bytes

    def test_get_major_size_with_data(self) -> None:
        """Test get_major_size with data."""
        buffer_data = b"fake_video_data"
        frames = {"frame1": np.zeros((100, 100, 3), dtype=np.uint8)}
        intern_frames = np.zeros((10, 100, 100, 3), dtype=np.float32)
        intern_embedding = np.zeros((512,), dtype=np.float32)

        clip = Clip(
            uuid=uuid4(),
            source_video="test.mp4",
            span=(0.0, 10.0),
            buffer=buffer_data,
            extracted_frames=frames,
            intern_video_2_frames=intern_frames,
            intern_video_2_embedding=intern_embedding,
        )

        size = clip.get_major_size()

        expected_size = (
            16  # UUID bytes
            + len(buffer_data)
            + frames["frame1"].nbytes
            + intern_frames.nbytes
            + intern_embedding.nbytes
        )

        assert size >= expected_size


class TestClipStats:
    """Test suite for ClipStats class."""

    def test_clip_stats_initialization(self) -> None:
        """Test ClipStats initialization with default values."""
        stats = ClipStats()

        assert stats.num_filtered_by_motion == 0
        assert stats.num_filtered_by_aesthetic == 0
        assert stats.num_passed == 0
        assert stats.num_transcoded == 0
        assert stats.num_with_embeddings == 0
        assert stats.num_with_caption == 0
        assert stats.num_with_webp == 0
        assert stats.total_clip_duration == 0.0
        assert stats.max_clip_duration == 0.0

    def test_clip_stats_initialization_with_values(self) -> None:
        """Test ClipStats initialization with values."""
        stats = ClipStats(
            num_filtered_by_motion=5,
            num_filtered_by_aesthetic=3,
            num_passed=10,
            num_transcoded=8,
            num_with_embeddings=7,
            num_with_caption=6,
            num_with_webp=4,
            total_clip_duration=120.5,
            max_clip_duration=15.2,
        )

        assert stats.num_filtered_by_motion == 5
        assert stats.num_filtered_by_aesthetic == 3
        assert stats.num_passed == 10
        assert stats.num_transcoded == 8
        assert stats.num_with_embeddings == 7
        assert stats.num_with_caption == 6
        assert stats.num_with_webp == 4
        assert stats.total_clip_duration == 120.5
        assert stats.max_clip_duration == 15.2

    def test_combine_stats(self) -> None:
        """Test combining two ClipStats objects."""
        stats1 = ClipStats(
            num_filtered_by_motion=5,
            num_filtered_by_aesthetic=3,
            num_passed=10,
            num_transcoded=8,
            num_with_embeddings=7,
            num_with_caption=6,
            num_with_webp=4,
            total_clip_duration=120.5,
            max_clip_duration=15.2,
        )

        stats2 = ClipStats(
            num_filtered_by_motion=2,
            num_filtered_by_aesthetic=1,
            num_passed=5,
            num_transcoded=4,
            num_with_embeddings=3,
            num_with_caption=2,
            num_with_webp=1,
            total_clip_duration=60.3,
            max_clip_duration=20.1,
        )

        stats1.combine(stats2)

        assert stats1.num_filtered_by_motion == 7
        assert stats1.num_filtered_by_aesthetic == 4
        assert stats1.num_passed == 15
        assert stats1.num_transcoded == 12
        assert stats1.num_with_embeddings == 10
        assert stats1.num_with_caption == 8
        assert stats1.num_with_webp == 5
        assert stats1.total_clip_duration == 180.8
        assert stats1.max_clip_duration == 20.1  # Should be max of both


class TestVideoMetadata:
    """Test suite for VideoMetadata class."""

    def test_video_metadata_initialization(self) -> None:
        """Test VideoMetadata initialization with default values."""
        metadata = VideoMetadata()

        assert metadata.size is None
        assert metadata.height is None
        assert metadata.width is None
        assert metadata.framerate is None
        assert metadata.num_frames is None
        assert metadata.duration is None
        assert metadata.video_codec is None
        assert metadata.pixel_format is None
        assert metadata.audio_codec is None
        assert metadata.bit_rate_k is None

    def test_video_metadata_initialization_with_values(self) -> None:
        """Test VideoMetadata initialization with values."""
        metadata = VideoMetadata(
            size=1024000,
            height=1080,
            width=1920,
            framerate=30.0,
            num_frames=900,
            duration=30.0,
            video_codec="h264",
            pixel_format="yuv420p",
            audio_codec="aac",
            bit_rate_k=5000,
        )

        assert metadata.size == 1024000
        assert metadata.height == 1080
        assert metadata.width == 1920
        assert metadata.framerate == 30.0
        assert metadata.num_frames == 900
        assert metadata.duration == 30.0
        assert metadata.video_codec == "h264"
        assert metadata.pixel_format == "yuv420p"
        assert metadata.audio_codec == "aac"
        assert metadata.bit_rate_k == 5000


class TestVideo:
    """Test suite for Video class."""

    def test_video_initialization(self) -> None:
        """Test Video initialization with default values."""
        video_path = pathlib.Path("test_video.mp4")
        video = Video(input_video=video_path)

        assert video.input_video == video_path
        assert video.source_bytes is None
        assert isinstance(video.metadata, VideoMetadata)
        assert video.frame_array is None
        assert video.clips == []
        assert video.filtered_clips == []
        assert video.num_total_clips == 0
        assert video.num_clip_chunks == 0
        assert video.clip_chunk_index == 0
        assert isinstance(video.clip_stats, ClipStats)
        assert video.errors == {}

    @patch("nemo_curator.tasks.video.extract_video_metadata")
    def test_populate_metadata_success(self, mock_extract: "MagicMock") -> None:
        """Test successful metadata population."""
        # Mock the metadata extraction
        mock_metadata = DecoderVideoMetadata(
            height=1080,
            width=1920,
            fps=30.0,
            num_frames=300,
            video_codec="h264",
            pixel_format="yuv420p",
            video_duration=10.0,
            audio_codec="aac",
            bit_rate_k=5000,
        )
        mock_extract.return_value = mock_metadata

        video_data = b"fake_video_data"
        video = Video(
            input_video=pathlib.Path("test.mp4"),
            source_bytes=video_data,
        )

        video.populate_metadata()

        assert video.metadata.size == len(video_data)
        assert video.metadata.height == 1080
        assert video.metadata.width == 1920
        assert video.metadata.framerate == 30.0
        assert video.metadata.num_frames == 300
        assert video.metadata.duration == 10.0
        assert video.metadata.video_codec == "h264"
        assert video.metadata.pixel_format == "yuv420p"
        assert video.metadata.audio_codec == "aac"
        assert video.metadata.bit_rate_k == 5000

        mock_extract.assert_called_once_with(video_data)

    def test_populate_metadata_no_bytes(self) -> None:
        """Test metadata population with no source bytes."""
        video = Video(input_video=pathlib.Path("test.mp4"))

        with pytest.raises(ValueError, match="No video data available: source_bytes is None"):
            video.populate_metadata()

    def test_fraction_property(self) -> None:
        """Test fraction property calculation."""
        video = Video(input_video=pathlib.Path("test.mp4"))

        # Test with no clips
        assert video.fraction == 1.0

        # Test with clips
        video.num_total_clips = 10
        video.clips = [Mock() for _ in range(3)]
        video.filtered_clips = [Mock() for _ in range(2)]

        # (3 + 2) / 10 = 0.5
        assert video.fraction == 0.5

    def test_weight_property(self) -> None:
        """Test weight property calculation."""
        video = Video(input_video=pathlib.Path("test.mp4"))

        # Test with no size
        assert video.weight == 0

        # Test with size and duration
        video.metadata.size = 1024000
        video.metadata.duration = 150.0  # 2.5 minutes
        video.num_total_clips = 10
        video.clips = [Mock() for _ in range(5)]
        video.filtered_clips = [Mock() for _ in range(3)]

        # weight = (150 / 300) * ((5 + 3) / 10) = 0.5 * 0.8 = 0.4
        assert video.weight == 0.4

    def test_weight_property_no_duration(self) -> None:
        """Test weight property with no duration."""
        video = Video(input_video=pathlib.Path("test.mp4"))
        video.metadata.size = 1024000

        with pytest.raises(ValueError, match="metadata.duration is None"):
            _ = video.weight

    def test_get_major_size(self) -> None:
        """Test get_major_size calculation."""
        video_data = b"fake_video_data"
        frame_array = np.zeros((100, 100, 3), dtype=np.uint8)

        video = Video(
            input_video=pathlib.Path("test.mp4"),
            source_bytes=video_data,
            frame_array=frame_array,
        )

        # Add mock clips
        mock_clip = Mock()
        mock_clip.get_major_size.return_value = 1000
        video.clips = [mock_clip]

        size = video.get_major_size()

        expected_size = len(video_data) + frame_array.nbytes + 1000
        assert size >= expected_size

    def test_has_metadata(self) -> None:
        """Test has_metadata method."""
        video = Video(input_video=pathlib.Path("test.mp4"))

        # Test with empty metadata
        assert not video.has_metadata()

        # Test with partial metadata
        video.metadata.height = 1080
        video.metadata.width = 1920
        assert not video.has_metadata()

        # Test with complete metadata
        video.metadata.duration = 30.0
        video.metadata.framerate = 30.0
        video.metadata.num_frames = 900
        video.metadata.video_codec = "h264"

        assert video.has_metadata()

    def test_is_10_bit_color(self) -> None:
        """Test is_10_bit_color method."""
        video = Video(input_video=pathlib.Path("test.mp4"))

        # Test with no pixel format
        assert video.is_10_bit_color() is None

        # Test with 8-bit format
        video.metadata.pixel_format = "yuv420p"
        assert video.is_10_bit_color() is False

        # Test with 10-bit little endian format
        video.metadata.pixel_format = "yuv420p10le"
        assert video.is_10_bit_color() is True

        # Test with 10-bit big endian format
        video.metadata.pixel_format = "yuv420p10be"
        assert video.is_10_bit_color() is True

    def test_input_path_property(self) -> None:
        """Test input_path property."""
        video_path = pathlib.Path("/home/user/test_video.mp4")
        video = Video(input_video=video_path)

        assert video.input_path == "/home/user/test_video.mp4"


class TestVideoTask:
    """Test suite for VideoTask class."""

    def test_video_task_initialization(self) -> None:
        """Test VideoTask initialization."""
        video_data = Video(input_video=pathlib.Path("test.mp4"))
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=video_data,
        )

        assert task.task_id == "test_task"
        assert task.dataset_name == "test_dataset"
        assert isinstance(task.data, Video)

    def test_video_task_initialization_with_data(self) -> None:
        """Test VideoTask initialization with video data."""
        video_data = Video(input_video=pathlib.Path("test.mp4"))
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=video_data,
        )

        assert task.data is video_data

    def test_validate_existing_file(self) -> None:
        """Test validate method with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)

            try:
                video_data = Video(input_video=tmp_path)
                task = VideoTask(
                    task_id="test_task",
                    dataset_name="test_dataset",
                    data=video_data,
                )

                assert task.validate() is True
            finally:
                tmp_path.unlink()

    def test_validate_non_existing_file(self) -> None:
        """Test validate method with non-existing file."""
        video_data = Video(input_video=pathlib.Path("non_existing_file.mp4"))
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=video_data,
        )

        # Should print error message and return False
        assert task.validate() is False

    def test_num_items_property(self) -> None:
        """Test num_items property."""
        video_data = Video(input_video=pathlib.Path("test.mp4"))
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=video_data,
        )

        assert task.num_items == 1
