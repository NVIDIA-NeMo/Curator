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

import io
import json
import pathlib
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from nemo_curator.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    Resolution,
    VideoMetadata,
    _make_video_stream,
    decode_video_cpu,
    extract_frames,
    extract_video_metadata,
    find_closest_indices,
    get_avg_frame_rate,
    get_frame_count,
    get_video_timestamps,
    sample_closest,
    save_stream_position,
)


class TestResolution:
    """Test suite for Resolution NamedTuple."""

    def test_resolution_creation(self) -> None:
        """Test Resolution creation with height and width."""
        resolution = Resolution(height=1080, width=1920)

        assert resolution.height == 1080
        assert resolution.width == 1920

    def test_resolution_indexing(self) -> None:
        """Test Resolution indexing access."""
        resolution = Resolution(720, 1280)

        assert resolution[0] == 720  # height
        assert resolution[1] == 1280  # width

    def test_resolution_unpacking(self) -> None:
        """Test Resolution unpacking."""
        resolution = Resolution(480, 640)
        height, width = resolution

        assert height == 480
        assert width == 640


class TestVideoMetadata:
    """Test suite for VideoMetadata dataclass."""

    def test_video_metadata_default_initialization(self) -> None:
        """Test VideoMetadata initialization with default values."""
        metadata = VideoMetadata()

        assert metadata.height is None
        assert metadata.width is None
        assert metadata.fps is None
        assert metadata.num_frames is None
        assert metadata.video_codec is None
        assert metadata.pixel_format is None
        assert metadata.video_duration is None
        assert metadata.audio_codec is None
        assert metadata.bit_rate_k is None

    def test_video_metadata_initialization_with_values(self) -> None:
        """Test VideoMetadata initialization with specific values."""
        metadata = VideoMetadata(
            height=1080,
            width=1920,
            fps=30.0,
            num_frames=900,
            video_codec="h264",
            pixel_format="yuv420p",
            video_duration=30.0,
            audio_codec="aac",
            bit_rate_k=5000,
        )

        assert metadata.height == 1080
        assert metadata.width == 1920
        assert metadata.fps == 30.0
        assert metadata.num_frames == 900
        assert metadata.video_codec == "h264"
        assert metadata.pixel_format == "yuv420p"
        assert metadata.video_duration == 30.0
        assert metadata.audio_codec == "aac"
        assert metadata.bit_rate_k == 5000


class TestFrameExtractionPolicy:
    """Test suite for FrameExtractionPolicy enum."""

    def test_frame_extraction_policy_values(self) -> None:
        """Test FrameExtractionPolicy enum values."""
        assert FrameExtractionPolicy.first.value == 0
        assert FrameExtractionPolicy.middle.value == 1
        assert FrameExtractionPolicy.last.value == 2
        assert FrameExtractionPolicy.sequence.value == 3

    def test_frame_extraction_policy_string_representation(self) -> None:
        """Test FrameExtractionPolicy string representation."""
        assert str(FrameExtractionPolicy.first) == "FrameExtractionPolicy.first"
        assert str(FrameExtractionPolicy.middle) == "FrameExtractionPolicy.middle"
        assert str(FrameExtractionPolicy.last) == "FrameExtractionPolicy.last"
        assert str(FrameExtractionPolicy.sequence) == "FrameExtractionPolicy.sequence"


class TestFrameExtractionSignature:
    """Test suite for FrameExtractionSignature dataclass."""

    def test_frame_extraction_signature_initialization(self) -> None:
        """Test FrameExtractionSignature initialization."""
        signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.middle,
            target_fps=30.0,
        )

        assert signature.extraction_policy == FrameExtractionPolicy.middle
        assert signature.target_fps == 30.0

    def test_to_str_method(self) -> None:
        """Test FrameExtractionSignature to_str method."""
        signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=24.0,
        )

        result = signature.to_str()

        assert result == "FrameExtractionPolicy.sequence-24000"

    def test_to_str_method_with_fractional_fps(self) -> None:
        """Test FrameExtractionSignature to_str method with fractional FPS."""
        signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.first,
            target_fps=29.97,
        )

        result = signature.to_str()

        assert result == "FrameExtractionPolicy.first-29970"


class TestExtractVideoMetadata:
    """Test suite for extract_video_metadata function."""

    def create_mock_ffprobe_output(self) -> dict:
        """Create mock ffprobe output for testing."""
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "height": 1080,
                    "width": 1920,
                    "avg_frame_rate": "30/1",
                    "duration": "10.0",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "bit_rate": "5000000",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                },
            ],
            "format": {"duration": "10.0"},
        }

    @patch("subprocess.run")
    @patch("nemo_curator.utils.decoder_utils.make_pipeline_named_temporary_file")
    def test_extract_video_metadata_with_bytes(self, mock_temp_file: Mock, mock_subprocess: Mock) -> None:
        """Test extract_video_metadata with bytes input."""
        mock_video_data = b"fake_video_data"
        mock_output = self.create_mock_ffprobe_output()

        # Mock the temporary file
        mock_file_path = Mock()
        mock_file_path.write_bytes = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.as_posix.return_value = "/tmp/test_video.mp4"  # noqa: S108
        mock_temp_file.return_value.__enter__.return_value = mock_file_path

        # Mock subprocess
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output).encode()
        mock_subprocess.return_value = mock_result

        result = extract_video_metadata(mock_video_data)

        assert isinstance(result, VideoMetadata)
        assert result.height == 1080
        assert result.width == 1920
        assert result.fps == 30.0
        assert result.num_frames == 300  # 10.0 * 30.0
        assert result.video_codec == "h264"
        assert result.pixel_format == "yuv420p"
        assert result.audio_codec == "aac"
        assert result.video_duration == 10.0
        assert result.bit_rate_k == 4882  # 5000000 / 1024

        mock_file_path.write_bytes.assert_called_once_with(mock_video_data)

    @patch("subprocess.run")
    def test_extract_video_metadata_with_string_path(self, mock_subprocess: Mock) -> None:
        """Test extract_video_metadata with string path input."""
        mock_output = self.create_mock_ffprobe_output()

        # Mock subprocess
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output).encode()
        mock_subprocess.return_value = mock_result

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            result = extract_video_metadata(tmp_path)

            assert isinstance(result, VideoMetadata)
            assert result.height == 1080
            assert result.width == 1920
            assert result.fps == 30.0
        finally:
            pathlib.Path(tmp_path).unlink()

    def test_extract_video_metadata_file_not_found(self) -> None:
        """Test extract_video_metadata with non-existent file."""
        non_existent_path = "/path/to/non/existent/file.mp4"

        with pytest.raises(FileNotFoundError, match="not found"):
            extract_video_metadata(non_existent_path)

    @patch("subprocess.run")
    @patch("nemo_curator.utils.decoder_utils.make_pipeline_named_temporary_file")
    def test_extract_video_metadata_no_video_stream(self, mock_temp_file: Mock, mock_subprocess: Mock) -> None:
        """Test extract_video_metadata with no video stream."""
        mock_output = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                }
            ],
            "format": {},
        }

        # Mock the temporary file
        mock_file_path = Mock()
        mock_file_path.write_bytes = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.as_posix.return_value = "/tmp/test_video.mp4"  # noqa: S108
        mock_temp_file.return_value.__enter__.return_value = mock_file_path

        # Mock subprocess
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output).encode()
        mock_subprocess.return_value = mock_result

        with pytest.raises(ValueError, match="No video stream found"):
            extract_video_metadata(b"fake_data")

    @patch("subprocess.run")
    @patch("nemo_curator.utils.decoder_utils.make_pipeline_named_temporary_file")
    def test_extract_video_metadata_no_duration(self, mock_temp_file: Mock, mock_subprocess: Mock) -> None:
        """Test extract_video_metadata with no duration information."""
        mock_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "height": 1080,
                    "width": 1920,
                    "avg_frame_rate": "30/1",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                }
            ],
            "format": {},
        }

        # Mock the temporary file
        mock_file_path = Mock()
        mock_file_path.write_bytes = Mock()
        mock_file_path.exists.return_value = True
        mock_file_path.as_posix.return_value = "/tmp/test_video.mp4"  # noqa: S108
        mock_temp_file.return_value.__enter__.return_value = mock_file_path

        # Mock subprocess
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output).encode()
        mock_subprocess.return_value = mock_result

        with pytest.raises(KeyError, match="Could not find `duration` in video metadata"):
            extract_video_metadata(b"fake_data")


class TestMakeVideoStream:
    """Test suite for _make_video_stream function."""

    def test_make_video_stream_with_path_string(self) -> None:
        """Test _make_video_stream with string path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(b"test_data")

        try:
            result = _make_video_stream(tmp_path)

            assert hasattr(result, "read")
            assert hasattr(result, "seek")
            result.close()
        finally:
            pathlib.Path(tmp_path).unlink()

    def test_make_video_stream_with_path_object(self) -> None:
        """Test _make_video_stream with Path object."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = pathlib.Path(tmp_file.name)
            tmp_file.write(b"test_data")

        try:
            result = _make_video_stream(tmp_path)

            assert hasattr(result, "read")
            assert hasattr(result, "seek")
            result.close()
        finally:
            tmp_path.unlink()

    def test_make_video_stream_with_bytes(self) -> None:
        """Test _make_video_stream with bytes."""
        test_data = b"test_video_data"

        result = _make_video_stream(test_data)

        assert isinstance(result, io.BytesIO)
        assert result.read() == test_data
        result.seek(0)  # Reset position

    def test_make_video_stream_with_bytes_io(self) -> None:
        """Test _make_video_stream with BytesIO object."""
        test_data = b"test_video_data"
        bytes_io = io.BytesIO(test_data)

        result = _make_video_stream(bytes_io)

        assert result is bytes_io
        assert result.read() == test_data

    def test_make_video_stream_with_invalid_type(self) -> None:
        """Test _make_video_stream with invalid input type."""
        with pytest.raises(ValueError, match="Invalid video type"):
            _make_video_stream(123)


class TestSaveStreamPosition:
    """Test suite for save_stream_position context manager."""

    def test_save_stream_position_success(self) -> None:
        """Test save_stream_position saves and restores position."""
        test_data = b"test_data_for_stream_position"
        stream = io.BytesIO(test_data)

        # Move to a specific position
        stream.seek(5)
        initial_position = stream.tell()

        with save_stream_position(stream) as saved_stream:
            assert saved_stream is stream
            # Move to different position inside context
            stream.seek(10)
            assert stream.tell() == 10

        # Position should be restored
        assert stream.tell() == initial_position

    def _raise_test_exception(self) -> None:
        """Helper function to raise test exception."""
        msg = "Test exception"
        raise ValueError(msg)

    def test_save_stream_position_with_exception(self) -> None:
        """Test save_stream_position restores position even with exception."""
        test_data = b"test_data_for_exception_test"
        stream = io.BytesIO(test_data)

        # Move to a specific position
        stream.seek(3)
        initial_position = stream.tell()

        try:
            with save_stream_position(stream):
                stream.seek(8)
                self._raise_test_exception()
        except ValueError:
            pass

        # Position should still be restored
        assert stream.tell() == initial_position


class TestFindClosestIndices:
    """Test suite for find_closest_indices function."""

    def test_find_closest_indices_exact_matches(self) -> None:
        """Test find_closest_indices with exact matches."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        dst = np.array([2.0, 4.0], dtype=np.float32)

        result = find_closest_indices(src, dst)

        expected = np.array([1, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_find_closest_indices_closest_matches(self) -> None:
        """Test find_closest_indices with closest matches."""
        src = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        dst = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

        result = find_closest_indices(src, dst)

        expected = np.array([0, 1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_find_closest_indices_beyond_range(self) -> None:
        """Test find_closest_indices with values beyond source range."""
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dst = np.array([0.5, 3.5, 5.0], dtype=np.float32)

        result = find_closest_indices(src, dst)

        expected = np.array([0, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_find_closest_indices_equidistant(self) -> None:
        """Test find_closest_indices with equidistant values (should prefer left)."""
        src = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        dst = np.array([2.0, 4.0], dtype=np.float32)

        result = find_closest_indices(src, dst)

        expected = np.array([0, 1], dtype=np.int32)  # Should prefer left indices
        np.testing.assert_array_equal(result, expected)


class TestSampleClosest:
    """Test suite for sample_closest function."""

    def test_sample_closest_basic(self) -> None:
        """Test sample_closest with basic parameters."""
        src = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample_rate = 2.0

        indices, counts, sample_elements = sample_closest(src, sample_rate)

        assert len(indices) == len(counts)
        assert np.all(counts >= 1)
        # When dedup=True, indices are deduplicated but sample_elements contains all original samples
        assert len(sample_elements) >= len(indices)

    def test_sample_closest_with_start_stop(self) -> None:
        """Test sample_closest with start and stop parameters."""
        src = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample_rate = 1.0
        start = 1.0
        stop = 3.0

        indices, counts, sample_elements = sample_closest(src, sample_rate, start=start, stop=stop)

        assert sample_elements[0] >= start
        assert sample_elements[-1] <= stop

    def test_sample_closest_no_endpoint(self) -> None:
        """Test sample_closest with endpoint=False."""
        src = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample_rate = 1.0

        indices_with_endpoint, _, _ = sample_closest(src, sample_rate, endpoint=True)
        indices_without_endpoint, _, _ = sample_closest(src, sample_rate, endpoint=False)

        assert len(indices_without_endpoint) <= len(indices_with_endpoint)

    def test_sample_closest_no_dedup(self) -> None:
        """Test sample_closest with dedup=False."""
        src = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sample_rate = 10.0  # High rate to cause duplicates

        indices_dedup, counts_dedup, _ = sample_closest(src, sample_rate, dedup=True)
        indices_no_dedup, counts_no_dedup, _ = sample_closest(src, sample_rate, dedup=False)

        assert len(indices_no_dedup) >= len(indices_dedup)
        assert np.all(counts_no_dedup == 1)

    def test_sample_closest_invalid_sample_rate(self) -> None:
        """Test sample_closest with invalid sample rate."""
        src = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match="Sample rate must be greater than 0"):
            sample_closest(src, 0.0)

        with pytest.raises(ValueError, match="Sample rate must be greater than 0"):
            sample_closest(src, -1.0)


class TestMockedVideoFunctions:
    """Test suite for video functions that require mocking external dependencies."""

    @patch("av.open")
    def test_get_video_timestamps_basic(self, mock_av_open: Mock) -> None:
        """Test get_video_timestamps with basic functionality."""
        # Mock the AV container and stream
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.time_base = 1 / 30.0  # 30 FPS
        mock_container.streams.video = [mock_stream]

        # Mock packets with timestamps
        mock_packet1 = Mock()
        mock_packet1.pts = 0
        mock_packet2 = Mock()
        mock_packet2.pts = 30
        mock_packet3 = Mock()
        mock_packet3.pts = 60

        mock_container.demux.return_value = [mock_packet1, mock_packet2, mock_packet3]
        mock_av_open.return_value.__enter__.return_value = mock_container

        test_data = b"fake_video_data"
        result = get_video_timestamps(test_data)

        expected = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    @patch("av.open")
    def test_get_video_timestamps_none_time_base(self, mock_av_open: Mock) -> None:
        """Test get_video_timestamps with None time_base."""
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.time_base = None
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value.__enter__.return_value = mock_container

        test_data = b"fake_video_data"

        with pytest.raises(ValueError, match="Time base is None"):
            get_video_timestamps(test_data)

    @patch("av.open")
    def test_get_avg_frame_rate_with_average_rate(self, mock_av_open: Mock) -> None:
        """Test get_avg_frame_rate when average_rate is available."""
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.average_rate = 30.0
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value.__enter__.return_value = mock_container

        test_data = b"fake_video_data"
        result = get_avg_frame_rate(test_data)

        assert result == 30.0

    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    @patch("av.open")
    def test_get_avg_frame_rate_fallback(self, mock_av_open: Mock, mock_get_timestamps: Mock) -> None:
        """Test get_avg_frame_rate fallback to timestamp calculation."""
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.average_rate = None
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value.__enter__.return_value = mock_container

        # Mock timestamps for fallback calculation
        mock_get_timestamps.return_value = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        test_data = b"fake_video_data"
        result = get_avg_frame_rate(test_data)

        expected = 3.0 / 4  # (last - first) / num_frames
        assert result == expected

    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    @patch("av.open")
    def test_get_avg_frame_rate_insufficient_frames(self, mock_av_open: Mock, mock_get_timestamps: Mock) -> None:
        """Test get_avg_frame_rate with insufficient frames."""
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.average_rate = None
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value.__enter__.return_value = mock_container

        # Mock single frame
        mock_get_timestamps.return_value = np.array([0.0], dtype=np.float32)

        test_data = b"fake_video_data"

        with pytest.raises(ValueError, match="Not enough frames"):
            get_avg_frame_rate(test_data)

    @patch("av.open")
    def test_get_frame_count_with_frames_attribute(self, mock_av_open: Mock) -> None:
        """Test get_frame_count when frames attribute is available."""
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.frames = 300
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value.__enter__.return_value = mock_container

        test_data = b"fake_video_data"
        result = get_frame_count(test_data)

        assert result == 300

    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    @patch("av.open")
    def test_get_frame_count_fallback(self, mock_av_open: Mock, mock_get_timestamps: Mock) -> None:
        """Test get_frame_count fallback to timestamp counting."""
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.frames = None
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value.__enter__.return_value = mock_container

        # Mock timestamps for fallback
        mock_get_timestamps.return_value = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        test_data = b"fake_video_data"
        result = get_frame_count(test_data)

        assert result == 4

    @patch("nemo_curator.utils.decoder_utils.decode_video_cpu_frame_ids")
    @patch("nemo_curator.utils.decoder_utils.sample_closest")
    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    def test_decode_video_cpu_basic(
        self, mock_get_timestamps: Mock, mock_sample_closest: Mock, mock_decode_frame_ids: Mock
    ) -> None:
        """Test decode_video_cpu with basic parameters."""
        # Mock timestamps
        mock_timestamps = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        mock_get_timestamps.return_value = mock_timestamps

        # Mock sample_closest
        mock_frame_ids = np.array([0, 1, 2], dtype=np.int32)
        mock_counts = np.array([1, 1, 1], dtype=np.int32)
        mock_sample_elements = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        mock_sample_closest.return_value = (mock_frame_ids, mock_counts, mock_sample_elements)

        # Mock decode_video_cpu_frame_ids
        mock_frames = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        mock_decode_frame_ids.return_value = mock_frames

        test_data = b"fake_video_data"
        result = decode_video_cpu(test_data, sample_rate_fps=1.0)

        assert result.shape == (3, 480, 640, 3)
        mock_get_timestamps.assert_called_once()
        mock_sample_closest.assert_called_once()
        mock_decode_frame_ids.assert_called_once()

    @patch("nemo_curator.utils.decoder_utils.decode_video_cpu")
    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    def test_extract_frames_sequence_policy(self, mock_get_timestamps: Mock, mock_decode_cpu: Mock) -> None:
        """Test extract_frames with sequence policy."""
        # Mock timestamps
        mock_timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        mock_get_timestamps.return_value = mock_timestamps

        # Mock decode_video_cpu
        mock_frames = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        mock_decode_cpu.return_value = mock_frames

        test_data = b"fake_video_data"
        result = extract_frames(
            test_data, extraction_policy=FrameExtractionPolicy.sequence, sample_rate_fps=1.0, target_res=(-1, -1)
        )

        assert result.shape == (3, 480, 640, 3)
        mock_get_timestamps.assert_called_once()
        mock_decode_cpu.assert_called_once()

    @patch("nemo_curator.utils.decoder_utils.decode_video_cpu")
    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    def test_extract_frames_middle_policy(self, mock_get_timestamps: Mock, mock_decode_cpu: Mock) -> None:
        """Test extract_frames with middle policy."""
        # Mock timestamps - even number of frames
        mock_timestamps = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        mock_get_timestamps.return_value = mock_timestamps

        # Mock decode_video_cpu
        mock_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        mock_decode_cpu.return_value = mock_frames

        test_data = b"fake_video_data"
        result = extract_frames(test_data, extraction_policy=FrameExtractionPolicy.middle, sample_rate_fps=1.0)

        assert result.shape == (1, 480, 640, 3)
        # Should call decode_cpu with middle frame timestamp
        mock_decode_cpu.assert_called_once()

    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    def test_extract_frames_empty_video(self, mock_get_timestamps: Mock) -> None:
        """Test extract_frames with empty video."""
        mock_get_timestamps.return_value = np.array([], dtype=np.float32)

        test_data = b"fake_video_data"

        with pytest.raises(ValueError, match="Can't extract frames from empty video"):
            extract_frames(test_data, extraction_policy=FrameExtractionPolicy.sequence, sample_rate_fps=1.0)

    @patch("nemo_curator.utils.decoder_utils.get_video_timestamps")
    def test_extract_frames_unsupported_policy(self, mock_get_timestamps: Mock) -> None:
        """Test extract_frames with unsupported extraction policy."""
        mock_timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        mock_get_timestamps.return_value = mock_timestamps

        test_data = b"fake_video_data"

        with pytest.raises(NotImplementedError, match="Extraction policies apart from Sequence and Middle"):
            extract_frames(test_data, extraction_policy=FrameExtractionPolicy.first, sample_rate_fps=1.0)
