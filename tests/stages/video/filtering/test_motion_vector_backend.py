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
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from nemo_curator.stages.video.filtering.motion_vector_backend import (
    DecodedData,
    MotionInfo,
    VideoResolutionTooSmallError,
    check_if_small_motion,
    decode_for_motion,
    motion_vectors_to_flowfield,
)

# Set up numpy random generator
rng = np.random.default_rng(seed=42)


class TestVideoResolutionTooSmallError:
    """Test cases for VideoResolutionTooSmallError."""

    def test_error_message(self):
        """Test that error message is properly formatted."""
        error_msg = "Test error message"
        error = VideoResolutionTooSmallError(error_msg)
        assert str(error) == error_msg


class TestMotionInfo:
    """Test cases for MotionInfo dataclass."""

    def test_creation(self):
        """Test MotionInfo creation."""
        motion_info = MotionInfo(is_small_motion=True, per_patch_min_256=0.001, global_mean=0.002)
        assert motion_info.is_small_motion is True
        assert motion_info.per_patch_min_256 == 0.001
        assert motion_info.global_mean == 0.002

    def test_creation_with_large_motion(self):
        """Test MotionInfo creation with large motion."""
        motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.01, global_mean=0.02)
        assert motion_info.is_small_motion is False
        assert motion_info.per_patch_min_256 == 0.01
        assert motion_info.global_mean == 0.02


class TestDecodedData:
    """Test cases for DecodedData dataclass."""

    def test_creation(self):
        """Test DecodedData creation."""
        frames = [np.ones((100, 100, 3), dtype=np.uint8)]
        frame_size = torch.Size([100, 100, 3])
        decoded_data = DecodedData(frames=frames, frame_size=frame_size)

        assert len(decoded_data.frames) == 1
        assert decoded_data.frame_size == frame_size

    def test_get_major_size(self):
        """Test get_major_size method."""
        # Create test frames with known sizes
        frame1 = np.ones((100, 100, 3), dtype=np.uint8)  # 30,000 bytes
        frame2 = np.ones((50, 50, 3), dtype=np.uint8)  # 7,500 bytes
        frames = [frame1, frame2]

        decoded_data = DecodedData(frames=frames, frame_size=torch.Size([100, 100, 3]))
        total_size = decoded_data.get_major_size()

        expected_size = frame1.nbytes + frame2.nbytes
        assert total_size == expected_size

    def test_get_major_size_empty_frames(self):
        """Test get_major_size with empty frames list."""
        decoded_data = DecodedData(frames=[], frame_size=torch.Size([100, 100, 3]))
        assert decoded_data.get_major_size() == 0


class TestMotionVectorsToFlowfield:
    """Test cases for motion_vectors_to_flowfield function."""

    def test_basic_functionality(self):
        """Test basic functionality of motion_vectors_to_flowfield."""
        # Create minimal test data
        batch_size = 1
        n_vectors = 1
        size = [64, 64]

        # Mock motion vector data (10 fields as expected by the function)
        mvs = torch.zeros(batch_size, n_vectors, 10)
        mvs[0, 0, 0:2] = torch.tensor([8.0, 8.0])  # block_sizes
        mvs[0, 0, 4:6] = torch.tensor([32.0, 32.0])  # dst position
        mvs[0, 0, 7:9] = torch.tensor([1.0, 1.0])  # motion
        mvs[0, 0, 9] = 1.0  # motion_scale

        flow = motion_vectors_to_flowfield(mvs, size)

        assert flow.shape == (batch_size, *size, 2)
        assert flow.dtype == torch.float32

    def test_multiple_batch_sizes(self):
        """Test with multiple batch sizes."""
        batch_size = 3
        n_vectors = 2
        size = [32, 32]

        # Create test data
        mvs = torch.zeros(batch_size, n_vectors, 10)
        # Fill with valid data
        mvs[:, :, 0:2] = torch.tensor([8.0, 8.0])  # block_sizes
        mvs[:, :, 4:6] = torch.tensor([16.0, 16.0])  # dst position
        mvs[:, :, 7:9] = torch.tensor([2.0, 2.0])  # motion
        mvs[:, :, 9] = 1.0  # motion_scale

        flow = motion_vectors_to_flowfield(mvs, size)

        assert flow.shape == (batch_size, *size, 2)

    def test_with_preallocated_flow(self):
        """Test with preallocated flow tensor."""
        batch_size = 1
        n_vectors = 1
        size = [32, 32]

        # Create preallocated flow
        preallocated_flow = torch.ones(batch_size, *size, 2)

        mvs = torch.zeros(batch_size, n_vectors, 10)
        mvs[0, 0, 0:2] = torch.tensor([8.0, 8.0])  # block_sizes
        mvs[0, 0, 4:6] = torch.tensor([16.0, 16.0])  # dst position
        mvs[0, 0, 7:9] = torch.tensor([1.0, 1.0])  # motion
        mvs[0, 0, 9] = 1.0  # motion_scale

        flow = motion_vectors_to_flowfield(mvs, size, preallocated_flow)

        assert flow.shape == (batch_size, *size, 2)
        # Flow should be modified (not all ones anymore)
        assert not torch.all(flow == 1.0)

    def test_edge_cases_boundary_conditions(self):
        """Test edge cases with boundary conditions."""
        batch_size = 1
        n_vectors = 1
        size = [16, 16]

        # Test with destination at boundary
        mvs = torch.zeros(batch_size, n_vectors, 10)
        mvs[0, 0, 0:2] = torch.tensor([8.0, 8.0])  # block_sizes
        mvs[0, 0, 4:6] = torch.tensor([15.0, 15.0])  # dst position at boundary
        mvs[0, 0, 7:9] = torch.tensor([1.0, 1.0])  # motion
        mvs[0, 0, 9] = 1.0  # motion_scale

        flow = motion_vectors_to_flowfield(mvs, size)

        assert flow.shape == (batch_size, *size, 2)


class TestDecodeForMotion:
    """Test cases for decode_for_motion function."""

    def test_video_resolution_too_small_error(self):
        """Test that VideoResolutionTooSmallError is raised for small resolution."""
        mock_video = io.BytesIO(b"mock_video_data")

        with patch("av.open") as mock_open, patch("av.codec.context.Flags2.EXPORT_MVS", create=True):
            # Mock frame with small resolution
            mock_frame = Mock()
            mock_frame.height = 100  # Less than 256
            mock_frame.width = 100  # Less than 256
            mock_frame.side_data = []

            # Mock stream
            mock_stream = Mock()
            mock_stream.codec_context = Mock()
            mock_stream.codec_context.flags2 = 0
            mock_stream.codec_context.thread_type = Mock()
            mock_stream.codec_context.thread_count = 4
            mock_stream.average_rate = 30.0
            mock_stream.duration = 30
            mock_stream.time_base = 1.0

            # Mock container
            mock_container = Mock()
            mock_container.streams.video = [mock_stream]
            mock_container.decode.return_value = iter([mock_frame])
            mock_container.duration = 30
            mock_container.__enter__ = Mock(return_value=mock_container)
            mock_container.__exit__ = Mock(return_value=None)

            mock_open.return_value = mock_container

            with pytest.raises(VideoResolutionTooSmallError):
                decode_for_motion(mock_video)

    def test_successful_decode(self):
        """Test successful decode operation."""
        mock_video = io.BytesIO(b"mock_video_data")

        with patch("av.open") as mock_open, patch("av.codec.context.Flags2.EXPORT_MVS", create=True):
            # Mock motion vector side data
            mock_side_data = Mock()
            mock_side_data.type = Mock()
            mock_side_data.type.__eq__ = Mock(return_value=True)

            # Create proper structured array
            dtype = [
                ("field0", "i4"),
                ("field1", "i4"),
                ("field2", "i4"),
                ("field3", "i4"),
                ("field4", "i4"),
                ("field5", "i4"),
                ("field6", "i4"),
                ("field7", "i4"),
                ("field8", "i4"),
                ("field9", "i4"),
                ("field10", "i4"),
            ]
            structured_array = np.ones((5,), dtype=dtype)
            mock_side_data.to_ndarray.return_value = structured_array

            # Mock frame with valid resolution
            mock_frame = Mock()
            mock_frame.height = 480
            mock_frame.width = 640
            mock_frame.side_data = [mock_side_data]

            # Mock stream
            mock_stream = Mock()
            mock_stream.codec_context = Mock()
            mock_stream.codec_context.flags2 = 0
            mock_stream.codec_context.thread_type = Mock()
            mock_stream.codec_context.thread_count = 4
            mock_stream.average_rate = 30.0
            mock_stream.duration = 30
            mock_stream.time_base = 1.0

            # Mock container
            mock_container = Mock()
            mock_container.streams.video = [mock_stream]
            mock_container.decode.return_value = iter([mock_frame])
            mock_container.duration = 30
            mock_container.__enter__ = Mock(return_value=mock_container)
            mock_container.__exit__ = Mock(return_value=None)

            mock_open.return_value = mock_container

            with patch("av.sidedata.sidedata.Type.MOTION_VECTORS", create=True):
                result = decode_for_motion(mock_video)

                assert isinstance(result, DecodedData)
                assert len(result.frames) == 1
                assert result.frame_size == torch.Size([480, 640, 3])

    def test_no_motion_vectors(self):
        """Test decode with no motion vectors."""
        mock_video = io.BytesIO(b"mock_video_data")

        with patch("av.open") as mock_open, patch("av.codec.context.Flags2.EXPORT_MVS", create=True):
            # Mock frame with no motion vector side data
            mock_frame = Mock()
            mock_frame.height = 480
            mock_frame.width = 640
            mock_frame.side_data = []  # No side data

            # Mock stream
            mock_stream = Mock()
            mock_stream.codec_context = Mock()
            mock_stream.codec_context.flags2 = 0
            mock_stream.codec_context.thread_type = Mock()
            mock_stream.codec_context.thread_count = 4
            mock_stream.average_rate = 30.0
            mock_stream.duration = 30
            mock_stream.time_base = 1.0

            # Mock container
            mock_container = Mock()
            mock_container.streams.video = [mock_stream]
            mock_container.decode.return_value = iter([mock_frame])
            mock_container.duration = 30
            mock_container.__enter__ = Mock(return_value=mock_container)
            mock_container.__exit__ = Mock(return_value=None)

            mock_open.return_value = mock_container

            result = decode_for_motion(mock_video)

            assert isinstance(result, DecodedData)
            assert len(result.frames) == 0  # No motion vectors found

    def test_custom_parameters(self):
        """Test decode with custom parameters."""
        mock_video = io.BytesIO(b"mock_video_data")

        with patch("av.open") as mock_open, patch("av.codec.context.Flags2.EXPORT_MVS", create=True):
            # Mock motion vector side data
            mock_side_data = Mock()
            mock_side_data.type = Mock()
            mock_side_data.type.__eq__ = Mock(return_value=True)

            # Create proper structured array
            dtype = [
                ("field0", "i4"),
                ("field1", "i4"),
                ("field2", "i4"),
                ("field3", "i4"),
                ("field4", "i4"),
                ("field5", "i4"),
                ("field6", "i4"),
                ("field7", "i4"),
                ("field8", "i4"),
                ("field9", "i4"),
                ("field10", "i4"),
            ]
            structured_array = np.ones((5,), dtype=dtype)
            mock_side_data.to_ndarray.return_value = structured_array

            # Mock frame with valid resolution
            mock_frame = Mock()
            mock_frame.height = 480
            mock_frame.width = 640
            mock_frame.side_data = [mock_side_data]

            # Mock stream
            mock_stream = Mock()
            mock_stream.codec_context = Mock()
            mock_stream.codec_context.flags2 = 0
            mock_stream.codec_context.thread_type = Mock()
            mock_stream.codec_context.thread_count = 8
            mock_stream.average_rate = 30.0
            mock_stream.duration = 60
            mock_stream.time_base = 1.0

            # Mock container
            mock_container = Mock()
            mock_container.streams.video = [mock_stream]
            mock_container.decode.return_value = iter([mock_frame] * 5)
            mock_container.duration = 60
            mock_container.__enter__ = Mock(return_value=mock_container)
            mock_container.__exit__ = Mock(return_value=None)

            mock_open.return_value = mock_container

            with patch("av.sidedata.sidedata.Type.MOTION_VECTORS", create=True):
                result = decode_for_motion(mock_video, thread_count=8, target_fps=5.0, target_duration_ratio=0.3)

                assert isinstance(result, DecodedData)
                assert result.frame_size == torch.Size([480, 640, 3])


class TestCheckIfSmallMotion:
    """Test cases for check_if_small_motion function."""

    def test_small_motion_detection(self):
        """Test detection of small motion."""
        # Create test data with small motion
        mv_data = []
        for _ in range(3):
            # Create motion vector data (shape: (n_vectors, 11)) - original has 11 columns
            mv_frame = rng.random((10, 11)).astype(np.float32)
            mv_frame[:, 1] = 8.0  # block_sizes_x
            mv_frame[:, 2] = 8.0  # block_sizes_y
            mv_frame[:, 5] = rng.random(10) * 32  # dst_x
            mv_frame[:, 6] = rng.random(10) * 32  # dst_y
            mv_frame[:, 8] = rng.random(10) * 0.001  # motion_x (small)
            mv_frame[:, 9] = rng.random(10) * 0.001  # motion_y (small)
            mv_frame[:, 10] = np.ones(10)  # motion_scale
            mv_data.append(mv_frame)

        frame_shape = torch.Size([256, 256])

        result = check_if_small_motion(mv_data, frame_shape)

        assert isinstance(result, MotionInfo)
        assert result.is_small_motion is True
        assert result.global_mean < 0.001
        assert result.per_patch_min_256 < 0.001

    def test_large_motion_detection(self):
        """Test detection of large motion."""
        # Create test data with large motion
        mv_data = []
        for _ in range(3):
            # Create motion vector data with large motion
            mv_frame = rng.random((10, 11)).astype(np.float32)
            mv_frame[:, 1] = 8.0  # block_sizes_x
            mv_frame[:, 2] = 8.0  # block_sizes_y
            mv_frame[:, 5] = rng.random(10) * 100  # dst_x (spread across frame)
            mv_frame[:, 6] = rng.random(10) * 100  # dst_y (spread across frame)
            mv_frame[:, 8] = rng.random(10) * 200  # motion_x (large)
            mv_frame[:, 9] = rng.random(10) * 200  # motion_y (large)
            mv_frame[:, 10] = np.ones(10)  # motion_scale
            mv_data.append(mv_frame)

        frame_shape = torch.Size([256, 256])

        result = check_if_small_motion(
            mv_data, frame_shape, global_mean_threshold=0.001, per_patch_min_256_threshold=0.000001
        )

        assert isinstance(result, MotionInfo)
        # Test that motion values are calculated
        assert result.global_mean is not None
        assert result.per_patch_min_256 is not None
        # Function should return some motion detection result
        assert isinstance(result.is_small_motion, bool)

    def test_empty_motion_data(self):
        """Test with empty motion data."""
        mv_data = []
        frame_shape = torch.Size([256, 256])

        result = check_if_small_motion(mv_data, frame_shape)

        assert isinstance(result, MotionInfo)
        # With no data, the function returns NaN due to division by zero
        assert np.isnan(result.global_mean)
        assert np.isnan(result.per_patch_min_256)

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        # Create test data with moderate motion
        mv_data = []
        for _ in range(2):
            mv_frame = rng.random((5, 11)).astype(np.float32)
            mv_frame[:, 1] = 8.0  # block_sizes_x
            mv_frame[:, 2] = 8.0  # block_sizes_y
            mv_frame[:, 5] = rng.random(5) * 100  # dst_x
            mv_frame[:, 6] = rng.random(5) * 100  # dst_y
            mv_frame[:, 8] = rng.random(5) * 50  # motion_x (moderate)
            mv_frame[:, 9] = rng.random(5) * 50  # motion_y (moderate)
            mv_frame[:, 10] = np.ones(5)  # motion_scale
            mv_data.append(mv_frame)

        frame_shape = torch.Size([256, 256])

        # Test that different thresholds produce different results
        result_high = check_if_small_motion(
            mv_data,
            frame_shape,
            global_mean_threshold=1.0,  # Very high threshold
            per_patch_min_256_threshold=1.0,  # Very high threshold
        )

        result_low = check_if_small_motion(
            mv_data,
            frame_shape,
            global_mean_threshold=0.0,  # Very low threshold
            per_patch_min_256_threshold=0.0,  # Very low threshold
        )

        assert isinstance(result_high, MotionInfo)
        assert isinstance(result_low, MotionInfo)
        # With high threshold, motion should be considered small
        assert result_high.is_small_motion is True
        # Test that thresholds are applied correctly
        assert result_high.global_mean < 1.0
        assert result_high.per_patch_min_256 < 1.0

    def test_gpu_processing(self):
        """Test GPU processing when available."""
        # Create test data
        mv_data = []
        for _ in range(2):
            mv_frame = rng.random((5, 11)).astype(np.float32)
            mv_frame[:, 1] = 8.0  # block_sizes_x
            mv_frame[:, 2] = 8.0  # block_sizes_y
            mv_frame[:, 5] = rng.random(5) * 32  # dst_x
            mv_frame[:, 6] = rng.random(5) * 32  # dst_y
            mv_frame[:, 8] = rng.random(5) * 0.001  # motion_x (small)
            mv_frame[:, 9] = rng.random(5) * 0.001  # motion_y (small)
            mv_frame[:, 10] = np.ones(5)  # motion_scale
            mv_data.append(mv_frame)

        frame_shape = torch.Size([256, 256])

        # Test with GPU if available
        if torch.cuda.is_available():
            result = check_if_small_motion(mv_data, frame_shape, use_gpu=True)
            assert isinstance(result, MotionInfo)
        else:
            # Test that it falls back to CPU gracefully
            result = check_if_small_motion(mv_data, frame_shape, use_gpu=False)
            assert isinstance(result, MotionInfo)

    def test_batch_processing(self):
        """Test batch processing with different batch sizes."""
        # Create test data
        mv_data = []
        for _ in range(10):  # More frames to test batching
            mv_frame = rng.random((5, 11)).astype(np.float32)
            mv_frame[:, 1] = 8.0  # block_sizes_x
            mv_frame[:, 2] = 8.0  # block_sizes_y
            mv_frame[:, 5] = rng.random(5) * 32  # dst_x
            mv_frame[:, 6] = rng.random(5) * 32  # dst_y
            mv_frame[:, 8] = rng.random(5) * 0.001  # motion_x (small)
            mv_frame[:, 9] = rng.random(5) * 0.001  # motion_y (small)
            mv_frame[:, 10] = np.ones(5)  # motion_scale
            mv_data.append(mv_frame)

        frame_shape = torch.Size([256, 256])

        # Test with different batch sizes
        result_small_batch = check_if_small_motion(mv_data, frame_shape, batch_size=3)

        result_large_batch = check_if_small_motion(mv_data, frame_shape, batch_size=256)

        # Results should be consistent regardless of batch size
        assert isinstance(result_small_batch, MotionInfo)
        assert isinstance(result_large_batch, MotionInfo)
        assert result_small_batch.is_small_motion == result_large_batch.is_small_motion
