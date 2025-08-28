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

# ruff: noqa: ANN401, PT019

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nemo_curator.utils.nvcodec_utils import (
    FrameExtractionPolicy,
    NvVideoDecoder,
    PyNvcFrameExtractor,
    VideoBatchDecoder,
    gpu_decode_for_stitching,
    pixel_format_to_cvcuda_code,
)


class TestFrameExtractionPolicy:
    """Test suite for FrameExtractionPolicy enum."""

    def test_frame_extraction_policy_values(self) -> None:
        """Test that FrameExtractionPolicy has expected values."""
        assert FrameExtractionPolicy.full.value == 0
        assert FrameExtractionPolicy.fps.value == 1


class TestImportHandling:
    """Test suite for handling missing GPU dependencies."""

    def test_pixel_format_mapping_when_dependencies_missing(self) -> None:
        """Test that pixel format mapping is properly handled when dependencies are missing."""
        # Test that the mapping is a dictionary (empty or populated)
        assert isinstance(pixel_format_to_cvcuda_code, dict)
        # When dependencies are missing, it should be empty
        # When dependencies are present, it should have entries
        # Both cases are valid

    @patch("nemo_curator.utils.nvcodec_utils.Nvc", None)
    def test_video_batch_decoder_init_without_dependencies(self) -> None:
        """Test VideoBatchDecoder raises error when PyNvVideoCodec is not available."""
        with pytest.raises(RuntimeError, match="PyNvVideoCodec is not available"):
            VideoBatchDecoder(
                batch_size=2,
                target_width=224,
                target_height=224,
                device_id=0,
                cuda_ctx=Mock(),
                cvcuda_stream=Mock(),
            )

    @patch("nemo_curator.utils.nvcodec_utils.cuda", None)
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda", None)
    @patch("nemo_curator.utils.nvcodec_utils.nvcv", None)
    @patch("nemo_curator.utils.nvcodec_utils.Nvc", None)
    def test_py_nvc_frame_extractor_without_dependencies(self) -> None:
        """Test PyNvcFrameExtractor fails gracefully when dependencies are missing."""
        with pytest.raises((RuntimeError, AttributeError)):
            PyNvcFrameExtractor(width=224, height=224, batch_size=2)

    @patch("nemo_curator.utils.nvcodec_utils.cvcuda", None)
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    def test_gpu_decode_for_stitching_without_cvcuda(self, _mock_torch: Any) -> None:
        """Test gpu_decode_for_stitching fails gracefully when cvcuda is missing."""
        with pytest.raises(AttributeError):
            gpu_decode_for_stitching(
                device_id=0,
                ctx=Mock(),
                stream=Mock(),
                input_path=Path("test.mp4"),
                frame_list=[0, 1],
                batch_size=2,
            )

    def test_module_imports_gracefully_without_dependencies(self) -> None:
        """Test that the module can be imported even when GPU dependencies are missing."""
        # If we got here, the import was successful
        # This test verifies that import failures are handled gracefully
        from nemo_curator.utils import nvcodec_utils

        # Verify the module has expected attributes
        assert hasattr(nvcodec_utils, "FrameExtractionPolicy")
        assert hasattr(nvcodec_utils, "VideoBatchDecoder")
        assert hasattr(nvcodec_utils, "NvVideoDecoder")
        assert hasattr(nvcodec_utils, "PyNvcFrameExtractor")


class TestVideoBatchDecoder:
    """Test suite for VideoBatchDecoder class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 4
        self.target_width = 224
        self.target_height = 224
        self.device_id = 0
        self.mock_cuda_ctx = Mock()
        self.mock_cvcuda_stream = Mock()

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")  # Ensure Nvc is available for this test
    def test_init_valid_params(self, _mock_nvc: Any) -> None:
        """Test initialization with valid parameters."""
        decoder = VideoBatchDecoder(
            batch_size=self.batch_size,
            target_width=self.target_width,
            target_height=self.target_height,
            device_id=self.device_id,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        assert decoder.batch_size == self.batch_size
        assert decoder.target_width == self.target_width
        assert decoder.target_height == self.target_height
        assert decoder.device_id == self.device_id
        assert decoder.cuda_ctx == self.mock_cuda_ctx
        assert decoder.cvcuda_stream == self.mock_cvcuda_stream
        assert decoder.decoder is None
        assert decoder.input_path is None

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")  # Ensure Nvc is available for this test
    def test_init_invalid_batch_size(self, _mock_nvc: Any) -> None:
        """Test initialization with invalid batch size."""
        with pytest.raises(ValueError, match="Batch size should be a valid number"):
            VideoBatchDecoder(
                batch_size=0,
                target_width=self.target_width,
                target_height=self.target_height,
                device_id=self.device_id,
                cuda_ctx=self.mock_cuda_ctx,
                cvcuda_stream=self.mock_cvcuda_stream,
            )

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")  # Ensure Nvc is available for this test
    def test_get_fps_no_decoder(self, _mock_nvc: Any) -> None:
        """Test get_fps when no decoder is initialized."""
        decoder = VideoBatchDecoder(
            batch_size=self.batch_size,
            target_width=self.target_width,
            target_height=self.target_height,
            device_id=self.device_id,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        assert decoder.get_fps() is None

    @patch("nemo_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.Nvc")  # Ensure Nvc is available
    def test_call_first_time(self, _mock_nvc: Any, _mock_torch: Any, _mock_cvcuda: Any, mock_nvdecoder: Any) -> None:
        """Test calling decoder for the first time."""
        # Setup mocks
        mock_decoder_instance = Mock()
        mock_decoder_instance.nvDemux.FrameRate.return_value = 30
        mock_decoder_instance.w = 640
        mock_decoder_instance.h = 480
        mock_decoder_instance.pixelFormat = Mock()
        mock_decoder_instance.get_next_frames.return_value = None
        mock_nvdecoder.return_value = mock_decoder_instance

        # Mock pixel format mapping
        with patch.dict(
            "nemo_curator.utils.nvcodec_utils.pixel_format_to_cvcuda_code",
            {mock_decoder_instance.pixelFormat: "YUV2RGB"},
        ):
            decoder = VideoBatchDecoder(
                batch_size=self.batch_size,
                target_width=self.target_width,
                target_height=self.target_height,
                device_id=self.device_id,
                cuda_ctx=self.mock_cuda_ctx,
                cvcuda_stream=self.mock_cvcuda_stream,
            )

            result = decoder("test_video.mp4")

            # Verify decoder was created
            mock_nvdecoder.assert_called_once()
            assert decoder.fps == 30
            assert result is None  # No frames returned

    @patch("nemo_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.Nvc")  # Ensure Nvc is available
    def test_call_unsupported_pixel_format(
        self, _mock_nvc: Any, _mock_torch: Any, _mock_cvcuda: Any, mock_nvdecoder: Any
    ) -> None:
        """Test calling decoder with unsupported pixel format."""
        # Setup mocks
        mock_decoder_instance = Mock()
        mock_decoder_instance.nvDemux.FrameRate.return_value = 30
        mock_decoder_instance.w = 640
        mock_decoder_instance.h = 480
        mock_decoder_instance.pixelFormat = "UNSUPPORTED_FORMAT"
        mock_decoder_instance.get_next_frames.return_value = Mock()
        mock_nvdecoder.return_value = mock_decoder_instance

        # Mock torch tensor
        mock_yuv_tensor = Mock()
        mock_yuv_tensor.cuda.return_value = mock_yuv_tensor
        mock_decoder_instance.get_next_frames.return_value = mock_yuv_tensor

        decoder = VideoBatchDecoder(
            batch_size=self.batch_size,
            target_width=self.target_width,
            target_height=self.target_height,
            device_id=self.device_id,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        with pytest.raises(ValueError, match="Unsupported pixel format"):
            decoder("test_video.mp4")

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")  # Ensure Nvc is available for this test
    def test_call_no_decoder(self, _mock_nvc: Any) -> None:
        """Test calling decoder when not initialized."""
        decoder = VideoBatchDecoder(
            batch_size=self.batch_size,
            target_width=self.target_width,
            target_height=self.target_height,
            device_id=self.device_id,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        # Manually set input_path to trigger the decoder is None check
        decoder.input_path = "test_video.mp4"

        with pytest.raises(RuntimeError, match="Decoder is not initialized"):
            decoder("test_video.mp4")

    @patch("nemo_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    def test_call_dynamic_sizing(self, _mock_nvc: Any, mock_torch: Any, mock_cvcuda: Any, mock_nvdecoder: Any) -> None:
        """Test calling decoder with dynamic width/height calculation (-1 values)."""
        # Setup mocks
        mock_decoder_instance = Mock()
        mock_decoder_instance.nvDemux.FrameRate.return_value = 30
        mock_decoder_instance.w = 640
        mock_decoder_instance.h = 480
        mock_decoder_instance.pixelFormat = Mock()
        mock_yuv_tensor = Mock()
        mock_yuv_tensor.cuda.return_value = mock_yuv_tensor
        mock_yuv_tensor.shape = [2, 480, 640, 3]  # NHWC format
        mock_decoder_instance.get_next_frames.return_value = mock_yuv_tensor
        mock_nvdecoder.return_value = mock_decoder_instance

        # Mock cvcuda tensor
        mock_cvcuda_tensor = Mock()
        mock_cvcuda_tensor.layout = "NHWC"
        mock_cvcuda_tensor.shape = [2, 480, 640, 3]
        mock_cvcuda.as_tensor.return_value = mock_cvcuda_tensor

        # Mock pixel format mapping
        with patch.dict(
            "nemo_curator.utils.nvcodec_utils.pixel_format_to_cvcuda_code",
            {mock_decoder_instance.pixelFormat: "YUV2RGB"},
        ):
            # Use -1 for dynamic sizing
            decoder = VideoBatchDecoder(
                batch_size=self.batch_size,
                target_width=-1,  # Dynamic width
                target_height=-1,  # Dynamic height
                device_id=self.device_id,
                cuda_ctx=self.mock_cuda_ctx,
                cvcuda_stream=self.mock_cvcuda_stream,
            )

            # Mock torch operations
            mock_rgb_tensor = Mock()
            mock_rgb_tensor.shape = [2, 480, 640, 3]
            mock_torch.empty.return_value = mock_rgb_tensor
            mock_cvcuda.as_tensor.return_value = mock_cvcuda_tensor

            decoder("test_video.mp4")

            # Verify dynamic sizing was calculated (width=640, minimum_width=256, so downscale_factor=2)
            # target_width should be 320, target_height should be 240
            assert decoder.target_width == 320  # round(640 / 2)
            assert decoder.target_height == 240  # round(480 / 2)

    @patch("nemo_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    def test_call_unexpected_layout_error(
        self, _mock_nvc: Any, _mock_torch: Any, mock_cvcuda: Any, mock_nvdecoder: Any
    ) -> None:
        """Test calling decoder with unexpected tensor layout."""
        # Setup mocks
        mock_decoder_instance = Mock()
        mock_decoder_instance.nvDemux.FrameRate.return_value = 30
        mock_decoder_instance.w = 640
        mock_decoder_instance.h = 480
        mock_decoder_instance.pixelFormat = Mock()
        mock_yuv_tensor = Mock()
        mock_yuv_tensor.cuda.return_value = mock_yuv_tensor
        mock_decoder_instance.get_next_frames.return_value = mock_yuv_tensor
        mock_nvdecoder.return_value = mock_decoder_instance

        # Mock cvcuda tensor with unexpected layout
        mock_cvcuda_tensor = Mock()
        mock_cvcuda_tensor.layout = "NCHW"  # Wrong layout, should be NHWC
        mock_cvcuda.as_tensor.return_value = mock_cvcuda_tensor

        # Mock pixel format mapping
        with patch.dict(
            "nemo_curator.utils.nvcodec_utils.pixel_format_to_cvcuda_code",
            {mock_decoder_instance.pixelFormat: "YUV2RGB"},
        ):
            decoder = VideoBatchDecoder(
                batch_size=self.batch_size,
                target_width=self.target_width,
                target_height=self.target_height,
                device_id=self.device_id,
                cuda_ctx=self.mock_cuda_ctx,
                cvcuda_stream=self.mock_cvcuda_stream,
            )

            with pytest.raises(ValueError, match="Unexpected tensor layout, NHWC expected"):
                decoder("test_video.mp4")

    @patch("nemo_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    def test_call_full_processing_pipeline(
        self, _mock_nvc: Any, mock_torch: Any, mock_cvcuda: Any, mock_nvdecoder: Any
    ) -> None:
        """Test the complete processing pipeline with all tensor operations."""
        # Setup mocks
        mock_decoder_instance = Mock()
        mock_decoder_instance.nvDemux.FrameRate.return_value = 30
        mock_decoder_instance.w = 640
        mock_decoder_instance.h = 480
        mock_decoder_instance.pixelFormat = Mock()
        mock_yuv_tensor = Mock()
        mock_yuv_tensor.cuda.return_value = mock_yuv_tensor
        mock_decoder_instance.get_next_frames.return_value = mock_yuv_tensor
        mock_nvdecoder.return_value = mock_decoder_instance

        # Mock cvcuda tensor operations
        mock_cvcuda_yuv_tensor = Mock()
        mock_cvcuda_yuv_tensor.layout = "NHWC"
        mock_cvcuda_yuv_tensor.shape = [2, 480, 640, 3]

        mock_cvcuda_rgb_tensor = Mock()
        mock_cvcuda_rgb_tensor.shape = [2, 480, 640, 3]

        mock_cvcuda_rgb_tensor_resized = Mock()

        # Setup side effects for as_tensor calls
        def as_tensor_side_effect(*args: Any, **_kwargs: Any) -> Any:
            if len(args) >= 2 and args[1] == "NHWC":
                # First call for YUV tensor
                if not hasattr(as_tensor_side_effect, "call_count"):
                    as_tensor_side_effect.call_count = 0
                as_tensor_side_effect.call_count += 1

                if as_tensor_side_effect.call_count == 1:
                    return mock_cvcuda_yuv_tensor
                elif as_tensor_side_effect.call_count == 2:
                    return mock_cvcuda_rgb_tensor
                else:
                    return mock_cvcuda_rgb_tensor_resized
            return Mock()

        mock_cvcuda.as_tensor.side_effect = as_tensor_side_effect

        # Mock torch tensor creation
        mock_rgb_tensor = Mock()
        mock_rgb_resized_tensor = Mock()

        def torch_empty_side_effect(shape: Any, **_kwargs: Any) -> Any:
            if shape == (2, 480, 640, 3):
                return mock_rgb_tensor
            else:
                return mock_rgb_resized_tensor

        mock_torch.empty.side_effect = torch_empty_side_effect

        # Mock pixel format mapping
        with patch.dict(
            "nemo_curator.utils.nvcodec_utils.pixel_format_to_cvcuda_code",
            {mock_decoder_instance.pixelFormat: "YUV2RGB"},
        ):
            decoder = VideoBatchDecoder(
                batch_size=2,
                target_width=self.target_width,
                target_height=self.target_height,
                device_id=self.device_id,
                cuda_ctx=self.mock_cuda_ctx,
                cvcuda_stream=self.mock_cvcuda_stream,
            )

            result = decoder("test_video.mp4")

            # Verify all the processing steps were called
            mock_cvcuda.cvtcolor_into.assert_called_once()
            mock_cvcuda.resize_into.assert_called_once()

            # Verify tensor creation calls
            assert mock_torch.empty.call_count >= 2  # At least RGB and resized tensor creation

            # Should return the resized tensor
            assert result == mock_rgb_resized_tensor

    @patch("nemo_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    def test_call_batch_size_variation(
        self, _mock_nvc: Any, mock_torch: Any, mock_cvcuda: Any, mock_nvdecoder: Any
    ) -> None:
        """Test handling of different batch sizes (last batch may be smaller)."""
        # Setup mocks
        mock_decoder_instance = Mock()
        mock_decoder_instance.nvDemux.FrameRate.return_value = 30
        mock_decoder_instance.w = 640
        mock_decoder_instance.h = 480
        mock_decoder_instance.pixelFormat = Mock()
        mock_yuv_tensor = Mock()
        mock_yuv_tensor.cuda.return_value = mock_yuv_tensor
        mock_decoder_instance.get_next_frames.return_value = mock_yuv_tensor
        mock_nvdecoder.return_value = mock_decoder_instance

        # Mock cvcuda tensor with smaller batch size
        mock_cvcuda_yuv_tensor = Mock()
        mock_cvcuda_yuv_tensor.layout = "NHWC"
        mock_cvcuda_yuv_tensor.shape = [1, 480, 640, 3]  # Only 1 frame instead of batch_size 4
        mock_cvcuda.as_tensor.return_value = mock_cvcuda_yuv_tensor

        # Mock tensor creation - need enough calls for both decoder runs
        mock_rgb_tensor = Mock()
        mock_resized_tensor = Mock()
        mock_torch.empty.side_effect = [mock_rgb_tensor, mock_resized_tensor, mock_rgb_tensor, mock_resized_tensor]

        # Mock pixel format mapping
        with patch.dict(
            "nemo_curator.utils.nvcodec_utils.pixel_format_to_cvcuda_code",
            {mock_decoder_instance.pixelFormat: "YUV2RGB"},
        ):
            decoder = VideoBatchDecoder(
                batch_size=4,  # Larger than actual batch
                target_width=self.target_width,
                target_height=self.target_height,
                device_id=self.device_id,
                cuda_ctx=self.mock_cuda_ctx,
                cvcuda_stream=self.mock_cvcuda_stream,
            )

            # First call to establish previous batch size
            decoder("test_video.mp4")

            # Second call with different batch size should trigger reallocation
            decoder("test_video.mp4")

            # Should create new tensors for the different batch size
            assert mock_torch.empty.call_count >= 2


class TestNvVideoDecoder:
    """Test suite for NvVideoDecoder class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.device_id = 0
        self.batch_size = 4
        self.mock_cuda_ctx = Mock()
        self.mock_cuda_ctx.handle = Mock()
        self.mock_cvcuda_stream = Mock()
        self.mock_cvcuda_stream.handle = Mock()

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.logger")
    def test_init(self, _mock_logger: Any, mock_nvc: Any) -> None:
        """Test NvVideoDecoder initialization."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=self.batch_size,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        assert decoder.device_id == self.device_id
        assert decoder.batch_size == self.batch_size
        assert decoder.w == 640
        assert decoder.h == 480
        assert decoder.pixelFormat == "NV12"
        assert decoder.decoded_frame_cnt == 0
        assert decoder.local_frame_index == 0
        assert decoder.sent_frame_cnt == 0

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.nvcv")
    def test_generate_decoded_frames(
        self, _mock_nvcv: Any, _mock_cvcuda: Any, _mock_torch: Any, mock_nvc: Any
    ) -> None:
        """Test generate_decoded_frames method."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_decoder.Decode.return_value = []
        mock_nvc.CreateDecoder.return_value = mock_decoder

        # Mock demux iteration (no packets)
        mock_demux.__iter__ = Mock(return_value=iter([]))

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=self.batch_size,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        result = decoder.generate_decoded_frames()
        assert result == []

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    def test_get_next_frames_no_frames(self, mock_nvc: Any) -> None:
        """Test get_next_frames when no frames available."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=self.batch_size,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        # Mock generate_decoded_frames to return empty list
        decoder.generate_decoded_frames = Mock(return_value=[])

        result = decoder.get_next_frames()
        assert result is None

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    def test_get_next_frames_single_frame(self, _mock_torch: Any, mock_nvc: Any) -> None:
        """Test get_next_frames with single frame."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=self.batch_size,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        # Mock generate_decoded_frames to return single frame
        mock_frame = Mock()
        decoder.generate_decoded_frames = Mock(return_value=[mock_frame])

        result = decoder.get_next_frames()
        assert result == mock_frame

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    def test_get_next_frames_multiple_frames(self, _mock_torch: Any, mock_nvc: Any) -> None:
        """Test get_next_frames with multiple frames."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=self.batch_size,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        # Mock generate_decoded_frames to return multiple frames
        mock_frames = [Mock(), Mock()]
        decoder.generate_decoded_frames = Mock(return_value=mock_frames)

        # Mock torch.cat
        mock_result = Mock()
        _mock_torch.cat.return_value = mock_result

        result = decoder.get_next_frames()
        assert result == mock_result
        _mock_torch.cat.assert_called_once_with(mock_frames)

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.nvcv")
    def test_generate_decoded_frames_with_frames(
        self, mock_nvcv: Any, mock_cvcuda: Any, mock_torch: Any, mock_nvc: Any
    ) -> None:
        """Test generate_decoded_frames with actual frame processing."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        # Mock packet and decoded frame
        mock_packet = Mock()
        mock_decoded_frame = Mock()
        mock_decoded_frame.nvcv_image.return_value = Mock()

        # Mock tensor operations
        mock_nvcv_tensor = Mock()
        mock_nvcv_tensor.layout = "NCHW"
        mock_nvcv_tensor.shape = (1, 3, 480, 640)  # NCHW format
        mock_nvcv.as_tensor.return_value = mock_nvcv_tensor
        mock_nvcv.as_image.return_value = Mock()
        mock_nvcv.Format.U8 = Mock()

        # Mock torch tensor
        mock_torch_nhwc = Mock()
        mock_torch.empty.return_value = mock_torch_nhwc

        # Mock cvcuda tensor
        mock_cvcuda_nhwc = Mock()
        mock_cvcuda.as_tensor.return_value = mock_cvcuda_nhwc

        # Mock demux iteration - return one packet with one frame
        mock_demux.__iter__ = Mock(return_value=iter([mock_packet]))
        mock_decoder.Decode.return_value = [mock_decoded_frame]

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=1,  # Set batch_size to 1 to test the batch logic
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        result = decoder.generate_decoded_frames()

        # Verify frame processing was called
        mock_nvcv.as_tensor.assert_called_once()
        mock_torch.empty.assert_called_once()
        mock_cvcuda.as_tensor.assert_called_once()
        mock_cvcuda.reformat_into.assert_called_once()

        # Should return the processed frames
        assert result == [mock_torch_nhwc]

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.nvcv")
    def test_generate_decoded_frames_unexpected_layout(
        self, mock_nvcv: Any, _mock_cvcuda: Any, _mock_torch: Any, mock_nvc: Any
    ) -> None:
        """Test generate_decoded_frames with unexpected tensor layout."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        # Mock packet and decoded frame
        mock_packet = Mock()
        mock_decoded_frame = Mock()
        mock_decoded_frame.nvcv_image.return_value = Mock()

        # Mock tensor with unexpected layout
        mock_nvcv_tensor = Mock()
        mock_nvcv_tensor.layout = "NHWC"  # Unexpected layout - should be NCHW
        mock_nvcv.as_tensor.return_value = mock_nvcv_tensor
        mock_nvcv.as_image.return_value = Mock()
        mock_nvcv.Format.U8 = Mock()

        # Mock demux iteration
        mock_demux.__iter__ = Mock(return_value=iter([mock_packet]))
        mock_decoder.Decode.return_value = [mock_decoded_frame]

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=self.batch_size,
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        with pytest.raises(ValueError, match="Unexpected tensor layout, NCHW expected"):
            decoder.generate_decoded_frames()

    @patch("nemo_curator.utils.nvcodec_utils.Nvc")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.nvcv")
    def test_generate_decoded_frames_partial_batch(
        self, mock_nvcv: Any, mock_cvcuda: Any, mock_torch: Any, mock_nvc: Any
    ) -> None:
        """Test generate_decoded_frames with partial batch (less frames than batch_size)."""
        # Setup mocks
        mock_demux = Mock()
        mock_demux.Width.return_value = 640
        mock_demux.Height.return_value = 480
        mock_demux.GetNvCodecId.return_value = "H264"
        mock_nvc.PyNvDemuxer.return_value = mock_demux

        mock_decoder = Mock()
        mock_decoder.GetPixelFormat.return_value = "NV12"
        mock_nvc.CreateDecoder.return_value = mock_decoder

        # Mock multiple packets with frames
        mock_packets = [Mock() for _ in range(3)]
        mock_frames = []
        for _ in range(2):  # Only 2 frames, less than batch_size of 4
            frame = Mock()
            frame.nvcv_image.return_value = Mock()
            mock_frames.append(frame)

        # Mock tensor operations
        mock_nvcv_tensor = Mock()
        mock_nvcv_tensor.layout = "NCHW"
        mock_nvcv_tensor.shape = (1, 3, 480, 640)
        mock_nvcv.as_tensor.return_value = mock_nvcv_tensor
        mock_nvcv.as_image.return_value = Mock()
        mock_nvcv.Format.U8 = Mock()

        mock_torch_nhwc = Mock()
        mock_torch.empty.return_value = mock_torch_nhwc
        mock_cvcuda_nhwc = Mock()
        mock_cvcuda.as_tensor.return_value = mock_cvcuda_nhwc

        # Mock demux to return frames across multiple packets then end
        decode_calls = []
        decode_calls.extend(
            [mock_frames[:1], mock_frames[1:2], []]
        )  # First packet has 1 frame, second has 1 frame, third has none
        mock_decoder.Decode.side_effect = decode_calls

        mock_demux.__iter__ = Mock(return_value=iter(mock_packets))

        decoder = NvVideoDecoder(
            enc_file="test_video.mp4",
            device_id=self.device_id,
            batch_size=4,  # Larger than number of frames available
            cuda_ctx=self.mock_cuda_ctx,
            cvcuda_stream=self.mock_cvcuda_stream,
        )

        # Call should process and return frames based on mocked demux iteration
        result = decoder.generate_decoded_frames()

        # Should return the processed frames (2 frames from the 2 packets)
        assert len(result) == 2

        # Verify frame processing was called
        mock_nvcv.as_tensor.assert_called()
        mock_torch.empty.assert_called()
        mock_cvcuda.as_tensor.assert_called()
        mock_cvcuda.reformat_into.assert_called()


class TestPyNvcFrameExtractor:
    """Test suite for PyNvcFrameExtractor class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.width = 224
        self.height = 224
        self.batch_size = 4

    @patch("nemo_curator.utils.nvcodec_utils.cuda")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_init(self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any) -> None:
        """Test PyNvcFrameExtractor initialization."""
        # Setup mocks
        mock_device = Mock()
        mock_cuda.Device.return_value = mock_device
        mock_ctx = Mock()
        mock_device.retain_primary_context.return_value = mock_ctx
        mock_stream = Mock()
        mock_stream.handle = Mock()
        mock_cvcuda.Stream.return_value = mock_stream
        mock_torch.cuda.ExternalStream.return_value = Mock()

        PyNvcFrameExtractor(
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
        )

        # Verify decoder was created with correct parameters
        mock_decoder_class.assert_called_once_with(
            self.batch_size,
            self.width,
            self.height,
            0,  # device_id
            mock_ctx,
            mock_stream,
        )

    @patch("nemo_curator.utils.nvcodec_utils.cuda")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_call_full_extraction(
        self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any
    ) -> None:
        """Test frame extraction with full policy."""
        # Setup mocks
        mock_device = Mock()
        mock_cuda.Device.return_value = mock_device
        mock_ctx = Mock()
        mock_device.retain_primary_context.return_value = mock_ctx
        mock_stream = Mock()
        mock_stream.handle = Mock()
        mock_cvcuda.Stream.return_value = mock_stream
        mock_torch.cuda.ExternalStream.return_value = Mock()

        # Mock decoder
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock decoder behavior: return two batches then None
        mock_batch1 = Mock()
        mock_batch2 = Mock()
        mock_decoder.side_effect = [mock_batch1, mock_batch2, None]

        # Mock torch.cat
        mock_result = Mock()
        mock_torch.cat.return_value = mock_result

        extractor = PyNvcFrameExtractor(
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
        )

        result = extractor(Path("test_video.mp4"))

        # Verify torch.cat was called with the batches
        mock_torch.cat.assert_called_once_with([mock_batch1, mock_batch2], dim=0)
        assert result == mock_result

    @patch("nemo_curator.utils.nvcodec_utils.cuda")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_call_fps_extraction(
        self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any
    ) -> None:
        """Test frame extraction with FPS policy."""
        # Setup mocks
        mock_device = Mock()
        mock_cuda.Device.return_value = mock_device
        mock_ctx = Mock()
        mock_device.retain_primary_context.return_value = mock_ctx
        mock_stream = Mock()
        mock_stream.handle = Mock()
        mock_cvcuda.Stream.return_value = mock_stream
        mock_torch.cuda.ExternalStream.return_value = Mock()

        # Mock decoder
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.get_fps.return_value = 30

        # Mock batch with shape that supports slicing
        mock_batch = Mock()
        mock_batch.__getitem__ = Mock(return_value=Mock())
        mock_decoder.side_effect = [mock_batch, None]

        # Mock torch.cat
        mock_result = Mock()
        mock_torch.cat.return_value = mock_result

        extractor = PyNvcFrameExtractor(
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
        )

        result = extractor(
            Path("test_video.mp4"),
            extraction_policy=FrameExtractionPolicy.fps,
            sampling_fps=2,
        )

        # Verify FPS-based sampling was applied
        mock_batch.__getitem__.assert_called_once_with(slice(None, None, 15))  # 30 / 2 = 15
        assert result == mock_result

    @patch("nemo_curator.utils.nvcodec_utils.cuda")
    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_call_fps_extraction_no_fps(
        self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any
    ) -> None:
        """Test frame extraction with FPS policy when FPS is unavailable."""
        # Setup mocks
        mock_device = Mock()
        mock_cuda.Device.return_value = mock_device
        mock_ctx = Mock()
        mock_device.retain_primary_context.return_value = mock_ctx
        mock_stream = Mock()
        mock_stream.handle = Mock()
        mock_cvcuda.Stream.return_value = mock_stream
        mock_torch.cuda.ExternalStream.return_value = Mock()

        # Mock decoder
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.get_fps.return_value = None

        # Mock batch
        mock_batch = Mock()
        mock_decoder.side_effect = [mock_batch, None]

        extractor = PyNvcFrameExtractor(
            width=self.width,
            height=self.height,
            batch_size=self.batch_size,
        )

        with pytest.raises(RuntimeError, match="Unable to get video FPS"):
            extractor(
                Path("test_video.mp4"),
                extraction_policy=FrameExtractionPolicy.fps,
                sampling_fps=2,
            )


class TestGpuDecodeForStitching:
    """Test suite for gpu_decode_for_stitching function."""

    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_gpu_decode_for_stitching(self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any) -> None:
        """Test gpu_decode_for_stitching function."""
        # Setup mocks
        mock_stream = Mock()
        mock_stream.handle = Mock()
        mock_cvcuda.cuda.as_stream.return_value = mock_stream
        mock_torch.cuda.ExternalStream.return_value = Mock()

        # Mock decoder
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock batch with specific shape for testing
        mock_batch = Mock()
        mock_batch.shape = [2]  # 2 frames in batch
        mock_batch.cuda.return_value = mock_batch
        mock_batch.__getitem__ = Mock(side_effect=lambda _: Mock())
        mock_decoder.side_effect = [mock_batch, None]

        # Mock torch.as_tensor
        mock_torch.as_tensor.return_value = mock_batch

        device_id = 0
        ctx = Mock()
        stream = Mock()
        input_path = Path("test_video.mp4")
        frame_list = [0, 1, 1]  # Frame 1 appears twice
        batch_size = 2

        result = gpu_decode_for_stitching(
            device_id=device_id,
            ctx=ctx,
            stream=stream,
            input_path=input_path,
            frame_list=frame_list,
            batch_size=batch_size,
        )

        # Verify decoder was created
        mock_decoder_class.assert_called_once_with(
            batch_size,
            224,
            224,
            device_id,
            ctx,
            mock_stream,
        )

        # Verify result is a list
        assert isinstance(result, list)
        # Should have 3 frames (frame 1 appears twice)
        assert len(result) == 3

    @patch("nemo_curator.utils.nvcodec_utils.cvcuda")
    @patch("nemo_curator.utils.nvcodec_utils.torch")
    @patch("nemo_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_gpu_decode_for_stitching_complex_frame_list(
        self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any
    ) -> None:
        """Test gpu_decode_for_stitching with complex frame list scenarios."""
        # Setup mocks
        mock_stream = Mock()
        mock_stream.handle = Mock()
        mock_cvcuda.cuda.as_stream.return_value = mock_stream
        mock_torch.cuda.ExternalStream.return_value = Mock()

        # Mock decoder
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock multiple batches with different frame counts
        mock_batch1 = Mock()
        mock_batch1.shape = [3]  # 3 frames in first batch
        mock_batch1.cuda.return_value = mock_batch1

        # Handle tensor indexing - idx will be like (0, :, :, :)
        def batch1_getitem(idx: Any) -> str:
            if isinstance(idx, tuple) and len(idx) > 0:
                return f"frame_{idx[0]}"
            return f"frame_{idx}"

        mock_batch1.__getitem__ = Mock(side_effect=batch1_getitem)

        mock_batch2 = Mock()
        mock_batch2.shape = [2]  # 2 frames in second batch
        mock_batch2.cuda.return_value = mock_batch2

        # Handle tensor indexing - idx will be like (1, :, :, :) for frame index 4 (4-3=1)
        def batch2_getitem(idx: Any) -> str:
            if isinstance(idx, tuple) and len(idx) > 0:
                return f"frame_{idx[0] + 3}"
            return f"frame_{idx + 3}"

        mock_batch2.__getitem__ = Mock(side_effect=batch2_getitem)

        mock_decoder.side_effect = [mock_batch1, mock_batch2, None]

        # Mock torch.as_tensor
        def as_tensor_side_effect(tensor: Any, *_args: Any, **_kwargs: Any) -> Any:
            return tensor

        mock_torch.as_tensor.side_effect = as_tensor_side_effect

        device_id = 0
        ctx = Mock()
        stream = Mock()
        input_path = Path("test_video.mp4")
        # Complex frame list with repeats and gaps
        frame_list = [0, 1, 1, 2, 4, 4, 4]  # Some frames repeat, frame 3 is skipped, frame 4 repeats 3 times
        batch_size = 3

        result = gpu_decode_for_stitching(
            device_id=device_id,
            ctx=ctx,
            stream=stream,
            input_path=input_path,
            frame_list=frame_list,
            batch_size=batch_size,
        )

        # Verify correct number of frames returned
        # Frame 0: 1 occurrence -> 1 result
        # Frame 1: 2 occurrences -> 2 results
        # Frame 2: 1 occurrence -> 1 result
        # Frame 4: 3 occurrences -> 3 results (frame 4 is in second batch at index 1)
        # Total: 7 results
        assert len(result) == 7


class TestPixelFormatMapping:
    """Test suite for pixel format mapping."""

    def test_pixel_format_mapping_exists(self) -> None:
        """Test that pixel format mapping dictionary exists and has expected keys."""
        # This test verifies the mapping exists and has the expected structure
        assert isinstance(pixel_format_to_cvcuda_code, dict)
        # When dependencies are missing, dict is empty
        # When dependencies are present, dict has entries
        # Both cases are valid


class TestGracefulDegradation:
    """Test suite for ensuring graceful degradation when dependencies are missing."""

    def test_import_without_gpu_dependencies(self) -> None:
        """Test that module imports successfully even without GPU dependencies."""
        # This test passes if we can import the module, which we already did at the top
        # It serves as documentation that this is an important requirement
        assert True  # If we got here, the import worked

    def test_error_messages_are_helpful(self) -> None:
        """Test that error messages guide users to install missing dependencies."""
        with (
            patch("nemo_curator.utils.nvcodec_utils.Nvc", None),
            pytest.raises(RuntimeError, match="PyNvVideoCodec is not available"),
        ):
            VideoBatchDecoder(
                batch_size=2,
                target_width=224,
                target_height=224,
                device_id=0,
                cuda_ctx=Mock(),
                cvcuda_stream=Mock(),
            )

    def test_all_classes_can_be_imported(self) -> None:
        """Test that all public classes can be imported regardless of dependency availability."""
        # All these should be importable even when dependencies are missing
        from nemo_curator.utils.nvcodec_utils import (
            FrameExtractionPolicy,
            NvVideoDecoder,
            PyNvcFrameExtractor,
            VideoBatchDecoder,
            gpu_decode_for_stitching,
        )

        # Verify they're actually classes/functions
        assert FrameExtractionPolicy is not None
        assert VideoBatchDecoder is not None
        assert NvVideoDecoder is not None
        assert PyNvcFrameExtractor is not None
        assert callable(gpu_decode_for_stitching)
