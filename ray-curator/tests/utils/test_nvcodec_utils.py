"""Test suite for nvcodec_utils module."""
# ruff: noqa: ANN401, PT019

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ray_curator.utils.nvcodec_utils import (
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

    def test_init_valid_params(self) -> None:
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

    def test_init_invalid_batch_size(self) -> None:
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

    def test_get_fps_no_decoder(self) -> None:
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

    @patch("ray_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    def test_call_first_time(self, _mock_torch: Any, _mock_cvcuda: Any, mock_nvdecoder: Any) -> None:
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
        with patch.dict("ray_curator.utils.nvcodec_utils.pixel_format_to_cvcuda_code",
                       {mock_decoder_instance.pixelFormat: "YUV2RGB"}):

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

    @patch("ray_curator.utils.nvcodec_utils.NvVideoDecoder")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    def test_call_unsupported_pixel_format(self, _mock_torch: Any, _mock_cvcuda: Any, mock_nvdecoder: Any) -> None:
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

    def test_call_no_decoder(self) -> None:
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

    @patch("ray_curator.utils.nvcodec_utils.Nvc")
    @patch("ray_curator.utils.nvcodec_utils.logger")
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

    @patch("ray_curator.utils.nvcodec_utils.Nvc")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.nvcv")
    def test_generate_decoded_frames(self, _mock_nvcv: Any, _mock_cvcuda: Any, _mock_torch: Any, mock_nvc: Any) -> None:
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

    @patch("ray_curator.utils.nvcodec_utils.Nvc")
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

    @patch("ray_curator.utils.nvcodec_utils.Nvc")
    @patch("ray_curator.utils.nvcodec_utils.torch")
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

    @patch("ray_curator.utils.nvcodec_utils.Nvc")
    @patch("ray_curator.utils.nvcodec_utils.torch")
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


class TestPyNvcFrameExtractor:
    """Test suite for PyNvcFrameExtractor class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.width = 224
        self.height = 224
        self.batch_size = 4

    @patch("ray_curator.utils.nvcodec_utils.cuda")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    @patch("ray_curator.utils.nvcodec_utils.VideoBatchDecoder")
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

    @patch("ray_curator.utils.nvcodec_utils.cuda")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    @patch("ray_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_call_full_extraction(self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any) -> None:
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

    @patch("ray_curator.utils.nvcodec_utils.cuda")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    @patch("ray_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_call_fps_extraction(self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any) -> None:
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

    @patch("ray_curator.utils.nvcodec_utils.cuda")
    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    @patch("ray_curator.utils.nvcodec_utils.VideoBatchDecoder")
    def test_call_fps_extraction_no_fps(self, mock_decoder_class: Any, mock_torch: Any, mock_cvcuda: Any, mock_cuda: Any) -> None:
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

    @patch("ray_curator.utils.nvcodec_utils.cvcuda")
    @patch("ray_curator.utils.nvcodec_utils.torch")
    @patch("ray_curator.utils.nvcodec_utils.VideoBatchDecoder")
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


class TestPixelFormatMapping:
    """Test suite for pixel format mapping."""

    def test_pixel_format_mapping_exists(self) -> None:
        """Test that pixel format mapping dictionary exists and has expected keys."""
        # This test verifies the mapping exists and has the expected structure
        assert isinstance(pixel_format_to_cvcuda_code, dict)
        assert len(pixel_format_to_cvcuda_code) > 0
