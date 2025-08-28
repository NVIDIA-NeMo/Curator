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

# ruff: noqa: ANN401
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.clipping.video_frame_extraction import (
    VideoFrameExtractionStage,
    get_frames_from_ffmpeg,
)
from nemo_curator.tasks.video import Video, VideoMetadata, VideoTask


class TestGetFramesFromFfmpeg:
    """Test suite for get_frames_from_ffmpeg function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.video_file = Path("test_video.mp4")
        self.width = 224
        self.height = 224

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.subprocess.Popen")
    def test_get_frames_from_ffmpeg_cpu_success(self, mock_popen: Any) -> None:
        """Test successful frame extraction using CPU."""
        # Create mock process
        mock_process = Mock()
        mock_process.returncode = 0
        mock_video_stream = b"fake_video_data" * (self.width * self.height * 3)
        mock_process.communicate.return_value = (mock_video_stream, b"")
        mock_popen.return_value = mock_process

        result = get_frames_from_ffmpeg(self.video_file, self.width, self.height, use_gpu=False)

        # Verify correct command was used
        expected_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "4",
            "-i",
            self.video_file.as_posix(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-",
        ]
        mock_popen.assert_called_once_with(expected_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.subprocess.Popen")
    def test_get_frames_from_ffmpeg_gpu_success(self, mock_popen: Any) -> None:
        """Test successful frame extraction using GPU."""
        # Create mock process
        mock_process = Mock()
        mock_process.returncode = 0
        mock_video_stream = b"fake_video_data" * (self.width * self.height * 3)
        mock_process.communicate.return_value = (mock_video_stream, b"")
        mock_popen.return_value = mock_process

        result = get_frames_from_ffmpeg(self.video_file, self.width, self.height, use_gpu=True)

        # Verify correct command was used
        expected_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "1",
            "-hwaccel",
            "auto",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            self.video_file.as_posix(),
            "-vf",
            f"scale_npp={self.width}:{self.height},hwdownload,format=nv12",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        mock_popen.assert_called_once_with(expected_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.subprocess.Popen")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_get_frames_from_ffmpeg_gpu_failure_fallback(self, mock_logger: Any, mock_popen: Any) -> None:
        """Test GPU failure with fallback to CPU."""
        # Mock GPU failure, then CPU success
        mock_process_gpu = Mock()
        mock_process_gpu.returncode = 1
        mock_process_gpu.communicate.return_value = (b"", b"GPU error")

        mock_process_cpu = Mock()
        mock_process_cpu.returncode = 0
        mock_video_stream = b"fake_video_data" * (self.width * self.height * 3)
        mock_process_cpu.communicate.return_value = (mock_video_stream, b"")

        mock_popen.side_effect = [mock_process_gpu, mock_process_cpu]

        result = get_frames_from_ffmpeg(self.video_file, self.width, self.height, use_gpu=True)

        # Verify both GPU and CPU commands were called
        assert mock_popen.call_count == 2

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "Caught ffmpeg runtime error with `use_gpu=True` option, falling back to CPU."
        )

        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.subprocess.Popen")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_get_frames_from_ffmpeg_cpu_failure(self, mock_logger: Any, mock_popen: Any) -> None:
        """Test CPU failure with no fallback."""
        # Create mock process with failure
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"CPU error")
        mock_popen.return_value = mock_process

        result = get_frames_from_ffmpeg(self.video_file, self.width, self.height, use_gpu=False)

        # Verify error was logged
        mock_logger.exception.assert_called_once_with("FFmpeg error: CPU error")

        # Verify result is None
        assert result is None

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.subprocess.Popen")
    def test_get_frames_from_ffmpeg_reshape(self, mock_popen: Any) -> None:
        """Test that frames are properly reshaped."""
        # Create mock process with specific data size
        mock_process = Mock()
        mock_process.returncode = 0
        # Create data for 2 frames of size width x height x 3 channels
        frame_size = self.width * self.height * 3
        mock_video_stream = b"x" * (frame_size * 2)
        mock_process.communicate.return_value = (mock_video_stream, b"")
        mock_popen.return_value = mock_process

        result = get_frames_from_ffmpeg(self.video_file, self.width, self.height, use_gpu=False)

        # Verify result shape
        assert result is not None
        assert result.shape == (2, self.height, self.width, 3)
        assert result.dtype == np.uint8


class TestVideoFrameExtractionStage:
    """Test suite for VideoFrameExtractionStage class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.output_hw = (27, 48)
        self.batch_size = 64
        self.stage = VideoFrameExtractionStage(
            output_hw=self.output_hw,
            pyncv_batch_size=self.batch_size,
            decoder_mode="pynvc",
            verbose=False,
        )

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        stage = VideoFrameExtractionStage()
        assert stage.output_hw == (27, 48)
        assert stage.pyncv_batch_size == 64
        assert stage.decoder_mode == "pynvc"
        assert not hasattr(stage, "pynvc_frame_extractor") or stage.pynvc_frame_extractor is None
        assert stage.verbose is False

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        custom_output_hw = (100, 200)
        custom_batch_size = 32
        stage = VideoFrameExtractionStage(
            output_hw=custom_output_hw,
            pyncv_batch_size=custom_batch_size,
            decoder_mode="ffmpeg",
            verbose=True,
        )
        assert stage.output_hw == custom_output_hw
        assert stage.pyncv_batch_size == custom_batch_size
        assert stage.decoder_mode == "ffmpeg"
        assert stage.verbose is True

    def test_name_property(self) -> None:
        """Test name property."""
        assert self.stage.name == "video_frame_extraction"

    def test_inputs_property(self) -> None:
        """Test inputs property."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], [])

    def test_outputs_property(self) -> None:
        """Test outputs property."""
        outputs = self.stage.outputs()
        assert outputs == (["data"], [])

    def test_resources_property_pynvc(self) -> None:
        """Test resources property with PyNvCodec mode."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        resources = stage.resources
        assert isinstance(resources, Resources)
        assert resources.gpu_memory_gb == 10

    def test_resources_property_ffmpeg(self) -> None:
        """Test resources property with FFmpeg mode."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg")
        resources = stage.resources
        assert isinstance(resources, Resources)
        assert resources.cpus == 4.0

    def test_resources_property_ffmpeg_gpu(self) -> None:
        """Test resources property with FFmpeg GPU mode."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg_gpu")
        resources = stage.resources
        assert isinstance(resources, Resources)
        assert resources.cpus == 4.0

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction._PYNVC_AVAILABLE", True)
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.PyNvcFrameExtractor")
    def test_setup_pynvc_mode(self, mock_pynvc_extractor: Any) -> None:
        """Test setup method with PyNvCodec mode."""
        stage = VideoFrameExtractionStage(
            output_hw=self.output_hw,
            pyncv_batch_size=self.batch_size,
            decoder_mode="pynvc",
        )

        mock_extractor_instance = Mock()
        mock_pynvc_extractor.return_value = mock_extractor_instance

        stage.setup()

        # Verify PyNvcFrameExtractor was created with correct parameters
        mock_pynvc_extractor.assert_called_once_with(
            width=self.output_hw[1],
            height=self.output_hw[0],
            batch_size=self.batch_size,
        )
        assert stage.pynvc_frame_extractor == mock_extractor_instance

    def test_setup_ffmpeg_mode(self) -> None:
        """Test setup method with FFmpeg mode."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg")
        stage.setup()
        # Should not create PyNvcFrameExtractor
        assert not hasattr(stage, "pynvc_frame_extractor") or stage.pynvc_frame_extractor is None

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction._PYNVC_AVAILABLE", False)
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_setup_pynvc_mode_unavailable(self, mock_logger: Any) -> None:
        """Test setup method with PyNvCodec mode when PyNvcFrameExtractor is not available."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        stage.setup()

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "PyNvcFrameExtractor not available, will fall back to FFmpeg for video processing"
        )

        # Should not create PyNvcFrameExtractor
        assert stage.pynvc_frame_extractor is None

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.get_frames_from_ffmpeg")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_no_pynvc_extractor(self, mock_logger: Any, mock_get_frames: Any, mock_temp_file: Any) -> None:
        """Test process method without PyNvCodec extractor initialization."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        # Don't call setup, so extractor remains None
        stage.pynvc_frame_extractor = None

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock FFmpeg to return None (simulating failure)
        mock_get_frames.return_value = None

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify fallback message was logged
        mock_logger.info.assert_called_with("PyNvcFrameExtractor not available, using FFmpeg CPU fallback")

        # Verify FFmpeg was called as fallback
        mock_get_frames.assert_called_once_with(mock_temp_path, width=27, height=48, use_gpu=False)

        # The implementation returns None when frame extraction fails
        assert result is None

    def test_process_no_source_bytes(self) -> None:
        """Test process method with no source bytes."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        stage.pynvc_frame_extractor = Mock()

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=None,
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        with pytest.raises(ValueError, match="Video source bytes are not available"):
            stage.process(task)

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_incomplete_metadata(self, mock_logger: Any) -> None:
        """Test process method with incomplete metadata."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        stage.pynvc_frame_extractor = Mock()

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(framerate=30, width=640, height=480),  # Missing required fields
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("Incomplete metadata for test_video.mp4. Skipping...")

        # Verify error was set
        assert "metadata" in result.data.errors
        assert result.data.errors["metadata"] == "incomplete"

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    def test_process_pynvc_success(self, mock_temp_file: Any) -> None:
        """Test successful processing with PyNvCodec."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        mock_extractor = Mock()
        stage.pynvc_frame_extractor = mock_extractor

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock extractor output
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value.astype.return_value = np.zeros((10, 27, 48, 3), dtype=np.uint8)
        mock_extractor.return_value = mock_tensor

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify frame array was set
        assert result.data.frame_array is not None
        assert result.data.frame_array.shape == (10, 27, 48, 3)
        assert result.data.frame_array.dtype == np.uint8

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.get_frames_from_ffmpeg")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_pynvc_exception_fallback(
        self, mock_logger: Any, mock_get_frames: Any, mock_temp_file: Any
    ) -> None:
        """Test PyNvCodec exception with FFmpeg fallback."""
        stage = VideoFrameExtractionStage(decoder_mode="pynvc")
        mock_extractor = Mock()
        stage.pynvc_frame_extractor = mock_extractor

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock extractor to raise exception
        mock_extractor.side_effect = Exception("PyNvCodec error")

        # Mock FFmpeg fallback
        mock_frames = np.zeros((10, 27, 48, 3), dtype=np.uint8)
        mock_get_frames.return_value = mock_frames

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "Got exception PyNvCodec error with PyNvVideoCodec decode, trying ffmpeg CPU fallback"
        )

        # Verify FFmpeg was called with CPU fallback
        mock_get_frames.assert_called_once_with(
            mock_temp_path,
            width=27,
            height=48,
            use_gpu=False,
        )

        # Verify frame array was set
        assert result.data.frame_array is not None
        assert np.array_equal(result.data.frame_array, mock_frames)

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.get_frames_from_ffmpeg")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_ffmpeg_mode(self, mock_logger: Any, mock_get_frames: Any, mock_temp_file: Any) -> None:
        """Test processing with FFmpeg mode."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg")
        # Set a mock extractor even though we're in FFmpeg mode due to current implementation
        stage.pynvc_frame_extractor = Mock()

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock FFmpeg output
        mock_frames = np.zeros((10, 27, 48, 3), dtype=np.uint8)
        mock_get_frames.return_value = mock_frames

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify info messages were logged
        mock_logger.info.assert_any_call("Decoding video test_video.mp4 with FFmpeg")
        mock_logger.info.assert_any_call("Decoded video test_video.mp4 with FFmpeg successfully")

        # Verify FFmpeg was called with correct parameters
        mock_get_frames.assert_called_once_with(
            mock_temp_path,
            width=27,
            height=48,
            use_gpu=False,
        )

        # Verify frame array was set
        assert result.data.frame_array is not None
        assert np.array_equal(result.data.frame_array, mock_frames)

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.get_frames_from_ffmpeg")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_ffmpeg_gpu_mode(self, mock_logger: Any, mock_get_frames: Any, mock_temp_file: Any) -> None:
        """Test processing with FFmpeg GPU mode."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg_gpu")
        # Set a mock extractor even though we're in FFmpeg mode due to current implementation
        stage.pynvc_frame_extractor = Mock()

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock FFmpeg output
        mock_frames = np.zeros((10, 27, 48, 3), dtype=np.uint8)
        mock_get_frames.return_value = mock_frames

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify info messages were logged
        mock_logger.info.assert_any_call("Decoding video test_video.mp4 with FFmpeg")
        mock_logger.info.assert_any_call("Decoded video test_video.mp4 with FFmpeg successfully")

        # Verify FFmpeg was called with GPU enabled
        mock_get_frames.assert_called_once_with(
            mock_temp_path,
            width=27,
            height=48,
            use_gpu=True,
        )

        # Verify frame array was set
        assert result.data.frame_array is not None
        assert np.array_equal(result.data.frame_array, mock_frames)

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.get_frames_from_ffmpeg")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_frame_extraction_failure(
        self, mock_logger: Any, mock_get_frames: Any, mock_temp_file: Any
    ) -> None:
        """Test processing when frame extraction fails."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg")
        # Set a mock extractor even though we're in FFmpeg mode due to current implementation
        stage.pynvc_frame_extractor = Mock()

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock FFmpeg to return None (failure)
        mock_get_frames.return_value = None

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify error was logged
        mock_logger.error.assert_called_once_with("Frame extraction failed, exiting...")

        # Verify None was returned
        assert result is None

    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.make_pipeline_named_temporary_file")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.get_frames_from_ffmpeg")
    @patch("nemo_curator.stages.video.clipping.video_frame_extraction.logger")
    def test_process_verbose_mode(self, mock_logger: Any, mock_get_frames: Any, mock_temp_file: Any) -> None:
        """Test processing with verbose mode enabled."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg", verbose=True)
        # Set a mock extractor even though we're in FFmpeg mode due to current implementation
        stage.pynvc_frame_extractor = Mock()

        # Mock temporary file
        mock_temp_path = Mock()
        mock_temp_path.open.return_value.__enter__ = Mock()
        mock_temp_path.open.return_value.__exit__ = Mock()
        mock_temp_file.return_value.__enter__ = Mock(return_value=mock_temp_path)
        mock_temp_file.return_value.__exit__ = Mock()

        # Mock FFmpeg output
        mock_frames = np.zeros((10, 27, 48, 3), dtype=np.uint8)
        mock_get_frames.return_value = mock_frames

        video = Video(
            input_video=Path("test_video.mp4"),
            source_bytes=b"fake_video_data",
            metadata=VideoMetadata(
                framerate=30, width=640, height=480, duration=10.0, video_codec="h264", num_frames=300
            ),
        )
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Verify verbose message was logged
        mock_logger.info.assert_any_call(f"Loaded video as numpy uint8 array with shape {mock_frames.shape}")

        # Verify frame array was set
        assert result.data.frame_array is not None
        assert np.array_equal(result.data.frame_array, mock_frames)

    def test_process_with_worker_metadata(self) -> None:
        """Test setup method with worker metadata."""
        stage = VideoFrameExtractionStage(decoder_mode="ffmpeg")
        worker_metadata = WorkerMetadata(worker_id="test_worker")

        # Should not raise any exception
        stage.setup(worker_metadata)

        # Should not create PyNvcFrameExtractor for FFmpeg mode
        assert not hasattr(stage, "pynvc_frame_extractor") or stage.pynvc_frame_extractor is None
