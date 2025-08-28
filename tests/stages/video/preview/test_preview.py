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
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.preview.preview import PreviewStage
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask, _Window

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestPreviewStage:
    """Test suite for PreviewStage class."""

    def test_initialization_defaults(self):
        """Test PreviewStage initialization with default values."""
        stage = PreviewStage()

        assert stage.target_fps == 1.0
        assert stage.target_height == 240
        assert stage.verbose is False
        assert stage.num_cpus_per_worker == 4.0
        assert stage.compression_level == 6
        assert stage.quality == 50
        assert stage._resources == Resources(cpus=4.0)

    def test_initialization_custom_values(self):
        """Test PreviewStage initialization with custom values."""
        stage = PreviewStage(
            target_fps=2.5, target_height=480, verbose=True, num_cpus_per_worker=8.0, compression_level=3, quality=75
        )

        assert stage.target_fps == 2.5
        assert stage.target_height == 480
        assert stage.verbose is True
        assert stage.num_cpus_per_worker == 8.0
        assert stage.compression_level == 3
        assert stage.quality == 75
        assert stage._resources == Resources(cpus=8.0)

    def test_inputs(self):
        """Test inputs method returns correct tuple."""
        stage = PreviewStage()
        inputs = stage.inputs()

        assert inputs == (["data"], ["clips"])

    def test_outputs(self):
        """Test outputs method returns correct tuple."""
        stage = PreviewStage()
        outputs = stage.outputs()

        assert outputs == (["data"], ["clips"])

    def test_process_with_adequate_metadata(self):
        """Test process method with adequate video metadata."""
        # Create mock video with adequate metadata
        video_metadata = VideoMetadata(framerate=30.0, height=720)

        # Create mock window with mp4 bytes
        window = _Window(start_frame=0, end_frame=30, mp4_bytes=b"fake_mp4_data")

        # Create mock clip with window
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0), windows=[window])

        # Create video with clips
        video = Video(input_video=pathlib.Path("test.mp4"), metadata=video_metadata, clips=[clip])

        # Create video task
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        stage = PreviewStage()

        # Mock the _generate_preview method
        with patch.object(stage, "_generate_preview") as mock_generate:
            result = stage.process(task)

            # Verify the task is returned unchanged
            assert result is task

            # Verify _generate_preview was called for the window
            mock_generate.assert_called_once_with(window)

    def test_process_with_low_framerate_warning(self):
        """Test process method logs warning for low framerate."""
        # Create mock video with low framerate
        video_metadata = VideoMetadata(
            framerate=0.5,  # Lower than target_fps (1.0)
            height=720,
        )

        video = Video(input_video=pathlib.Path("test.mp4"), metadata=video_metadata, clips=[])

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        stage = PreviewStage()

        # Mock logger to capture warning
        with patch("nemo_curator.stages.video.preview.preview.logger") as mock_logger:
            stage.process(task)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "framerate 0.5 < 1.0" in warning_msg
            assert "preview generation quality will be degraded" in warning_msg

    def test_process_with_low_height_warning(self):
        """Test process method logs warning for low height."""
        # Create mock video with low height
        video_metadata = VideoMetadata(
            framerate=30.0,
            height=120,  # Lower than target_height (240)
        )

        video = Video(input_video=pathlib.Path("test.mp4"), metadata=video_metadata, clips=[])

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        stage = PreviewStage()

        # Mock logger to capture warning
        with patch("nemo_curator.stages.video.preview.preview.logger") as mock_logger:
            stage.process(task)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "height 120 < 240" in warning_msg
            assert "preview generation quality will be degraded" in warning_msg

    def test_process_multiple_clips_and_windows(self):
        """Test process method handles multiple clips and windows."""
        # Create multiple windows
        window1 = _Window(start_frame=0, end_frame=30, mp4_bytes=b"data1")
        window2 = _Window(start_frame=30, end_frame=60, mp4_bytes=b"data2")
        window3 = _Window(start_frame=60, end_frame=90, mp4_bytes=b"data3")

        # Create multiple clips
        clip1 = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 1.0), windows=[window1, window2])
        clip2 = Clip(uuid=uuid4(), source_video="test.mp4", span=(1.0, 2.0), windows=[window3])

        video = Video(
            input_video=pathlib.Path("test.mp4"),
            metadata=VideoMetadata(framerate=30.0, height=720),
            clips=[clip1, clip2],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        stage = PreviewStage()

        # Mock the _generate_preview method
        with patch.object(stage, "_generate_preview") as mock_generate:
            stage.process(task)

            # Verify _generate_preview was called for all windows
            assert mock_generate.call_count == 3
            mock_generate.assert_any_call(window1)
            mock_generate.assert_any_call(window2)
            mock_generate.assert_any_call(window3)

    @patch("nemo_curator.stages.video.preview.preview.make_pipeline_temporary_dir")
    @patch("subprocess.check_output")
    def test_generate_preview_success(self, mock_subprocess: "MagicMock", mock_temp_dir: "MagicMock"):
        """Test successful preview generation."""
        # Mock temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = pathlib.Path("temp/preview")

        # Mock subprocess success
        mock_subprocess.return_value = b""

        # Create window with mp4 bytes
        window = _Window(start_frame=0, end_frame=30, mp4_bytes=b"fake_mp4_data")

        stage = PreviewStage(target_fps=2.0, target_height=360, compression_level=4, quality=80)

        # Mock pathlib.Path for input and output files
        with patch("pathlib.Path") as mock_path:
            mock_input = Mock()
            mock_input.as_posix.return_value = "temp/preview/input.mp4"
            mock_input.write_bytes.return_value = None

            mock_output = Mock()
            mock_output.as_posix.return_value = "temp/preview/output.webp"
            mock_output.read_bytes.return_value = b"fake_webp_data"

            mock_path.side_effect = [mock_input, mock_output]

            stage._generate_preview(window)

            # Verify mp4 bytes were written
            mock_input.write_bytes.assert_called_once_with(b"fake_mp4_data")

            # Verify subprocess was called with correct command
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]

            assert call_args[0] == "ffmpeg"
            assert call_args[1] == "-threads"
            assert call_args[2] == "4"  # num_cpus_per_worker
            assert call_args[3] == "-y"
            assert call_args[4] == "-i"
            assert call_args[5] == "temp/preview/input.mp4"
            assert call_args[6] == "-loglevel"
            assert call_args[7] == "error"
            assert call_args[8] == "-vf"
            assert call_args[9] == "fps=2.0,scale=-1:360"
            assert call_args[10] == "-c:v"
            assert call_args[11] == "libwebp"
            assert call_args[12] == "-lossless"
            assert call_args[13] == "0"
            assert call_args[14] == "-compression_level"
            assert call_args[15] == "4"
            assert call_args[16] == "-q:v"
            assert call_args[17] == "80"
            assert call_args[18] == "-loop"
            assert call_args[19] == "0"
            assert call_args[20] == "temp/preview/output.webp"

            # Verify webp bytes were read and assigned
            mock_output.read_bytes.assert_called_once()
            assert window.webp_bytes == b"fake_webp_data"

    @patch("nemo_curator.stages.video.preview.preview.make_pipeline_temporary_dir")
    @patch("subprocess.check_output")
    def test_generate_preview_subprocess_error(self, mock_subprocess: "MagicMock", mock_temp_dir: "MagicMock"):
        """Test preview generation with subprocess error."""
        # Mock temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = pathlib.Path("temp/preview")

        # Mock subprocess error
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["ffmpeg"], output=b"ffmpeg error output"
        )

        # Create window with mp4 bytes
        window = _Window(start_frame=0, end_frame=30, mp4_bytes=b"fake_mp4_data")

        stage = PreviewStage()

        # Mock pathlib.Path
        with patch("pathlib.Path") as mock_path:
            mock_input = Mock()
            mock_input.as_posix.return_value = "temp/preview/input.mp4"
            mock_input.write_bytes.return_value = None

            mock_output = Mock()
            mock_output.as_posix.return_value = "temp/preview/output.webp"

            mock_path.side_effect = [mock_input, mock_output]

            # Mock logger to capture error
            with patch("nemo_curator.stages.video.preview.preview.logger") as mock_logger:
                stage._generate_preview(window)

                # Verify error was logged
                mock_logger.error.assert_called_once()
                error_msg = mock_logger.error.call_args[0][0]
                assert "ffmpeg command failed with return code 1" in error_msg

                # Verify warning was logged with command
                mock_logger.warning.assert_called()
                warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
                assert any("ffmpeg command:" in msg for msg in warning_calls)
                assert any("ffmpeg output:" in msg for msg in warning_calls)

                # Verify webp_bytes was not set
                assert window.webp_bytes is None

    @patch("nemo_curator.stages.video.preview.preview.make_pipeline_temporary_dir")
    @patch("subprocess.check_output")
    def test_generate_preview_with_ffmpeg_output(self, mock_subprocess: "MagicMock", mock_temp_dir: "MagicMock"):
        """Test preview generation when ffmpeg produces output."""
        # Mock temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = pathlib.Path("temp/preview")

        # Mock subprocess with output
        mock_subprocess.return_value = b"ffmpeg info message"

        # Create window with mp4 bytes
        window = _Window(start_frame=0, end_frame=30, mp4_bytes=b"fake_mp4_data")

        stage = PreviewStage()

        # Mock pathlib.Path
        with patch("pathlib.Path") as mock_path:
            mock_input = Mock()
            mock_input.as_posix.return_value = "temp/preview/input.mp4"
            mock_input.write_bytes.return_value = None

            mock_output = Mock()
            mock_output.as_posix.return_value = "temp/preview/output.webp"
            mock_output.read_bytes.return_value = b"fake_webp_data"

            mock_path.side_effect = [mock_input, mock_output]

            # Mock logger to capture warning
            with patch("nemo_curator.stages.video.preview.preview.logger") as mock_logger:
                stage._generate_preview(window)

                # Verify warning was logged about ffmpeg output
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "ffmpeg output: ffmpeg info message" in warning_msg

                # Verify webp_bytes was set
                assert window.webp_bytes == b"fake_webp_data"

    def test_generate_preview_no_mp4_bytes(self):
        """Test preview generation when window has no mp4 bytes."""
        # Create window without mp4 bytes
        window = _Window(start_frame=0, end_frame=30, mp4_bytes=None)

        stage = PreviewStage()

        # Mock the temporary directory context manager
        with patch("nemo_curator.stages.video.preview.preview.make_pipeline_temporary_dir") as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = pathlib.Path("temp/preview")

            # When mp4_bytes is None, write_bytes will fail with TypeError
            # since we're trying to write None to a file
            with pytest.raises(TypeError):
                stage._generate_preview(window)

    def test_resources_property(self):
        """Test that resources property returns the correct Resources object."""
        stage = PreviewStage(num_cpus_per_worker=6.0)

        assert stage.resources == Resources(cpus=6.0)
        assert stage.resources.cpus == 6.0

    def test_with_method(self):
        """Test the with_ method for creating modified instances."""
        stage = PreviewStage()

        # Test modifying parameters that are supported by the base class
        new_stage = stage.with_(name="CustomPreviewStage", resources=Resources(cpus=8.0), batch_size=5)

        # Verify new instance has modified values
        assert new_stage.name == "CustomPreviewStage"
        assert new_stage.resources == Resources(cpus=8.0)
        assert new_stage.batch_size == 5

        # Verify original instance is unchanged
        assert stage.name == "ProcessingStage"  # Default name from base class
        assert stage.resources == Resources(cpus=4.0)
        assert stage.batch_size == 1

        # Verify other parameters are inherited
        assert new_stage.target_fps == stage.target_fps
        assert new_stage.target_height == stage.target_height
        assert new_stage.compression_level == stage.compression_level
        assert new_stage.quality == stage.quality
        assert new_stage.verbose == stage.verbose
        assert new_stage.num_cpus_per_worker == stage.num_cpus_per_worker
