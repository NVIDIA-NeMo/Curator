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

import copy
import pathlib
import subprocess
import tempfile
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask


# Mock GPU info class to simulate GPU information
class MockGpuInfo:
    def __init__(self, index: int, name: str):
        self.index = index
        self.name = name


# Mock GPU resources class to simulate GPU resources
class MockGpuResources:
    def __init__(self, num_nvencs: int = 3, num_nvdecs: int = 3):
        self.num_nvencs = num_nvencs
        self.num_nvdecs = num_nvdecs


class TestClipTranscodingStage:
    """Test cases for ClipTranscodingStage."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up class-level fixtures."""
        cls.temp_dir = pathlib.Path(tempfile.gettempdir()) / "test_clip_transcoding"

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.stage = ClipTranscodingStage(
            num_cpus_per_worker=4.0,
            encoder="libx264",
            encoder_threads=2,
            encode_batch_size=8,
            use_hwaccel=False,
            use_input_bit_rate=False,
            num_clips_per_chunk=16,
            ffmpeg_verbose=False,
            verbose=False,
        )

        # Create mock clips
        self.mock_clips = [
            Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0)),
            Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(5.0, 10.0)),
        ]

        # Create a mock video with clips
        self.mock_video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=300,
                duration=10.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=copy.deepcopy(self.mock_clips),
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self) -> None:
        """Test that the name property returns the correct value."""
        assert self.stage.name == "clip_transcoding"

    def test_inputs_outputs(self) -> None:
        """Test that inputs and outputs return the correct values."""
        inputs, input_additional = self.stage.inputs()
        outputs, _ = self.stage.outputs()

        assert inputs == ["data"]
        assert input_additional == ["source_bytes"]
        assert outputs == ["data"]

    def test_setup_invalid_encoder(self) -> None:
        """Test setup with invalid encoder raises ValueError."""
        stage = ClipTranscodingStage(encoder="invalid_encoder")

        with pytest.raises(ValueError, match="Expected encoder of"):
            stage.setup()

    def test_ray_stage_spec(self) -> None:
        """Test that ray_stage_spec returns the correct values."""
        spec = self.stage.ray_stage_spec()

        # Verify the expected keys and values based on the git diff
        from nemo_curator.backends.experimental.utils import RayStageSpecKeys

        assert RayStageSpecKeys.IS_FANOUT_STAGE in spec
        assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True

    def test_resources_cpu_encoder(self) -> None:
        """Test resource requirements for CPU encoders."""
        stage = ClipTranscodingStage(encoder="libx264", use_hwaccel=False, num_cpus_per_worker=6.0)

        resources = stage.resources
        assert isinstance(resources, Resources)
        assert resources.cpus == 6.0
        assert not resources.entire_gpu

    def test_process_no_clips(self) -> None:
        """Test processing when video has no clips."""
        self.mock_video.clips = []

        with patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger:
            result = self.stage.process(self.mock_task)

            # Should return early and log warning
            mock_logger.warning.assert_called_once()
            assert "No clips to transcode" in mock_logger.warning.call_args[0][0]
            assert result.data.source_bytes is None

    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir")
    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size")
    def test_process_successful_transcoding(self, mock_split: MagicMock, mock_temp_dir: MagicMock) -> None:
        """Test successful transcoding process."""
        # Setup mocks
        mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
        mock_split.return_value = [self.mock_video.clips]  # Return one chunk

        with (
            patch.object(self.stage, "_extract_clips") as mock_extract,
            patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger"),
            patch("pathlib.Path.write_bytes") as mock_write,
        ):
            result = self.stage.process(self.mock_task)

            # Should extract clips and create output tasks
            mock_extract.assert_called_once()
            mock_write.assert_called_once()
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], VideoTask)
            assert result[0].data.source_bytes is None  # Should be cleared

    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir")
    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size")
    def test_process_multiple_chunks(self, mock_split: MagicMock, mock_temp_dir: MagicMock) -> None:
        """Test processing with multiple clip chunks."""
        # Setup mocks to return multiple chunks
        chunk1 = [self.mock_clips[0]]
        chunk2 = [self.mock_clips[1]]
        mock_split.return_value = [chunk1, chunk2]
        mock_temp_dir.return_value.__enter__.return_value = self.temp_dir

        with (
            patch.object(self.stage, "_extract_clips"),
            patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger"),
            patch("pathlib.Path.write_bytes") as mock_write,
        ):
            result = self.stage.process(self.mock_task)

            # Should create multiple output tasks
            mock_write.assert_called_once()
            assert isinstance(result, list)
            assert len(result) == 2

            # Verify task properties
            for i, task in enumerate(result):
                assert isinstance(task, VideoTask)
                assert task.task_id == f"test_task_chunk_{i}"
                assert task.data.num_total_clips == len(self.mock_clips)
                assert task.data.num_clip_chunks == 2
                assert task.data.clip_chunk_index == i

    def test_process_with_input_bit_rate(self) -> None:
        """Test processing with input bit rate enabled."""
        stage = ClipTranscodingStage(use_input_bit_rate=True)

        with (
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir"
            ) as mock_temp_dir,
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size"
            ) as mock_split,
            patch.object(stage, "_extract_clips") as mock_extract,
            patch("pathlib.Path.write_bytes"),
        ):
            mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
            mock_split.return_value = [self.mock_video.clips]

            stage.process(self.mock_task)

            # Verify that use_bit_rate is passed correctly
            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert call_args[1]["use_bit_rate"] == "5000K"

    def test_extract_clips(self) -> None:
        """Test the _extract_clips method."""
        working_dir = self.temp_dir / "extract_test"
        video_filename = "input.mp4"

        with (
            patch.object(self.stage, "_build_ffmpeg_command") as mock_build,
            patch.object(self.stage, "_run_ffmpeg_command") as mock_run,
            patch.object(self.stage, "_read_clips_to_memory") as mock_read,
        ):
            mock_build.return_value = ["ffmpeg", "command"]

            self.stage._extract_clips(
                working_dir, video_filename, force_pix_fmt=False, use_bit_rate=None, clips=self.mock_clips
            )

            # Verify all steps are called
            mock_build.assert_called_once_with(video_filename, self.mock_clips, False, None)
            mock_run.assert_called_once_with(["ffmpeg", "command"], working_dir, self.mock_clips)
            mock_read.assert_called_once_with(working_dir, self.mock_clips)

    def test_build_ffmpeg_command_basic(self) -> None:
        """Test building basic FFmpeg command."""
        video_filename = "input.mp4"
        clips = [self.mock_clips[0]]

        command = self.stage._build_ffmpeg_command(video_filename, clips, False, None)

        # Verify basic command structure
        assert command[0] == "ffmpeg"
        assert "-hide_banner" in command
        assert "-loglevel" in command
        assert "error" in command  # Since ffmpeg_verbose is False

    def test_build_ffmpeg_command_verbose(self) -> None:
        """Test building FFmpeg command with verbose logging."""
        stage = ClipTranscodingStage(ffmpeg_verbose=True)
        command = stage._build_ffmpeg_command("input.mp4", [self.mock_clips[0]], False, None)

        # Should use warning level instead of error
        loglevel_idx = command.index("-loglevel")
        assert command[loglevel_idx + 1] == "warning"

    def test_add_decoder_threads_cpu(self) -> None:
        """Test adding decoder threads for CPU encoding."""
        command = []
        stage = ClipTranscodingStage(encoder_threads=4)

        stage._add_decoder_threads(command)

        assert "-threads" in command
        assert "4" in command

    def test_add_hwaccel_options_disabled(self) -> None:
        """Test hardware acceleration options when disabled."""
        command = []
        stage = ClipTranscodingStage(use_hwaccel=False)

        stage._add_hwaccel_options(command)

        # Should not add any hwaccel options
        assert "-hwaccel" not in command

    def test_add_input_options(self) -> None:
        """Test adding input options to FFmpeg command."""
        command = []
        clip = self.mock_clips[0]

        self.stage._add_input_options(command, clip, "input.mp4", 0)

        # Verify input options
        assert "-ss" in command
        assert "0.0" in command  # Start time
        assert "-to" in command
        assert "5.0" in command  # End time
        assert "-i" in command
        assert "input.mp4" in command
        assert "-map" in command
        assert "0:v:0" in command
        assert "-c:v" in command
        assert "libx264" in command

    def test_add_video_encoding_options_no_bitrate(self) -> None:
        """Test adding video encoding options without bit rate."""
        command = []

        self.stage._add_video_encoding_options(command, None, False)

        # Should not add bit rate options
        assert "-b:v" not in command

    def test_add_video_encoding_options_with_bitrate(self) -> None:
        """Test adding video encoding options with bit rate."""
        command = []

        self.stage._add_video_encoding_options(command, "5000K", False)

        # Should add bit rate options
        assert "-b:v" in command
        assert "5000K" in command

    def test_add_output_options(self) -> None:
        """Test adding output options to FFmpeg command."""
        command = []
        clip = self.mock_clips[0]

        self.stage._add_output_options(command, clip, 0)

        # Verify output options
        assert "-threads" in command
        assert "-map" in command
        assert "0:a:0?" in command
        assert "-c:a" in command
        assert "copy" in command
        assert f"{clip.uuid}.mp4" in command

    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.subprocess.check_output")
    def test_run_ffmpeg_command_success(self, mock_subprocess: MagicMock) -> None:
        """Test successful FFmpeg command execution."""
        command = ["ffmpeg", "-version"]
        working_dir = self.temp_dir / "ffmpeg_test"
        clips = self.mock_clips

        mock_subprocess.return_value = b"ffmpeg output"

        # Should not raise any exception
        self.stage._run_ffmpeg_command(command, working_dir, clips)

        mock_subprocess.assert_called_once_with(command, cwd=working_dir, stderr=subprocess.STDOUT)

    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.subprocess.check_output")
    def test_run_ffmpeg_command_verbose(self, mock_subprocess: MagicMock) -> None:
        """Test FFmpeg command execution with verbose logging."""
        stage = ClipTranscodingStage(verbose=True, ffmpeg_verbose=True)
        command = ["ffmpeg", "-version"]
        working_dir = self.temp_dir / "ffmpeg_verbose_test"
        clips = self.mock_clips

        mock_subprocess.return_value = b"ffmpeg output"

        with patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger:
            stage._run_ffmpeg_command(command, working_dir, clips)

            # Should log the command
            mock_logger.info.assert_called()
            mock_logger.warning.assert_called()  # For ffmpeg output

    @patch("nemo_curator.stages.video.clipping.clip_extraction_stages.subprocess.check_output")
    def test_run_ffmpeg_command_error(self, mock_subprocess: MagicMock) -> None:
        """Test FFmpeg command execution with error."""
        command = ["ffmpeg", "-invalid"]
        working_dir = self.temp_dir / "ffmpeg_error_test"
        clips = self.mock_clips

        error = subprocess.CalledProcessError(1, command, b"error output")
        mock_subprocess.side_effect = error

        with patch.object(self.stage, "_handle_ffmpeg_error") as mock_handle:
            self.stage._run_ffmpeg_command(command, working_dir, clips)

            mock_handle.assert_called_once_with(error, command, clips)

    def test_handle_ffmpeg_error(self) -> None:
        """Test FFmpeg error handling."""
        command = ["ffmpeg", "-invalid"]
        clips = self.mock_clips
        error = subprocess.CalledProcessError(1, command, b"error output")

        with patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger:
            self.stage._handle_ffmpeg_error(error, command, clips)

            # Should log errors
            assert mock_logger.error.call_count >= 2
            mock_logger.warning.assert_called()

            # Should add errors to clips
            for clip in clips:
                assert "transcode" in clip.errors
                assert clip.errors["transcode"] == "error output"

    def test_handle_ffmpeg_error_no_output(self) -> None:
        """Test FFmpeg error handling when there's no error output."""
        command = ["ffmpeg", "-invalid"]
        clips = self.mock_clips
        error = subprocess.CalledProcessError(1, command, None)

        with patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger"):
            self.stage._handle_ffmpeg_error(error, command, clips)

            # Should add string representation of error
            for clip in clips:
                assert "transcode" in clip.errors
                assert str(error) in clip.errors["transcode"]

    def test_read_clips_to_memory(self) -> None:
        """Test reading clips back into memory."""
        working_dir = self.temp_dir / "read_test"
        clips = self.mock_clips

        # Mock file reading
        mock_file_data = b"mock_clip_data"

        with patch("pathlib.Path.read_bytes") as mock_read:
            mock_read.return_value = mock_file_data

            self.stage._read_clips_to_memory(working_dir, clips)

            # Should read each clip file
            assert mock_read.call_count == len(clips)

            # Should set buffer for each clip
            for clip in clips:
                assert clip.buffer == mock_file_data

    def test_process_10_bit_color(self) -> None:
        """Test processing with 10-bit color video."""
        # Mock 10-bit color detection
        with (
            patch.object(self.mock_video, "is_10_bit_color", return_value=True),
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir"
            ) as mock_temp_dir,
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size"
            ) as mock_split,
            patch.object(self.stage, "_extract_clips") as mock_extract,
            patch("pathlib.Path.write_bytes"),
        ):
            mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
            mock_split.return_value = [self.mock_video.clips]

            self.stage.process(self.mock_task)

            # Should pass force_pix_fmt=True
            mock_extract.assert_called_once()
            assert mock_extract.call_args[1]["force_pix_fmt"] is True

    def test_process_logging_clip_stats(self) -> None:
        """Test that clip statistics are logged during processing."""
        with (
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir"
            ) as mock_temp_dir,
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size"
            ) as mock_split,
            patch.object(self.stage, "_extract_clips"),
            patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger,
            patch("pathlib.Path.write_bytes"),
        ):
            mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
            mock_split.return_value = [self.mock_video.clips]

            self.stage.process(self.mock_task)

            # Should log clip statistics
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            stats_calls = [call for call in info_calls if "clips and weight" in call]
            assert len(stats_calls) > 0

    def test_process_verbose_logging(self) -> None:
        """Test verbose logging during processing."""
        stage = ClipTranscodingStage(verbose=True)

        with (
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir"
            ) as mock_temp_dir,
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size"
            ) as mock_split,
            patch.object(stage, "_extract_clips"),
            patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger,
            patch("pathlib.Path.write_bytes"),
        ):
            mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
            mock_split.return_value = [self.mock_video.clips[:1], self.mock_video.clips[1:]]  # Two chunks

            stage.process(self.mock_task)

            # Should log subtask spawning
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            spawn_calls = [call for call in info_calls if "Spawning subtask" in call]
            assert len(spawn_calls) == 2  # One for each chunk

    def test_batch_processing(self) -> None:
        """Test processing clips in batches."""
        stage = ClipTranscodingStage(encode_batch_size=1)  # Process one clip at a time

        with (
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.make_pipeline_temporary_dir"
            ) as mock_temp_dir,
            patch(
                "nemo_curator.stages.video.clipping.clip_extraction_stages.grouping.split_by_chunk_size"
            ) as mock_split,
            patch.object(stage, "_extract_clips") as mock_extract,
            patch("pathlib.Path.write_bytes"),
        ):
            mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
            mock_split.return_value = [self.mock_video.clips]

            stage.process(self.mock_task)

            # Should call _extract_clips twice (once per clip)
            assert mock_extract.call_count == 2

    def test_worker_metadata_setup(self) -> None:
        """Test setup with worker metadata."""
        worker_metadata = Mock(spec=WorkerMetadata)

        # Should not raise any exception
        self.stage.setup(worker_metadata)

    def test_edge_case_empty_clips(self) -> None:
        """Test edge case where there are no clips to process."""
        stage = ClipTranscodingStage(encode_batch_size=1)

        # Create a video with no clips
        empty_video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=self.mock_video.metadata,
            clips=[],
        )

        empty_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=empty_video)

        with patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger:
            result = stage.process(empty_task)

            # Should handle empty clips gracefully
            assert result.data.source_bytes is None
            mock_logger.warning.assert_called_once()
            assert "No clips to transcode" in mock_logger.warning.call_args[0][0]
