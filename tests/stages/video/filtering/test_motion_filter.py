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
import uuid
from unittest.mock import Mock, patch

import torch

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.filtering.motion_filter import MotionFilterStage, MotionVectorDecodeStage
from nemo_curator.stages.video.filtering.motion_vector_backend import (
    DecodedData,
    MotionInfo,
    VideoResolutionTooSmallError,
)
from nemo_curator.tasks.video import Clip, ClipStats, Video, VideoMetadata, VideoTask


class TestMotionVectorDecodeStage:
    """Test cases for MotionVectorDecodeStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = MotionVectorDecodeStage(
            num_cpus_per_worker=4.0, verbose=False, target_fps=2.0, target_duration_ratio=0.5
        )

        # Create a mock video with clips
        self.mock_video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=480,
                width=640,
                framerate=30.0,
                num_frames=900,
                duration=30.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(0.0, 5.0),
                    buffer=b"mock_clip_data_1",
                    errors={},
                ),
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(5.0, 10.0),
                    buffer=b"mock_clip_data_2",
                    errors={},
                ),
            ],
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self):
        """Test the name property."""
        assert self.stage.name == "motion_vector_decoding"

    def test_resources_property(self):
        """Test the resources property."""
        resources = self.stage.resources
        assert isinstance(resources, Resources)
        assert resources.cpus == 4.0

    def test_inputs_property(self):
        """Test the inputs property."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], [])

    def test_outputs_property(self):
        """Test the outputs property."""
        outputs = self.stage.outputs()
        assert outputs == (["data"], ["decoded_motion_data"])

    def test_process_successful_decode(self):
        """Test successful processing of clips."""
        mock_decoded_data = Mock(spec=DecodedData)
        mock_decoded_data.frames = [Mock()]

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.return_value = mock_decoded_data

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            assert len(result.data.clips) == 2
            assert all(clip.decoded_motion_data is not None for clip in result.data.clips)
            assert all(clip.decoded_motion_data == mock_decoded_data for clip in result.data.clips)

    def test_process_with_empty_buffer(self):
        """Test processing clips with empty buffer."""
        # Create a clip with empty buffer
        empty_clip = Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=None, errors={})
        self.mock_video.clips = [empty_clip]

        result = self.stage.process(self.mock_task)

        assert result == self.mock_task
        assert empty_clip.decoded_motion_data is None
        assert empty_clip.errors["buffer"] == "empty"

    def test_process_with_resolution_too_small_error(self):
        """Test processing clips with resolution too small error."""
        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.side_effect = VideoResolutionTooSmallError("Resolution too small")

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            assert all(clip.decoded_motion_data is None for clip in result.data.clips)
            assert all(clip.errors["motion_decode"] == "resolution_too_small" for clip in result.data.clips)

    def test_process_with_decode_exception(self):
        """Test processing clips with decode exception."""
        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.side_effect = Exception("Decode failed")

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            assert all(clip.decoded_motion_data is None for clip in result.data.clips)
            assert all(clip.errors["motion_decode"] == "decode_failed" for clip in result.data.clips)

    def test_process_with_no_motion_frames(self):
        """Test processing clips with no motion frames."""
        mock_decoded_data = Mock(spec=DecodedData)
        mock_decoded_data.frames = []  # Empty frames list

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.return_value = mock_decoded_data

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            assert all(clip.decoded_motion_data is None for clip in result.data.clips)
            assert all(clip.errors["motion_decode"] == "no_motion_frames" for clip in result.data.clips)

    def test_process_with_none_decoded_data(self):
        """Test processing clips with None decoded data."""
        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.return_value = None

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            assert all(clip.decoded_motion_data is None for clip in result.data.clips)
            assert all(clip.errors["motion_decode"] == "no_motion_frames" for clip in result.data.clips)

    def test_process_with_verbose_mode(self):
        """Test processing with verbose mode enabled."""
        verbose_stage = MotionVectorDecodeStage(verbose=True)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.side_effect = VideoResolutionTooSmallError("Resolution too small")

            # This should not raise an exception
            result = verbose_stage.process(self.mock_task)

            assert result == self.mock_task
            assert all(clip.decoded_motion_data is None for clip in result.data.clips)

    def test_process_with_custom_parameters(self):
        """Test processing with custom parameters."""
        custom_stage = MotionVectorDecodeStage(num_cpus_per_worker=8.0, target_fps=5.0, target_duration_ratio=0.8)

        mock_decoded_data = Mock(spec=DecodedData)
        mock_decoded_data.frames = [Mock()]

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.decode_for_motion") as mock_decode:
            mock_decode.return_value = mock_decoded_data

            custom_stage.process(self.mock_task)

            # Check that decode_for_motion was called with custom parameters
            mock_decode.assert_called()
            call_args = mock_decode.call_args
            assert call_args[1]["thread_count"] == 8
            assert call_args[1]["target_fps"] == 5.0
            assert call_args[1]["target_duration_ratio"] == 0.8


class TestMotionFilterStage:
    """Test cases for MotionFilterStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = MotionFilterStage(
            score_only=False,
            global_mean_threshold=0.00098,
            per_patch_min_256_threshold=0.000001,
            num_gpus_per_worker=0.25,
            motion_filter_batch_size=256,
            verbose=False,
        )

        # Create a mock video with clips containing decoded motion data
        self.mock_video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=480,
                width=640,
                framerate=30.0,
                num_frames=900,
                duration=30.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(0.0, 5.0),
                    buffer=b"mock_clip_data_1",
                    decoded_motion_data=DecodedData(frames=[Mock()], frame_size=torch.Size([480, 640, 3])),
                    errors={},
                ),
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(5.0, 10.0),
                    buffer=b"mock_clip_data_2",
                    decoded_motion_data=DecodedData(frames=[Mock()], frame_size=torch.Size([480, 640, 3])),
                    errors={},
                ),
            ],
            filtered_clips=[],
            clip_stats=ClipStats(),
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self):
        """Test the name property."""
        assert self.stage.name == "motion_filter"

    def test_resources_property(self):
        """Test the resources property."""
        resources = self.stage.resources
        assert isinstance(resources, Resources)
        assert resources.gpus == 0.25

    def test_inputs_property(self):
        """Test the inputs property."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], [])

    def test_outputs_property(self):
        """Test the outputs property."""
        outputs = self.stage.outputs()
        assert outputs == (
            ["data"],
            [
                "decoded_motion_data",
                "motion_score_global_mean",
                "motion_score_per_patch_min_256",
                "filtered_clips",
                "clip_stats",
            ],
        )

    def test_process_with_small_motion(self):
        """Test processing clips with small motion."""
        mock_motion_info = MotionInfo(is_small_motion=True, per_patch_min_256=0.0000005, global_mean=0.0005)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            # All clips should be filtered out (small motion)
            assert len(result.data.clips) == 0
            assert len(result.data.filtered_clips) == 2
            assert result.data.clip_stats.num_filtered_by_motion == 2

    def test_process_with_large_motion(self):
        """Test processing clips with large motion."""
        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            # All clips should pass (large motion)
            assert len(result.data.clips) == 2
            assert len(result.data.filtered_clips) == 0
            assert result.data.clip_stats.num_filtered_by_motion == 0

    def test_process_with_no_decoded_motion_data(self):
        """Test processing clips with no decoded motion data."""
        # Remove decoded motion data
        for clip in self.mock_video.clips:
            clip.decoded_motion_data = None

        result = self.stage.process(self.mock_task)

        assert result == self.mock_task
        # All clips should be filtered out (fake small motion score)
        assert len(result.data.clips) == 0
        assert len(result.data.filtered_clips) == 2
        assert result.data.clip_stats.num_filtered_by_motion == 2

        # Check that fake scores were set
        for clip in result.data.filtered_clips:
            assert clip.motion_score_global_mean == -1.0
            assert clip.motion_score_per_patch_min_256 == -1.0

    def test_process_with_score_only_mode(self):
        """Test processing with score_only mode."""
        score_only_stage = MotionFilterStage(score_only=True)

        mock_motion_info = MotionInfo(is_small_motion=True, per_patch_min_256=0.0000005, global_mean=0.0005)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            result = score_only_stage.process(self.mock_task)

            assert result == self.mock_task
            # In score_only mode, clips with small motion should still be in clips list
            assert len(result.data.clips) == 2
            assert len(result.data.filtered_clips) == 0
            assert result.data.clip_stats.num_filtered_by_motion == 0

    def test_process_with_mixed_motion_clips(self):
        """Test processing with mixed motion clips."""
        small_motion_info = MotionInfo(is_small_motion=True, per_patch_min_256=0.0000005, global_mean=0.0005)

        large_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            # First clip has small motion, second has large motion
            mock_check.side_effect = [small_motion_info, large_motion_info]

            result = self.stage.process(self.mock_task)

            assert result == self.mock_task
            # One clip should pass, one should be filtered
            assert len(result.data.clips) == 1
            assert len(result.data.filtered_clips) == 1
            assert result.data.clip_stats.num_filtered_by_motion == 1

    def test_process_with_custom_thresholds(self):
        """Test processing with custom thresholds."""
        custom_stage = MotionFilterStage(global_mean_threshold=0.001, per_patch_min_256_threshold=0.0001)

        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.0005, global_mean=0.0008)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            custom_stage.process(self.mock_task)

            # Check that check_if_small_motion was called with custom thresholds
            mock_check.assert_called()
            call_args = mock_check.call_args
            assert call_args[1]["global_mean_threshold"] == 0.001
            assert call_args[1]["per_patch_min_256_threshold"] == 0.0001

    def test_process_with_gpu_enabled(self):
        """Test processing with GPU enabled."""
        gpu_stage = MotionFilterStage(num_gpus_per_worker=0.1)

        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            gpu_stage.process(self.mock_task)

            # Check that check_if_small_motion was called with GPU enabled
            mock_check.assert_called()
            call_args = mock_check.call_args
            assert call_args[1]["use_gpu"] is True

    def test_process_with_gpu_disabled(self):
        """Test processing with GPU disabled."""
        cpu_stage = MotionFilterStage(num_gpus_per_worker=0)

        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            cpu_stage.process(self.mock_task)

            # Check that check_if_small_motion was called with GPU disabled
            mock_check.assert_called()
            call_args = mock_check.call_args
            assert call_args[1]["use_gpu"] is False

    def test_process_with_custom_batch_size(self):
        """Test processing with custom batch size."""
        custom_stage = MotionFilterStage(motion_filter_batch_size=128)

        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            custom_stage.process(self.mock_task)

            # Check that check_if_small_motion was called with custom batch size
            mock_check.assert_called()
            call_args = mock_check.call_args
            assert call_args[1]["batch_size"] == 128

    def test_process_with_verbose_mode(self):
        """Test processing with verbose mode enabled."""
        verbose_stage = MotionFilterStage(verbose=True)

        mock_motion_info = MotionInfo(is_small_motion=True, per_patch_min_256=0.0000005, global_mean=0.0005)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            # This should not raise an exception
            result = verbose_stage.process(self.mock_task)

            assert result == self.mock_task

    def test_process_cleans_up_decoded_motion_data(self):
        """Test that decoded motion data is cleaned up after processing."""
        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            result = self.stage.process(self.mock_task)

            # Check that decoded_motion_data is cleaned up
            for clip in result.data.clips:
                assert clip.decoded_motion_data is None

    def test_process_sets_motion_scores(self):
        """Test that motion scores are properly set on clips."""
        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            result = self.stage.process(self.mock_task)

            # Check that motion scores are set
            for clip in result.data.clips:
                assert clip.motion_score_global_mean == 0.002
                assert clip.motion_score_per_patch_min_256 == 0.001

    def test_process_with_empty_clips_list(self):
        """Test processing with empty clips list."""
        self.mock_video.clips = []

        result = self.stage.process(self.mock_task)

        assert result == self.mock_task
        assert len(result.data.clips) == 0
        assert len(result.data.filtered_clips) == 0
        assert result.data.clip_stats.num_filtered_by_motion == 0

    def test_process_integration_with_video_task(self):
        """Test integration with VideoTask structure."""
        mock_motion_info = MotionInfo(is_small_motion=False, per_patch_min_256=0.001, global_mean=0.002)

        with patch("nemo_curator.stages.video.filtering.motion_vector_backend.check_if_small_motion") as mock_check:
            mock_check.return_value = mock_motion_info

            result = self.stage.process(self.mock_task)

            # Check that the task structure is maintained
            assert isinstance(result, VideoTask)
            assert result.task_id == "test_task"
            assert result.dataset_name == "test_dataset"
            assert isinstance(result.data, Video)
            assert result.data.input_video == pathlib.Path("test_video.mp4")
