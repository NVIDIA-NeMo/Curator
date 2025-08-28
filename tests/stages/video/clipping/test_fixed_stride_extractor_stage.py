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
from unittest.mock import patch

import pytest

from nemo_curator.stages.video.clipping.clip_extraction_stages import FixedStrideExtractorStage
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask


class TestFixedStrideExtractorStage:
    """Test cases for FixedStrideExtractorStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = FixedStrideExtractorStage(
            clip_len_s=5.0, clip_stride_s=2.5, min_clip_length_s=1.0, limit_clips=10
        )

        # Create a mock video with complete metadata
        self.mock_video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=900,  # 30 seconds at 30fps
                duration=30.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[],
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self):
        """Test that the name property returns the correct value."""
        assert self.stage.name == "fixed_stride_extractor"

    def test_inputs_outputs(self):
        """Test that inputs and outputs return the correct values."""
        inputs, _ = self.stage.inputs()
        outputs, _ = self.stage.outputs()

        assert inputs == ["data"]
        assert outputs == ["data"]

    def test_process_successful_extraction(self):
        """Test successful clip extraction with valid video data."""
        result = self.stage.process(self.mock_task)

        assert isinstance(result, VideoTask)
        assert result.task_id == "test_task"
        assert result.dataset_name == "test_dataset"
        assert len(result.data.clips) > 0

        # Verify clip properties
        for clip in result.data.clips:
            assert isinstance(clip, Clip)
            assert isinstance(clip.uuid, uuid.UUID)
            assert clip.source_video == "test_video.mp4"
            assert len(clip.span) == 2
            assert clip.span[0] < clip.span[1]
            assert (clip.span[1] - clip.span[0]) >= self.stage.min_clip_length_s

    def test_process_no_source_bytes(self):
        """Test processing when source_bytes is None."""
        self.mock_video.source_bytes = None

        with pytest.raises(ValueError, match="Video source bytes are not available"):
            self.stage.process(self.mock_task)

    def test_process_incomplete_metadata(self):
        """Test processing when video has incomplete metadata."""
        self.mock_video.metadata.framerate = None

        result = self.stage.process(self.mock_task)

        assert isinstance(result, VideoTask)
        assert "metadata" in result.data.errors
        assert result.data.errors["metadata"] == "incomplete"

    def test_process_already_clipped(self):
        """Test processing when video already has clips and limit is reached."""
        # Add existing clips to reach the limit
        for i in range(self.stage.limit_clips):
            clip = Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(i * 5.0, (i + 1) * 5.0))
            self.mock_video.clips.append(clip)

        result = self.stage.process(self.mock_task)

        assert isinstance(result, VideoTask)
        # Should return the same task without adding new clips
        assert len(result.data.clips) == self.stage.limit_clips

    def test_process_missing_framerate(self):
        """Test processing when framerate is None."""
        self.mock_video.metadata.framerate = None

        result = self.stage.process(self.mock_task)

        # Should return early with error logged due to incomplete metadata
        assert isinstance(result, VideoTask)
        assert "metadata" in result.data.errors
        assert result.data.errors["metadata"] == "incomplete"

    def test_process_missing_num_frames(self):
        """Test processing when num_frames is None."""
        self.mock_video.metadata.num_frames = None

        result = self.stage.process(self.mock_task)

        # Should return early with error logged due to incomplete metadata
        assert isinstance(result, VideoTask)
        assert "metadata" in result.data.errors
        assert result.data.errors["metadata"] == "incomplete"

    def test_process_zero_framerate(self):
        """Test processing when framerate is zero."""
        self.mock_video.metadata.framerate = 0.0

        result = self.stage.process(self.mock_task)

        # Should return early with error logged due to incomplete metadata
        assert isinstance(result, VideoTask)
        assert "metadata" in result.data.errors
        assert result.data.errors["metadata"] == "incomplete"

    def test_clip_generation_logic(self):
        """Test that clips are generated with correct timing."""
        # Create a video with specific duration
        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=600,  # 20 seconds at 30fps
                duration=20.0,
                video_codec="h264",
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = self.stage.process(task)

        # Calculate expected clips
        # clip_len_s=5.0, clip_stride_s=2.5, duration=20.0
        # Expected clips: (0,5), (2.5,7.5), (5,10), (7.5,12.5), (10,15), (12.5,17.5), (15,20), (17.5,20)
        expected_clips = [
            (0.0, 5.0),
            (2.5, 7.5),
            (5.0, 10.0),
            (7.5, 12.5),
            (10.0, 15.0),
            (12.5, 17.5),
            (15.0, 20.0),
            (17.5, 20.0),
        ]

        assert len(result.data.clips) == len(expected_clips)

        for _i, (clip, expected_span) in enumerate(zip(result.data.clips, expected_clips, strict=False)):
            assert clip.span == expected_span
            assert clip.source_video == "test_video.mp4"

            # Verify UUID generation
            expected_start_event = int(expected_span[0] * video.metadata.framerate)
            expected_end_event = int(expected_span[1] * video.metadata.framerate)
            expected_uuid = uuid.uuid5(
                uuid.NAMESPACE_URL, f"test_video.mp4_{expected_start_event}_{expected_end_event}"
            )
            assert clip.uuid == expected_uuid

    def test_min_clip_length_filtering(self):
        """Test that clips shorter than min_clip_length_s are filtered out."""
        # Create a stage with longer minimum clip length
        stage = FixedStrideExtractorStage(
            clip_len_s=2.0,
            clip_stride_s=1.0,
            min_clip_length_s=3.0,  # Longer than clip_len_s
            limit_clips=10,
        )

        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=600,  # 20 seconds
                duration=20.0,
                video_codec="h264",
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # All clips should be filtered out because they're shorter than min_clip_length_s
        assert len(result.data.clips) == 0

    def test_limit_clips_enforcement(self):
        """Test that the limit_clips parameter is respected when clips already exist."""
        stage = FixedStrideExtractorStage(
            clip_len_s=1.0,
            clip_stride_s=0.5,
            min_clip_length_s=0.5,
            limit_clips=3,  # Limit to 3 clips
        )

        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=900,  # 30 seconds
                duration=30.0,
                video_codec="h264",
            ),
            clips=[],
        )

        # Add existing clips to reach the limit
        for i in range(stage.limit_clips):
            clip = Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(i * 1.0, (i + 1) * 1.0))
            video.clips.append(clip)

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Should return the same task without adding new clips
        assert len(result.data.clips) == stage.limit_clips

    def test_no_limit_clips(self):
        """Test behavior when limit_clips is 0 (no limit)."""
        stage = FixedStrideExtractorStage(
            clip_len_s=5.0,
            clip_stride_s=2.5,
            min_clip_length_s=1.0,
            limit_clips=0,  # No limit
        )

        result = stage.process(self.mock_task)

        # Should generate all possible clips without limit
        assert len(result.data.clips) > 0
        # With 30 second video, 5s clips, 2.5s stride: should have 12 clips
        assert len(result.data.clips) == 12

    def test_edge_case_very_short_video(self):
        """Test behavior with very short video."""
        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=30,  # 1 second
                duration=1.0,
                video_codec="h264",
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = self.stage.process(task)

        # Should have one clip from 0.0 to 1.0 (min(clip_len_s, duration))
        assert len(result.data.clips) == 1
        assert result.data.clips[0].span == (0.0, 1.0)

    def test_edge_case_exact_clip_length(self):
        """Test behavior when video duration exactly matches clip length."""
        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=150,  # 5 seconds (exactly clip_len_s)
                duration=5.0,
                video_codec="h264",
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = self.stage.process(task)

        # Should have two clips: (0.0, 5.0) and (2.5, 5.0)
        assert len(result.data.clips) == 2
        assert result.data.clips[0].span == (0.0, 5.0)
        assert result.data.clips[1].span == (2.5, 5.0)

    def test_logging_behavior(self):
        """Test that appropriate logging occurs during processing."""
        with patch("nemo_curator.stages.video.clipping.clip_extraction_stages.logger") as mock_logger:
            self.stage.process(self.mock_task)

            # Verify that info logging occurred
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Extracted" in call_args
            assert "clips from" in call_args
            assert "test_video.mp4" in call_args

    def test_clip_uuid_uniqueness(self):
        """Test that generated clip UUIDs are unique."""
        result = self.stage.process(self.mock_task)

        uuids = [clip.uuid for clip in result.data.clips]
        assert len(uuids) == len(set(uuids))  # All UUIDs should be unique

    def test_clip_span_ordering(self):
        """Test that clips are generated in chronological order."""
        result = self.stage.process(self.mock_task)

        for i in range(len(result.data.clips) - 1):
            current_clip = result.data.clips[i]
            next_clip = result.data.clips[i + 1]

            # Each clip should start after or at the same time as the previous one
            assert current_clip.span[0] <= next_clip.span[0]
            # Clips can overlap based on stride settings

    def test_different_parameter_combinations(self):
        """Test various parameter combinations."""
        test_cases = [
            # format as: (clip_len_s, clip_stride_s, min_clip_length_s, limit_clips)
            (1.0, 0.5, 0.5, 5),
            (5.0, 10.0, 2.0, 0),  # Non-overlapping clips with reasonable min length
            (2.0, 1.0, 1.5, 3),
            (3.0, 3.0, 1.0, 10),  # No overlap between clips
        ]

        for clip_len_s, clip_stride_s, min_clip_length_s, limit_clips in test_cases:
            stage = FixedStrideExtractorStage(
                clip_len_s=clip_len_s,
                clip_stride_s=clip_stride_s,
                min_clip_length_s=min_clip_length_s,
                limit_clips=limit_clips,
            )

            # Create a fresh video for each test case
            video = Video(
                input_video=pathlib.Path("test_video.mp4"),
                source_bytes=b"mock_video_data",
                metadata=VideoMetadata(
                    height=1080,
                    width=1920,
                    framerate=30.0,
                    num_frames=900,  # 30 seconds at 30fps
                    duration=30.0,
                    video_codec="h264",
                    pixel_format="yuv420p",
                    audio_codec="aac",
                    bit_rate_k=5000,
                ),
                clips=[],
            )

            task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

            result = stage.process(task)

            # Basic validation
            assert isinstance(result, VideoTask)
            assert len(result.data.clips) >= 0

            # Validate clip properties
            for clip in result.data.clips:
                assert clip.span[1] - clip.span[0] >= min_clip_length_s
                assert clip.source_video == "test_video.mp4"

    def test_limit_clips_generation(self):
        """Test that limit_clips doesn't limit generation of new clips (current behavior)."""
        stage = FixedStrideExtractorStage(
            clip_len_s=1.0,
            clip_stride_s=0.5,
            min_clip_length_s=0.5,
            limit_clips=3,  # This currently only applies to existing clips
        )

        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=150,  # 5 seconds
                duration=5.0,
                video_codec="h264",
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = stage.process(task)

        # Currently, limit_clips doesn't limit generation, only prevents processing if clips exist
        # For a 5s video with 1s clips and 0.5s stride, we expect 10 clips
        assert len(result.data.clips) == 10  # Not limited to 3

    def test_metadata_validation_edge_cases(self):
        """Test various metadata validation scenarios."""
        # Test with missing video_codec (should still work)
        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=150,
                duration=5.0,
                video_codec=None,  # Missing codec
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = self.stage.process(task)

        # Should return early with incomplete metadata error
        assert isinstance(result, VideoTask)
        assert "metadata" in result.data.errors
        assert result.data.errors["metadata"] == "incomplete"

    def test_negative_duration_calculation(self):
        """Test the edge case where duration calculation might be negative."""
        # Create a video with very small num_frames compared to framerate
        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=1,  # Very small number of frames
                duration=0.033,  # About 1/30 second
                video_codec="h264",
            ),
            clips=[],
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        result = self.stage.process(task)

        # Should process successfully but likely generate no clips due to min_clip_length_s
        assert isinstance(result, VideoTask)
        assert len(result.data.clips) == 0  # No clips meet minimum length requirement
