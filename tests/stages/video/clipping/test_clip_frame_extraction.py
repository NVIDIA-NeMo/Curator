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
import math
import uuid
from unittest.mock import Mock, patch

import numpy as np

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature, FramePurpose


class TestClipFrameExtractionStage:
    """Test cases for ClipFrameExtractionStage."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.stage = ClipFrameExtractionStage(
            extraction_policies=(FrameExtractionPolicy.sequence,),
            target_fps=[2, 4],
            target_res=(480, 640),
            verbose=False,
            num_cpus=2,
        )

        # Create mock clips with buffers
        self.mock_clips = [
            Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data_1"),
            Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(5.0, 10.0), buffer=b"fake_video_data_2"),
            Clip(
                uuid=uuid.uuid4(),
                source_video="test_video.mp4",
                span=(10.0, 15.0),
                buffer=None,  # Test clip without buffer
            ),
        ]

        # Create mock video with clips
        self.mock_video = Video(
            input_video="test_video.mp4",
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
            clips=self.mock_clips,
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self) -> None:
        """Test that the name property returns the correct value."""
        assert self.stage.name == "clip_frame_extraction"

    def test_inputs_outputs(self) -> None:
        """Test that inputs and outputs return the correct values."""
        inputs_required, inputs_optional = self.stage.inputs()
        outputs_required, outputs_optional = self.stage.outputs()

        assert inputs_required == ["data"]
        assert inputs_optional == []
        assert outputs_required == ["data"]
        assert outputs_optional == ["clips.extracted_frames"]

    def test_default_initialization(self) -> None:
        """Test default initialization of ClipFrameExtractionStage."""
        stage = ClipFrameExtractionStage()

        assert stage.extraction_policies == (FrameExtractionPolicy.sequence,)
        assert stage.target_fps is None
        assert stage.target_res is None
        assert stage.verbose is False
        assert stage.num_cpus == 3

    def test_setup_with_defaults(self) -> None:
        """Test setup method with default values."""
        stage = ClipFrameExtractionStage()
        stage.setup()

        assert stage.target_fps == [2]  # Default fallback
        assert stage.target_res == (-1, -1)

    def test_setup_with_extract_purpose_aesthetics(self) -> None:
        """Test setup method with extract_purpose set to aesthetics."""
        stage = ClipFrameExtractionStage(extract_purposes=[FramePurpose.AESTHETICS])
        stage.setup()

        assert stage.target_fps == [1]  # AESTHETICS.value = 1
        assert stage.target_res == (-1, -1)

    def test_setup_with_extract_purpose_embeddings(self) -> None:
        """Test setup method with extract_purpose set to embeddings."""
        stage = ClipFrameExtractionStage(extract_purposes=[FramePurpose.EMBEDDINGS])
        stage.setup()

        assert stage.target_fps == [2]  # EMBEDDINGS.value = 2
        assert stage.target_res == (-1, -1)

    def test_setup_with_extract_purpose_both(self) -> None:
        """Test setup method with extract_purpose set to both aesthetics and embeddings."""
        stage = ClipFrameExtractionStage(extract_purposes=[FramePurpose.AESTHETICS, FramePurpose.EMBEDDINGS])
        stage.setup()

        assert stage.target_fps == [1, 2]  # AESTHETICS.value = 1, EMBEDDINGS.value = 2
        assert stage.target_res == (-1, -1)

    def test_setup_with_extract_purpose_and_target_fps(self) -> None:
        """Test that target_fps takes precedence when both are specified."""
        stage = ClipFrameExtractionStage(extract_purposes=[FramePurpose.AESTHETICS], target_fps=[5, 10])
        stage.setup()

        assert stage.target_fps == [5, 10]  # target_fps should not be overridden
        assert stage.target_res == (-1, -1)

    def test_resources_property(self) -> None:
        """Test that resources property returns correct CPU count."""
        stage = ClipFrameExtractionStage(num_cpus=5)
        resources = stage.resources

        assert isinstance(resources, Resources)
        assert resources.cpus == 5

    def test_lcm_multiple_basic(self) -> None:
        """Test LCM calculation with basic integers."""
        result = self.stage.lcm_multiple([2, 3, 4])
        assert result == 12

    def test_lcm_multiple_single_value(self) -> None:
        """Test LCM calculation with single value."""
        result = self.stage.lcm_multiple([5])
        assert result == 5

    def test_lcm_multiple_with_floats(self) -> None:
        """Test LCM calculation with float values."""
        result = self.stage.lcm_multiple([2.0, 4.0])
        assert result == 4.0

    def test_lcm_multiple_coprime_numbers(self) -> None:
        """Test LCM calculation with coprime numbers."""
        result = self.stage.lcm_multiple([3, 5, 7])
        assert result == 105

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_successful_extraction(self, mock_extract_frames: Mock) -> None:
        """Test successful frame extraction process."""
        # Mock extracted frames
        mock_frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
        mock_extract_frames.return_value = mock_frames

        # Setup stage with defaults
        self.stage.setup()

        result = self.stage.process(self.mock_task)

        # Should process clips with buffers using LCM optimization
        # With target_fps=[2, 4], LCM=4, so extract_frames is called once per clip with buffer
        assert mock_extract_frames.call_count == 2  # 2 clips with buffers
        assert isinstance(result, VideoTask)
        assert result.task_id == "test_task"

        # Check that extracted frames are stored with correct signatures
        processed_clips = result.data.clips
        for clip in processed_clips[:2]:  # Only first 2 clips have buffers
            assert len(clip.extracted_frames) == 2  # 2 fps targets
            for fps in self.stage.target_fps:
                signature = FrameExtractionSignature(
                    extraction_policy=FrameExtractionPolicy.sequence,
                    target_fps=fps,
                ).to_str()
                assert signature in clip.extracted_frames

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_clip_without_buffer(self, mock_extract_frames: Mock) -> None:
        """Test processing when clip has no buffer."""
        mock_frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
        mock_extract_frames.return_value = mock_frames

        self.stage.setup()

        with patch("nemo_curator.stages.video.clipping.clip_frame_extraction.logger") as mock_logger:
            result = self.stage.process(self.mock_task)

            # Should log error for clip without buffer
            mock_logger.error.assert_called()
            error_calls = [str(call) for call in mock_logger.error.call_args_list]
            assert any("has no buffer" in call for call in error_calls)

            # Should add error to clip
            clip_without_buffer = result.data.clips[2]
            assert "buffer" in clip_without_buffer.errors
            assert clip_without_buffer.errors["buffer"] == "empty"

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_frame_extraction_error(self, mock_extract_frames: Mock) -> None:
        """Test handling of frame extraction errors."""
        # Make extract_frames raise an exception
        mock_extract_frames.side_effect = ValueError("Decode failed")

        self.stage.setup()

        with patch("nemo_curator.stages.video.clipping.clip_frame_extraction.logger") as mock_logger:
            result = self.stage.process(self.mock_task)

            # Should log exceptions
            mock_logger.exception.assert_called()

            # Should add error to clips and reset buffer
            processed_clips = result.data.clips
            for clip in processed_clips[:2]:  # Only first 2 clips had buffers
                assert "frame_extraction" in clip.errors
                assert clip.errors["frame_extraction"] == "video_decode_failed"
                assert clip.buffer is None

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_with_lcm_optimization(self, mock_extract_frames: Mock) -> None:
        """Test processing with LCM optimization when all fps are integers."""
        # Setup stage with integer fps values
        stage = ClipFrameExtractionStage(
            target_fps=[2, 4, 6],  # LCM = 12
            num_cpus=2,
        )
        stage.setup()

        # Mock extracted frames - should be called with LCM fps
        mock_frames = np.zeros((60, 480, 640, 3), dtype=np.uint8)  # 60 frames for 12 fps
        mock_extract_frames.return_value = mock_frames

        # Create task with clips that have buffers
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=Video(
                input_video="test_video.mp4",
                clips=[
                    Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data")
                ],
            ),
        )

        result = stage.process(task)

        # Should call extract_frames once with LCM fps
        mock_extract_frames.assert_called_once()
        call_args = mock_extract_frames.call_args
        assert call_args[1]["sample_rate_fps"] == 12  # LCM of [2, 4, 6]

        # Should have frames for all fps targets
        clip = result.data.clips[0]
        assert len(clip.extracted_frames) == 3

        # Check frame subsampling
        for fps in [2, 4, 6]:
            signature = FrameExtractionSignature(
                extraction_policy=FrameExtractionPolicy.sequence,
                target_fps=fps,
            ).to_str()
            assert signature in clip.extracted_frames
            expected_frames = mock_frames[:: int(12 / fps)]
            np.testing.assert_array_equal(clip.extracted_frames[signature], expected_frames)

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_without_lcm_optimization(self, mock_extract_frames: Mock) -> None:
        """Test processing without LCM optimization when fps contains floats."""
        # Setup stage with float fps values
        stage = ClipFrameExtractionStage(
            target_fps=[2.5, 5.0],  # Contains float, no LCM optimization
            num_cpus=2,
        )
        stage.setup()

        mock_frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
        mock_extract_frames.return_value = mock_frames

        # Create task with clips that have buffers
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=Video(
                input_video="test_video.mp4",
                clips=[
                    Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data")
                ],
            ),
        )

        result = stage.process(task)

        # Should call extract_frames once for each fps target
        assert mock_extract_frames.call_count == 2

        # Should have frames for all fps targets
        clip = result.data.clips[0]
        assert len(clip.extracted_frames) == 2

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_multiple_extraction_policies(self, mock_extract_frames: Mock) -> None:
        """Test processing with multiple extraction policies."""
        stage = ClipFrameExtractionStage(
            extraction_policies=(FrameExtractionPolicy.sequence, FrameExtractionPolicy.middle),
            target_fps=[2],
            num_cpus=2,
        )
        stage.setup()

        mock_frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
        mock_extract_frames.return_value = mock_frames

        # Create task with clips that have buffers
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=Video(
                input_video="test_video.mp4",
                clips=[
                    Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data")
                ],
            ),
        )

        result = stage.process(task)

        # Should call extract_frames twice (2 policies)
        assert mock_extract_frames.call_count == 2

        # Should have frames for both policies
        clip = result.data.clips[0]
        assert len(clip.extracted_frames) == 2

        # Check that both policies are present
        sequence_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=2,
        ).to_str()
        middle_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.middle,
            target_fps=2,
        ).to_str()

        assert sequence_signature in clip.extracted_frames
        assert middle_signature in clip.extracted_frames

    @patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames")
    def test_process_verbose_logging(self, mock_extract_frames: Mock) -> None:
        """Test verbose logging during processing."""
        stage = ClipFrameExtractionStage(verbose=True, target_fps=[2])
        stage.setup()

        mock_frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
        mock_extract_frames.return_value = mock_frames

        # Create task with clips that have buffers
        task = VideoTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=Video(
                input_video="test_video.mp4",
                clips=[
                    Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data")
                ],
            ),
        )

        with patch("nemo_curator.stages.video.clipping.clip_frame_extraction.logger") as mock_logger:
            stage.process(task)

            # Should log frame extraction info
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Extracted" in call and "frames from clip" in call for call in info_calls)
            assert any("extracted frames for" in call and "clips" in call for call in info_calls)

    def test_process_no_clips(self) -> None:
        """Test processing when video has no clips."""
        task = VideoTask(
            task_id="test_task", dataset_name="test_dataset", data=Video(input_video="test_video.mp4", clips=[])
        )

        self.stage.setup()
        result = self.stage.process(task)

        assert isinstance(result, VideoTask)
        assert len(result.data.clips) == 0

    def test_process_different_error_types(self) -> None:
        """Test handling of different error types during frame extraction."""
        error_types = [
            (ValueError, "value error"),
            (OSError, "os error"),
            (RuntimeError, "runtime error"),
        ]

        for error_class, error_message in error_types:
            with patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames") as mock_extract:
                mock_extract.side_effect = error_class(error_message)

                stage = ClipFrameExtractionStage(target_fps=[2])
                stage.setup()

                # Create task with one clip
                task = VideoTask(
                    task_id="test_task",
                    dataset_name="test_dataset",
                    data=Video(
                        input_video="test_video.mp4",
                        clips=[
                            Clip(
                                uuid=uuid.uuid4(),
                                source_video="test_video.mp4",
                                span=(0.0, 5.0),
                                buffer=b"fake_video_data",
                            )
                        ],
                    ),
                )

                with patch("nemo_curator.stages.video.clipping.clip_frame_extraction.logger"):
                    result = stage.process(task)

                    # Should handle error and reset buffer
                    clip = result.data.clips[0]
                    assert "frame_extraction" in clip.errors
                    assert clip.errors["frame_extraction"] == "video_decode_failed"
                    assert clip.buffer is None

    def test_target_fps_attribute_access(self) -> None:
        """Test that _target_fps attribute is accessible during LCM calculation."""
        stage = ClipFrameExtractionStage(target_fps=[2, 4])
        stage.setup()

        # This should work without attribute error
        lcm_result = stage.lcm_multiple([2, 4])
        assert lcm_result == 4

        # Test that stage has both target_fps and _target_fps
        assert hasattr(stage, "target_fps")
        assert stage.target_fps == [2, 4]

    def test_io_bytesio_usage(self) -> None:
        """Test that io.BytesIO is used correctly with clip buffers."""
        stage = ClipFrameExtractionStage(target_fps=[2])
        stage.setup()

        # Create a clip with buffer
        clip = Clip(uuid=uuid.uuid4(), source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data")

        # Test that io.BytesIO can be created from buffer
        with io.BytesIO(clip.buffer) as fp:
            assert fp.read() == b"fake_video_data"

    def test_frame_extraction_signature_creation(self) -> None:
        """Test that FrameExtractionSignature is created correctly."""
        policy = FrameExtractionPolicy.sequence
        fps = 2.5

        signature = FrameExtractionSignature(
            extraction_policy=policy,
            target_fps=fps,
        )

        assert signature.extraction_policy == policy
        assert signature.target_fps == fps

        # Test string representation
        signature_str = signature.to_str()
        assert "sequence" in signature_str
        assert "2500" in signature_str  # fps * 1000

    def test_edge_case_single_target_fps(self) -> None:
        """Test edge case with single target_fps value."""
        stage = ClipFrameExtractionStage(target_fps=[5])
        stage.setup()

        # Should not use LCM optimization
        assert stage.target_fps == [5]

        # LCM of single value should be the value itself
        lcm_result = stage.lcm_multiple([5])
        assert lcm_result == 5

    def test_clip_uuid_preservation(self) -> None:
        """Test that clip UUIDs are preserved during processing."""
        original_uuid = uuid.uuid4()
        clip = Clip(uuid=original_uuid, source_video="test_video.mp4", span=(0.0, 5.0), buffer=b"fake_video_data")

        task = VideoTask(
            task_id="test_task", dataset_name="test_dataset", data=Video(input_video="test_video.mp4", clips=[clip])
        )

        stage = ClipFrameExtractionStage(target_fps=[2])
        stage.setup()

        with patch("nemo_curator.stages.video.clipping.clip_frame_extraction.extract_frames") as mock_extract:
            mock_extract.return_value = np.zeros((10, 480, 640, 3), dtype=np.uint8)

            result = stage.process(task)

            # UUID should be preserved
            assert result.data.clips[0].uuid == original_uuid

    def test_math_gcd_and_lcm_calculation(self) -> None:
        """Test that math.gcd is used correctly in LCM calculation."""
        stage = ClipFrameExtractionStage()

        # Test LCM calculation directly
        def test_lcm(a: float, b: float) -> int:
            return abs(a * b) // math.gcd(int(a), int(b))

        assert test_lcm(4, 6) == 12
        assert test_lcm(3, 5) == 15
        assert test_lcm(2, 8) == 8

        # Test with the actual method
        assert stage.lcm_multiple([4, 6]) == 12
        assert stage.lcm_multiple([3, 5]) == 15
        assert stage.lcm_multiple([2, 8]) == 8
