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
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestCosmosEmbed1FrameCreationStage:
    """Test cases for CosmosEmbed1FrameCreationStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = CosmosEmbed1FrameCreationStage(
            model_dir="test_models/cosmos_embed1", variant="336p", target_fps=2.0, verbose=False, num_cpus=3
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
        assert self.stage.name == "cosmos_embed1"

    def test_inputs_property(self):
        """Test the inputs property."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], [])

    def test_outputs_property(self):
        """Test the outputs property."""
        outputs = self.stage.outputs()
        assert outputs == (["data"], ["clips.cosmos_embed1_frames"])

    def test_initialization(self):
        """Test stage initialization with different parameters."""
        # Test with default parameters
        stage = CosmosEmbed1FrameCreationStage()
        assert stage.model_dir == "models/cosmos_embed1"
        assert stage.variant == "336p"
        assert stage.target_fps == 2.0
        assert stage.verbose is False
        assert stage.num_cpus == 3

        # Test with custom parameters
        stage = CosmosEmbed1FrameCreationStage(
            model_dir="custom_models", variant="448p", target_fps=4.0, verbose=True, num_cpus=8
        )
        assert stage.model_dir == "custom_models"
        assert stage.variant == "448p"
        assert stage.target_fps == 4.0
        assert stage.verbose is True
        assert stage.num_cpus == 8

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_setup(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test setup method."""
        mock_model = Mock()
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Check that the model was initialized correctly
        mock_cosmos_embed1.assert_called_once_with(
            variant="336p", utils_only=True, model_dir="test_models/cosmos_embed1"
        )
        mock_model.setup.assert_called_once()

        # Check that frame extraction signature was created
        assert hasattr(self.stage, "frame_extraction_signature")
        assert hasattr(self.stage, "model")

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_success(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with successful frame processing."""
        # Mock the model
        mock_model = Mock()
        mock_model.get_target_num_frames.return_value = 10
        mock_model.formulate_input_frames.return_value = "mock_input_frames"
        mock_cosmos_embed1.return_value = mock_model

        # Setup the stage
        self.stage.setup()

        # Mock extracted frames
        rng = np.random.default_rng(42)
        mock_frames = rng.random((15, 224, 224, 3))  # 15 frames, more than target
        for clip in self.mock_video.clips:
            clip.extracted_frames = {self.stage.frame_extraction_signature: mock_frames}

        # Process the task
        result = self.stage.process(self.mock_task)

        # Verify the result
        assert isinstance(result, VideoTask)
        assert result.data == self.mock_video

        # Check that frames were processed for each clip
        for clip in result.data.clips:
            assert clip.cosmos_embed1_frames == "mock_input_frames"
            assert len(clip.extracted_frames) == 0  # Should be cleared

        # Verify model calls
        assert mock_model.formulate_input_frames.call_count == 2

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_missing_buffer(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with missing buffer."""
        mock_model = Mock()
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Set buffer to None for first clip
        self.mock_video.clips[0].buffer = None

        result = self.stage.process(self.mock_task)

        # Check that error was recorded
        assert "buffer" in result.data.clips[0].errors
        assert result.data.clips[0].errors["buffer"] == "empty"

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_missing_frames(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with missing extracted frames."""
        mock_model = Mock()
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Don't set extracted_frames for clips
        result = self.stage.process(self.mock_task)

        # Check that error was recorded
        for clip in result.data.clips:
            expected_error_key = f"frames-{self.stage.frame_extraction_signature}"
            assert expected_error_key in clip.errors
            assert clip.errors[expected_error_key] == "missing"

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.extract_frames")
    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_insufficient_frames_reextraction(
        self, mock_cosmos_embed1: "MagicMock", mock_extract_frames: "MagicMock"
    ) -> None:
        """Test process method with insufficient frames requiring re-extraction."""
        # Mock the model
        mock_model = Mock()
        mock_model.get_target_num_frames.return_value = 20  # Need 20 frames
        mock_model.formulate_input_frames.return_value = "mock_input_frames"
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Mock insufficient frames initially
        rng = np.random.default_rng(42)
        insufficient_frames = rng.random((5, 224, 224, 3))  # Only 5 frames
        sufficient_frames = rng.random((25, 224, 224, 3))  # 25 frames after re-extraction

        for clip in self.mock_video.clips:
            clip.extracted_frames = {self.stage.frame_extraction_signature: insufficient_frames}

        # Mock re-extraction to return sufficient frames
        mock_extract_frames.return_value = sufficient_frames

        result = self.stage.process(self.mock_task)

        # Verify re-extraction was called
        assert mock_extract_frames.call_count == 2  # Once for each clip

        # Verify frames were processed
        for clip in result.data.clips:
            assert clip.cosmos_embed1_frames == "mock_input_frames"

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.extract_frames")
    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_clip_too_short(self, mock_cosmos_embed1: "MagicMock", mock_extract_frames: "MagicMock") -> None:
        """Test process method with clip too short to extract enough frames."""
        mock_model = Mock()
        mock_model.get_target_num_frames.return_value = 100  # Very high requirement
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Mock insufficient frames that won't increase even with re-extraction
        rng = np.random.default_rng(42)
        insufficient_frames = rng.random((5, 224, 224, 3))  # Only 5 frames
        for clip in self.mock_video.clips:
            clip.extracted_frames = {self.stage.frame_extraction_signature: insufficient_frames}

        # Mock re-extraction to still return insufficient frames
        mock_extract_frames.return_value = insufficient_frames

        with patch("nemo_curator.stages.video.embedding.cosmos_embed1.logger") as mock_logger:
            self.stage.process(self.mock_task)

            # Verify error was logged
            mock_logger.error.assert_called()
            assert "is too short to extract enough frames" in str(mock_logger.error.call_args)


class TestCosmosEmbed1EmbeddingStage:
    """Test cases for CosmosEmbed1EmbeddingStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = CosmosEmbed1EmbeddingStage(
            model_dir="test_models/cosmos_embed1",
            variant="336p",
            texts_to_verify=["a person walking", "a cat running"],
            gpu_memory_gb=20,
            verbose=False,
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
                    cosmos_embed1_frames="mock_frames_1",
                ),
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(5.0, 10.0),
                    buffer=b"mock_clip_data_2",
                    errors={},
                    cosmos_embed1_frames="mock_frames_2",
                ),
            ],
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self):
        """Test the name property."""
        assert self.stage.name == "cosmos_embed1_embedding"

    def test_inputs_property(self):
        """Test the inputs property."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], [])

    def test_outputs_property(self):
        """Test the outputs property."""
        outputs = self.stage.outputs()
        assert outputs == (["data"], ["clips.cosmos_embed1_embeddings", "clips.cosmos_embed1_embedding"])

    def test_resources_property(self):
        """Test the resources property."""
        resources = self.stage.resources
        assert isinstance(resources, Resources)
        assert resources.gpu_memory_gb == 20

    def test_initialization(self):
        """Test stage initialization with different parameters."""
        # Test with default parameters
        stage = CosmosEmbed1EmbeddingStage(gpu_memory_gb=8)
        assert stage.model_dir == "models/cosmos_embed1"
        assert stage.variant == "336p"
        assert stage.texts_to_verify is None
        assert stage.gpu_memory_gb == 8
        assert stage.verbose is False

        # Test with custom parameters
        stage = CosmosEmbed1EmbeddingStage(
            model_dir="custom_models", variant="448p", texts_to_verify=["custom text"], gpu_memory_gb=8, verbose=True
        )
        assert stage.model_dir == "custom_models"
        assert stage.variant == "448p"
        assert stage.texts_to_verify == ["custom text"]
        assert stage.gpu_memory_gb == 8
        assert stage.verbose is True

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_setup(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test setup method."""
        mock_model = Mock()
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Check that the model was initialized correctly
        mock_cosmos_embed1.assert_called_once_with(
            variant="336p", utils_only=False, model_dir="test_models/cosmos_embed1"
        )
        mock_model.setup.assert_called_once()

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_success(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with successful embedding."""
        # Mock the model
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_embedding.numel.return_value = 3
        mock_model.encode_video_frames.return_value = mock_embedding
        mock_model.get_text_embedding.return_value = "text_embedding"
        mock_model.evaluate.return_value = ([0.8, 0.2], [0, 1])  # probs, indices
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        result = self.stage.process(self.mock_task)

        # Verify the result
        assert isinstance(result, VideoTask)
        assert result.data == self.mock_video

        # Check that embeddings were created for each clip
        for clip in result.data.clips:
            np.testing.assert_array_equal(clip.cosmos_embed1_embedding, np.array([1.0, 2.0, 3.0]))
            assert clip.cosmos_embed1_frames is None  # Should be cleared

        # Verify model calls
        assert mock_model.encode_video_frames.call_count == 2

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_with_text_verification(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with text verification."""
        # Mock the model
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_embedding.numel.return_value = 3
        mock_model.encode_video_frames.return_value = mock_embedding
        mock_model.get_text_embedding.return_value = "text_embedding"
        mock_model.evaluate.return_value = ([0.8, 0.2], [0, 1])  # probs, indices
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        result = self.stage.process(self.mock_task)

        # Check that text verification was performed
        for clip in result.data.clips:
            assert hasattr(clip, "cosmos_embed1_text_match")
            assert clip.cosmos_embed1_text_match == ("a person walking", 0.8)

        # Verify model calls
        assert mock_model.get_text_embedding.call_count == 4  # 2 texts x 2 clips
        assert mock_model.evaluate.call_count == 2

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_missing_frames(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with missing cosmos_embed1_frames."""
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_embedding.numel.return_value = 3
        mock_model.encode_video_frames.return_value = mock_embedding
        mock_model.get_text_embedding.return_value = "text_embedding"
        mock_model.evaluate.return_value = ([0.8, 0.2], [0, 1])  # probs, indices
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        # Set frames to None for first clip
        self.mock_video.clips[0].cosmos_embed1_frames = None

        result = self.stage.process(self.mock_task)

        # Check that error was recorded
        assert "cosmos_embed1_frames" in result.data.clips[0].errors
        assert result.data.clips[0].errors["cosmos_embed1_frames"] == "empty"

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_empty_embedding(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method with empty embedding."""
        # Mock the model to return empty embedding
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.numel.return_value = 0  # Empty embedding
        mock_model.encode_video_frames.return_value = mock_embedding
        mock_model.get_text_embedding.return_value = "text_embedding"
        mock_model.evaluate.return_value = ([0.8, 0.2], [0, 1])  # probs, indices
        mock_cosmos_embed1.return_value = mock_model

        self.stage.setup()

        with patch("nemo_curator.stages.video.embedding.cosmos_embed1.logger") as mock_logger:
            result = self.stage.process(self.mock_task)

            # Check that error was logged and recorded
            mock_logger.error.assert_called()
            for clip in result.data.clips:
                assert "cosmos_embed1_embedding" in clip.errors
                assert clip.errors["cosmos_embed1_embedding"] == "failed"

    @patch("nemo_curator.stages.video.embedding.cosmos_embed1.CosmosEmbed1")
    def test_process_without_text_verification(self, mock_cosmos_embed1: "MagicMock") -> None:
        """Test process method without text verification."""
        # Create stage without texts_to_verify
        stage = CosmosEmbed1EmbeddingStage(texts_to_verify=None)

        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_embedding.numel.return_value = 3
        mock_model.encode_video_frames.return_value = mock_embedding
        mock_cosmos_embed1.return_value = mock_model

        stage.setup()

        # Create fresh video with clips that don't have cosmos_embed1_text_match
        fresh_video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=self.mock_video.metadata,
            clips=[
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(0.0, 5.0),
                    buffer=b"mock_clip_data_1",
                    errors={},
                    cosmos_embed1_frames="mock_frames_1",
                ),
                Clip(
                    uuid=uuid.uuid4(),
                    source_video="test_video.mp4",
                    span=(5.0, 10.0),
                    buffer=b"mock_clip_data_2",
                    errors={},
                    cosmos_embed1_frames="mock_frames_2",
                ),
            ],
        )

        fresh_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=fresh_video)

        result = stage.process(fresh_task)

        # Check that text verification was not performed
        for clip in result.data.clips:
            # The attribute may exist but should be None if no text verification occurred
            assert getattr(clip, "cosmos_embed1_text_match", None) is None

        # Verify model calls
        assert mock_model.get_text_embedding.call_count == 0
        assert mock_model.evaluate.call_count == 0
