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
from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from nemo_curator.stages.video.clipping.transnetv2_extraction import (
    TransNetV2ClipExtractionStage,
    _create_spans,
    _crop_scenes,
    _get_batches,
    _get_filtered_scenes,
    _get_predictions,
    _get_scenes,
)
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask

# Create a random number generator for test data
rng = np.random.default_rng(42)


class TestTransNetV2ClipExtractionStage:
    """Test cases for TransNetV2ClipExtractionStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = TransNetV2ClipExtractionStage(
            threshold=0.4,
            min_length_s=2.0,
            max_length_s=10.0,
            max_length_mode="stride",
            crop_s=0.5,
            entire_scene_as_clip=True,
            gpu_memory_gb=10,
            limit_clips=5,
            verbose=True,
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
            frame_array=rng.integers(0, 255, (900, 27, 48, 3), dtype=np.uint8),
        )

        self.mock_task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=self.mock_video)

    def test_name_property(self):
        """Test the name property."""
        assert self.stage.name == "transnetv2_clip_extraction"

    def test_inputs_property(self):
        """Test the inputs property."""
        inputs, input_columns = self.stage.inputs()
        assert inputs == ["data"]
        assert input_columns == []

    def test_outputs_property(self):
        """Test the outputs property."""
        outputs, output_columns = self.stage.outputs()
        assert outputs == ["data"]
        assert output_columns == ["clips.extracted_frames"]

    def test_resources_property(self):
        """Test the resources property."""
        resources = self.stage.resources
        assert resources.gpu_memory_gb == 10

    def test_stage_initialization_defaults(self):
        """Test stage initialization with default parameters."""
        stage = TransNetV2ClipExtractionStage()
        assert stage.threshold == 0.4
        assert stage.min_length_s == 2.0
        assert stage.max_length_s == 10.0
        assert stage.max_length_mode == "stride"
        assert stage.crop_s == 0.5
        assert stage.entire_scene_as_clip is True
        assert stage.gpu_memory_gb == 10
        assert stage.limit_clips == -1
        assert stage.verbose is False

    def test_stage_initialization_custom_params(self):
        """Test stage initialization with custom parameters."""
        stage = TransNetV2ClipExtractionStage(
            threshold=0.6,
            min_length_s=1.0,
            max_length_s=20.0,
            max_length_mode="truncate",
            crop_s=1.0,
            entire_scene_as_clip=False,
            gpu_memory_gb=20,
            limit_clips=10,
            verbose=True,
        )
        assert stage.threshold == 0.6
        assert stage.min_length_s == 1.0
        assert stage.max_length_s == 20.0
        assert stage.max_length_mode == "truncate"
        assert stage.crop_s == 1.0
        assert stage.entire_scene_as_clip is False
        assert stage.gpu_memory_gb == 20
        assert stage.limit_clips == 10
        assert stage.verbose is True

    @patch("nemo_curator.models.transnetv2.TransNetV2")
    def test_setup(self, mock_transnetv2_class: Mock):
        """Test setup method."""
        mock_model = Mock()
        mock_transnetv2_class.return_value = mock_model

        self.stage.setup()

        mock_transnetv2_class.assert_called_once()
        mock_model.setup.assert_called_once()
        assert self.stage._model == mock_model

    @patch("nemo_curator.models.transnetv2.TransNetV2")
    def test_process_success(self, mock_transnetv2_class: Mock):
        """Test successful processing of a video task."""
        # Mock the model
        mock_model = Mock()
        mock_predictions = torch.tensor([[0], [1], [0], [0], [1], [0]], dtype=torch.uint8)
        mock_model.return_value = mock_predictions
        mock_transnetv2_class.return_value = mock_model

        # Setup stage
        self.stage.setup()

        # Mock the _get_predictions function to return controlled output
        with (
            patch("nemo_curator.stages.video.clipping.transnetv2_extraction._get_predictions") as mock_get_predictions,
            patch("nemo_curator.stages.video.clipping.transnetv2_extraction._get_scenes") as mock_get_scenes,
            patch(
                "nemo_curator.stages.video.clipping.transnetv2_extraction._get_filtered_scenes"
            ) as mock_get_filtered_scenes,
        ):
            mock_get_predictions.return_value = np.array([[0], [1], [0], [0], [1], [0]], dtype=np.uint8)
            mock_get_scenes.return_value = np.array([[0, 30], [60, 90]], dtype=np.int32)
            mock_get_filtered_scenes.return_value = np.array([[0, 30], [60, 90]], dtype=np.int32)

            # Process the task
            result = self.stage.process(self.mock_task)

            # Verify the result
            assert result == self.mock_task
            assert isinstance(result.data, Video)
            assert len(result.data.clips) > 0
            assert result.data.frame_array is None  # Should be cleared after processing

    def test_process_no_metadata(self):
        """Test processing with incomplete metadata."""
        # Create video without metadata
        video_without_metadata = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=None,
            clips=[],
            frame_array=rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8),
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video_without_metadata)

        with patch("nemo_curator.models.transnetv2.TransNetV2"):
            self.stage.setup()

            # Mock the has_metadata method to return False for null metadata
            with patch.object(video_without_metadata, "has_metadata", return_value=False):
                result = self.stage.process(task)

            assert result == task
            assert len(result.data.clips) == 0

    def test_process_no_framerate(self):
        """Test processing with no framerate metadata."""
        # Create video with metadata but no framerate
        video_no_framerate = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=None,
                num_frames=900,
                duration=30.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[],
            frame_array=rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8),
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video_no_framerate)

        with patch("nemo_curator.models.transnetv2.TransNetV2"):
            self.stage.setup()
            result = self.stage.process(task)

            assert result == task
            assert len(result.data.clips) == 0

    def test_process_no_frame_array(self):
        """Test processing with no frame array."""
        # Create video without frame array
        video_no_frames = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=900,
                duration=30.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[],
            frame_array=None,
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video_no_frames)

        with patch("nemo_curator.models.transnetv2.TransNetV2"):
            self.stage.setup()

            with pytest.raises(ValueError, match="Run `FrameExtractionStage` stage prior"):
                self.stage.process(task)

    def test_process_wrong_frame_shape(self):
        """Test processing with wrong frame shape."""
        # Create video with wrong frame shape
        video_wrong_shape = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"mock_video_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=900,
                duration=30.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[],
            frame_array=rng.integers(0, 255, (100, 28, 48, 3), dtype=np.uint8),  # Wrong height
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video_wrong_shape)

        with patch("nemo_curator.models.transnetv2.TransNetV2"):
            self.stage.setup()

            with pytest.raises(ValueError, match="Expected frames of shape 27x48x3"):
                self.stage.process(task)

    @patch("nemo_curator.models.transnetv2.TransNetV2")
    def test_process_with_limit_clips(self, mock_transnetv2_class: Mock):
        """Test processing with clip limit."""
        # Create stage with clip limit
        stage = TransNetV2ClipExtractionStage(limit_clips=2)

        # Mock the model
        mock_model = Mock()
        mock_transnetv2_class.return_value = mock_model
        stage.setup()

        # Mock helper functions to return many clips
        with (
            patch("nemo_curator.stages.video.clipping.transnetv2_extraction._get_predictions") as mock_get_predictions,
            patch("nemo_curator.stages.video.clipping.transnetv2_extraction._get_scenes") as mock_get_scenes,
            patch(
                "nemo_curator.stages.video.clipping.transnetv2_extraction._get_filtered_scenes"
            ) as mock_get_filtered_scenes,
        ):
            mock_get_predictions.return_value = np.array([[0], [1], [0], [1], [0]], dtype=np.uint8)
            mock_get_scenes.return_value = np.array([[0, 30], [60, 90], [120, 150]], dtype=np.int32)
            mock_get_filtered_scenes.return_value = np.array([[0, 30], [60, 90], [120, 150]], dtype=np.int32)

            result = stage.process(self.mock_task)

            # Should only have 2 clips due to limit
            assert len(result.data.clips) == 2


class TestGetBatches:
    """Test cases for _get_batches helper function."""

    def test_get_batches_basic(self):
        """Test basic functionality of _get_batches."""
        frames = rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8)

        batches = list(_get_batches(frames))

        assert len(batches) == 2  # 100 frames should create 2 batches
        # The actual batch sizes depend on the algorithm with padding
        assert batches[0].shape[0] == 100  # First batch size
        assert batches[1].shape[0] == 75  # Second batch size
        assert batches[0].shape[1:] == (27, 48, 3)  # Frame dimensions
        assert batches[1].shape[1:] == (27, 48, 3)  # Frame dimensions

    def test_get_batches_exact_multiple(self):
        """Test _get_batches with exact multiple of 50."""
        frames = rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8)

        batches = list(_get_batches(frames))

        assert len(batches) == 2
        # Check that all batches have the correct frame dimensions
        for batch in batches:
            assert batch.shape[1:] == (27, 48, 3)

    def test_get_batches_small_input(self):
        """Test _get_batches with small input."""
        frames = rng.integers(0, 255, (10, 27, 48, 3), dtype=np.uint8)

        batches = list(_get_batches(frames))

        assert len(batches) == 1
        # Should be padded with start and end frames
        assert batches[0].shape[0] == 35  # Actual batch size with padding
        assert batches[0].shape[1:] == (27, 48, 3)

    def test_get_batches_generator_type(self):
        """Test that _get_batches returns a generator."""
        frames = rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8)

        result = _get_batches(frames)

        assert isinstance(result, Generator)


class TestGetPredictions:
    """Test cases for _get_predictions helper function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.frames = rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8)

    def test_get_predictions_basic(self):
        """Test basic functionality of _get_predictions."""
        # Mock model - it should return predictions for all frames in batch
        mock_model = Mock()

        # Create a side effect that returns different shapes based on batch size
        def model_side_effect(batch_input: torch.Tensor) -> torch.Tensor:
            batch_size = batch_input.shape[1]  # Number of frames in batch
            return torch.tensor([[[0.3] * batch_size]], dtype=torch.float32).reshape(1, batch_size, 1)

        mock_model.side_effect = model_side_effect

        # Mock CUDA to avoid GPU requirements
        def mock_cuda_method(self: torch.Tensor) -> torch.Tensor:
            return self

        with patch("torch.Tensor.cuda", mock_cuda_method):
            # Use the actual _get_batches function
            predictions = _get_predictions(mock_model, self.frames, threshold=0.5)

        assert predictions.shape == (100, 1)
        assert predictions.dtype == np.uint8
        # All predictions should be 0 since 0.3 < 0.5
        assert np.all(predictions == 0)

    def test_get_predictions_invalid_frames_shape(self):
        """Test _get_predictions with invalid frames shape."""
        invalid_frames = rng.integers(0, 255, (100, 27, 48), dtype=np.uint8)  # Missing channel dimension
        mock_model = Mock()

        with pytest.raises(ValueError, match="Expected frames tensor to have rank 4"):
            _get_predictions(mock_model, invalid_frames, threshold=0.5)

    def test_get_predictions_threshold_application(self):
        """Test threshold application in _get_predictions."""
        mock_model = Mock()

        # Create a side effect that returns different shapes based on batch size
        def model_side_effect(batch_input: torch.Tensor) -> torch.Tensor:
            batch_size = batch_input.shape[1]  # Number of frames in batch
            # Create predictable values - place test values at positions 25-28 so they appear in final output
            values = [0.5] * batch_size  # Default values
            if batch_size > 28:  # Make sure we have enough frames
                values[25:29] = [0.3, 0.7, 0.2, 0.9]  # Place at positions that will be extracted
            return torch.tensor([values], dtype=torch.float32).reshape(1, batch_size, 1)

        mock_model.side_effect = model_side_effect

        # Mock CUDA to avoid GPU requirements
        def mock_cuda_method(self: torch.Tensor) -> torch.Tensor:
            return self

        with patch("torch.Tensor.cuda", mock_cuda_method):
            # Use the actual _get_batches function
            predictions = _get_predictions(mock_model, self.frames, threshold=0.6)

        # Only values > 0.6 should be 1
        expected = np.array([[0], [1], [0], [1]], dtype=np.uint8)
        assert np.array_equal(predictions[:4], expected)


class TestGetScenes:
    """Test cases for _get_scenes helper function."""

    def test_get_scenes_basic(self):
        """Test basic functionality of _get_scenes."""
        predictions = np.array([[0], [1], [0], [0], [1], [0]], dtype=np.uint8)

        scenes = _get_scenes(predictions, entire_scene_as_clip=True)

        # The actual implementation creates scenes based on transitions
        assert scenes.shape[0] >= 1  # At least one scene
        assert scenes.shape[1] == 2
        assert scenes.dtype == np.int32
        # Check that all scenes have valid start <= end (some may be single frame)
        for i in range(scenes.shape[0]):
            assert scenes[i, 0] <= scenes[i, 1]

    def test_get_scenes_no_transitions(self):
        """Test _get_scenes with no transitions."""
        predictions = np.array([[0], [0], [0], [0], [0]], dtype=np.uint8)

        scenes = _get_scenes(predictions, entire_scene_as_clip=True)

        assert scenes.shape == (1, 2)
        assert scenes[0, 0] == 0
        assert scenes[0, 1] == 5

    def test_get_scenes_no_transitions_no_entire_scene(self):
        """Test _get_scenes with no transitions and entire_scene_as_clip=False."""
        predictions = np.array([[0], [0], [0], [0], [0]], dtype=np.uint8)

        scenes = _get_scenes(predictions, entire_scene_as_clip=False)

        assert scenes.shape == (0, 2)

    def test_get_scenes_all_transitions(self):
        """Test _get_scenes with all transitions."""
        predictions = np.array([[1], [1], [1], [1], [1]], dtype=np.uint8)

        scenes = _get_scenes(predictions, entire_scene_as_clip=True)

        # Should have many small scenes
        assert scenes.shape[0] >= 1
        assert scenes.shape[1] == 2

    def test_get_scenes_single_transition(self):
        """Test _get_scenes with single transition."""
        predictions = np.array([[0], [0], [1], [0], [0]], dtype=np.uint8)

        scenes = _get_scenes(predictions, entire_scene_as_clip=True)

        assert scenes.shape == (2, 2)
        assert scenes[0, 0] == 0
        assert scenes[0, 1] == 2
        assert scenes[1, 0] == 3  # Scene starts after transition
        assert scenes[1, 1] == 4


class TestGetFilteredScenes:
    """Test cases for _get_filtered_scenes helper function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scenes = np.array([[0, 30], [30, 90], [90, 150]], dtype=np.int32)

    def test_get_filtered_scenes_basic(self):
        """Test basic functionality of _get_filtered_scenes."""
        filtered = _get_filtered_scenes(self.scenes)

        assert filtered.shape == (3, 2)
        assert np.array_equal(filtered, self.scenes)

    def test_get_filtered_scenes_min_length(self):
        """Test _get_filtered_scenes with minimum length filter."""
        filtered = _get_filtered_scenes(self.scenes, min_length=40)

        assert filtered.shape == (2, 2)  # First scene (length 30) should be filtered out
        assert np.array_equal(filtered, self.scenes[1:])

    def test_get_filtered_scenes_max_length_truncate(self):
        """Test _get_filtered_scenes with maximum length truncation."""
        filtered = _get_filtered_scenes(self.scenes, max_length=50, max_length_mode="truncate")

        assert filtered.shape == (3, 2)
        assert filtered[0, 1] == 30  # First scene unchanged
        assert filtered[1, 1] == 80  # Second scene truncated (30 + 50)
        assert filtered[2, 1] == 140  # Third scene truncated (90 + 50)

    def test_get_filtered_scenes_max_length_stride(self):
        """Test _get_filtered_scenes with maximum length stride."""
        filtered = _get_filtered_scenes(self.scenes, max_length=50, max_length_mode="stride")

        # Should create more scenes due to striding
        assert filtered.shape[0] >= 3
        assert filtered.shape[1] == 2

    def test_get_filtered_scenes_crop_length(self):
        """Test _get_filtered_scenes with crop length."""
        filtered = _get_filtered_scenes(self.scenes, crop_length=5)

        assert filtered.shape == (3, 2)
        assert filtered[0, 0] == 5  # Start cropped by 5
        assert filtered[0, 1] == 25  # End cropped by 5
        assert filtered[1, 0] == 35  # Start cropped by 5
        assert filtered[1, 1] == 85  # End cropped by 5

    def test_get_filtered_scenes_crop_removes_scenes(self):
        """Test _get_filtered_scenes where cropping removes scenes."""
        small_scenes = np.array([[0, 8], [10, 18]], dtype=np.int32)
        filtered = _get_filtered_scenes(small_scenes, crop_length=5)

        # Both scenes should be removed (8 - 2*5 = -2 < 0)
        assert filtered.shape == (0, 2)

    def test_get_filtered_scenes_invalid_shape(self):
        """Test _get_filtered_scenes with invalid scene shape."""
        invalid_scenes = np.array([0, 30, 60], dtype=np.int32)  # 1D array

        with pytest.raises(ValueError, match="Scenes numpy array needs to be a 2D rank matrix"):
            _get_filtered_scenes(invalid_scenes)

    def test_get_filtered_scenes_invalid_max_length_mode(self):
        """Test _get_filtered_scenes with invalid max_length_mode."""
        with pytest.raises(NotImplementedError, match="Method `invalid` not implemented"):
            _get_filtered_scenes(self.scenes, max_length=50, max_length_mode="invalid")

    def test_get_filtered_scenes_combined_filters(self):
        """Test _get_filtered_scenes with combined filters."""
        filtered = _get_filtered_scenes(
            self.scenes, min_length=20, max_length=50, max_length_mode="truncate", crop_length=5
        )

        # Should apply all filters in sequence
        assert filtered.shape[0] <= 3
        assert filtered.shape[1] == 2


class TestCropScenes:
    """Test cases for _crop_scenes helper function."""

    def test_crop_scenes_basic(self):
        """Test basic functionality of _crop_scenes."""
        scenes = np.array([[0, 30], [30, 90]], dtype=np.int32)

        cropped = _crop_scenes(scenes, crop_length=5)

        assert cropped.shape == (2, 2)
        assert cropped[0, 0] == 5
        assert cropped[0, 1] == 25
        assert cropped[1, 0] == 35
        assert cropped[1, 1] == 85

    def test_crop_scenes_removes_invalid(self):
        """Test _crop_scenes removes scenes with invalid length after cropping."""
        scenes = np.array([[0, 8], [10, 30]], dtype=np.int32)

        cropped = _crop_scenes(scenes, crop_length=5)

        # First scene should be removed (8 - 2*5 = -2 < 0)
        assert cropped.shape == (1, 2)
        assert cropped[0, 0] == 15
        assert cropped[0, 1] == 25

    def test_crop_scenes_all_removed(self):
        """Test _crop_scenes when all scenes are removed."""
        scenes = np.array([[0, 5], [10, 15]], dtype=np.int32)

        cropped = _crop_scenes(scenes, crop_length=5)

        assert cropped.shape == (0, 2)


class TestCreateSpans:
    """Test cases for _create_spans helper function."""

    def test_create_spans_basic(self):
        """Test basic functionality of _create_spans."""
        spans = _create_spans(0, 100, max_length=30, min_length=10)

        assert len(spans) == 4  # 100 / 30 = 3.33, so 4 spans
        assert spans[0] == [0, 30]
        assert spans[1] == [30, 60]
        assert spans[2] == [60, 90]
        assert spans[3] == [90, 100]

    def test_create_spans_exact_fit(self):
        """Test _create_spans with exact fit."""
        spans = _create_spans(0, 90, max_length=30, min_length=10)

        assert len(spans) == 3
        assert spans[0] == [0, 30]
        assert spans[1] == [30, 60]
        assert spans[2] == [60, 90]

    def test_create_spans_drop_last(self):
        """Test _create_spans drops last span if too small."""
        spans = _create_spans(0, 95, max_length=30, min_length=10)

        # Last span would be [90, 95] with length 5, which is < min_length=10
        assert len(spans) == 3
        assert spans[-1] == [60, 90]

    def test_create_spans_no_min_length(self):
        """Test _create_spans with no minimum length."""
        spans = _create_spans(0, 95, max_length=30, min_length=None)

        assert len(spans) == 4
        assert spans[-1] == [90, 95]  # Last span should be kept

    def test_create_spans_single_span(self):
        """Test _create_spans with single span."""
        spans = _create_spans(0, 20, max_length=30, min_length=10)

        assert len(spans) == 1
        assert spans[0] == [0, 20]

    def test_create_spans_empty_range(self):
        """Test _create_spans with empty range."""
        spans = _create_spans(10, 10, max_length=30, min_length=10)

        assert len(spans) == 0


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @patch("nemo_curator.models.transnetv2.TransNetV2")
    def test_complete_pipeline_integration(self, mock_transnetv2_class: Mock):
        """Test complete pipeline integration."""
        # Setup
        stage = TransNetV2ClipExtractionStage(
            threshold=0.5, min_length_s=1.0, max_length_s=5.0, max_length_mode="stride", crop_s=0.2, limit_clips=3
        )

        # Create test video
        frames = rng.integers(0, 255, (150, 27, 48, 3), dtype=np.uint8)
        video = Video(
            input_video=pathlib.Path("test_video.mp4"),
            source_bytes=b"test_data",
            metadata=VideoMetadata(
                height=1080,
                width=1920,
                framerate=30.0,
                num_frames=150,
                duration=5.0,
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                bit_rate_k=5000,
            ),
            clips=[],
            frame_array=frames,
        )

        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        # Mock model
        mock_model = Mock()
        mock_transnetv2_class.return_value = mock_model

        # Setup and process
        stage.setup()

        with patch("nemo_curator.stages.video.clipping.transnetv2_extraction._get_predictions") as mock_get_predictions:
            # Mock predictions to create some transitions
            mock_get_predictions.return_value = np.array([[0], [1], [0], [0], [1], [0]] * 25, dtype=np.uint8)

            result = stage.process(task)

            # Verify results
            assert isinstance(result, VideoTask)
            assert isinstance(result.data, Video)
            assert len(result.data.clips) <= 3  # Should respect limit_clips
            assert result.data.frame_array is None  # Should be cleared

            # Verify clip properties
            for clip in result.data.clips:
                assert isinstance(clip, Clip)
                assert isinstance(clip.uuid, uuid.UUID)
                assert clip.source_video == "test_video.mp4"
                assert isinstance(clip.span, tuple)
                assert len(clip.span) == 2
                assert clip.span[0] < clip.span[1]

    def test_helper_functions_integration(self):
        """Test integration of helper functions."""
        # Create test data
        frames = rng.integers(0, 255, (100, 27, 48, 3), dtype=np.uint8)

        # Mock model for predictions
        mock_model = Mock()

        # Create a side effect that returns different shapes based on batch size
        def model_side_effect(batch_input: torch.Tensor) -> torch.Tensor:
            batch_size = batch_input.shape[1]  # Number of frames in batch
            # Create predictable values for testing
            output_values = [0.3, 0.7, 0.2, 0.8, 0.1] + [0.6] * (batch_size - 5)
            return torch.tensor([output_values[:batch_size]], dtype=torch.float32).reshape(1, batch_size, 1)

        mock_model.side_effect = model_side_effect

        # Mock CUDA to avoid GPU requirements
        def mock_cuda_method(self: torch.Tensor) -> torch.Tensor:
            return self

        with patch("torch.Tensor.cuda", mock_cuda_method):
            # Test the pipeline using actual functions
            predictions = _get_predictions(mock_model, frames, threshold=0.5)

        # Get scenes
        scenes = _get_scenes(predictions, entire_scene_as_clip=True)

        # Filter scenes
        filtered_scenes = _get_filtered_scenes(
            scenes, min_length=10, max_length=50, max_length_mode="stride", crop_length=2
        )

        # Verify the pipeline works
        assert predictions.shape[0] == 100
        assert scenes.shape[1] == 2
        assert filtered_scenes.shape[1] == 2

        # Verify scene properties
        if filtered_scenes.shape[0] > 0:
            for i in range(filtered_scenes.shape[0]):
                assert filtered_scenes[i, 0] < filtered_scenes[i, 1]
