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

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.video.caption.caption_enhancement import (
    _ENHANCE_PROMPTS,
    CaptionEnhancementStage,
    _get_enhance_prompt,
)
from nemo_curator.tasks.video import Clip, Video, VideoTask, _Window


class TestCaptionEnhancementStage:
    """Test cases for CaptionEnhancementStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = CaptionEnhancementStage(
            model_dir="test/models",
            model_variant="qwen",
            prompt_variant="default",
            model_batch_size=2,
            fp8=False,
            max_output_tokens=256,
            verbose=True,
        )

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = CaptionEnhancementStage()
        assert stage.model_dir == "models/qwen"
        assert stage.model_variant == "qwen"
        assert stage.prompt_variant == "default"
        assert stage.prompt_text is None
        assert stage.model_batch_size == 128
        assert stage.fp8 is False
        assert stage.max_output_tokens == 512
        assert stage.verbose is False
        assert stage._name == "caption_enhancement"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        custom_prompt = "Custom enhancement prompt"
        stage = CaptionEnhancementStage(
            model_dir="custom/models",
            model_variant="qwen",
            prompt_variant="av-surveillance",
            prompt_text=custom_prompt,
            model_batch_size=64,
            fp8=True,
            max_output_tokens=1024,
            verbose=True,
        )
        assert stage.model_dir == "custom/models"
        assert stage.model_variant == "qwen"
        assert stage.prompt_variant == "av-surveillance"
        assert stage.prompt_text == custom_prompt
        assert stage.model_batch_size == 64
        assert stage.fp8 is True
        assert stage.max_output_tokens == 1024
        assert stage.verbose is True

    def test_inputs(self):
        """Test inputs method returns correct format."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], ["clips"])

    def test_outputs(self):
        """Test outputs method returns correct format."""
        outputs = self.stage.outputs()
        assert outputs == (["data"], ["clips"])

    def test_post_init_resources(self):
        """Test __post_init__ sets correct resources."""
        stage = CaptionEnhancementStage()
        assert stage.resources.gpus == 1

    def test_post_init_prompt_setup(self):
        """Test __post_init__ sets up prompt correctly."""
        with patch("nemo_curator.stages.video.caption.caption_enhancement._get_enhance_prompt") as mock_get_prompt:
            mock_get_prompt.return_value = "test prompt"
            stage = CaptionEnhancementStage(prompt_variant="default", prompt_text=None)
            mock_get_prompt.assert_called_once_with("default", None)
            assert stage.prompt == "test prompt"

    @patch("nemo_curator.stages.video.caption.caption_enhancement.QwenLM")
    def test_setup_qwen_variant(self, mock_qwen_lm: Mock):
        """Test setup method with qwen variant."""
        mock_model = Mock()
        mock_qwen_lm.return_value = mock_model

        self.stage.setup()

        mock_qwen_lm.assert_called_once_with(
            model_dir="test/models",
            caption_batch_size=2,
            fp8=False,
            max_output_tokens=256,
        )
        mock_model.setup.assert_called_once()
        assert self.stage.model == mock_model

    def test_setup_unsupported_variant(self):
        """Test setup method with unsupported model variant."""
        stage = CaptionEnhancementStage(model_variant="unsupported")

        with pytest.raises(ValueError, match="Unsupported model variant: unsupported"):
            stage.setup()

    def test_setup_with_worker_metadata(self):
        """Test setup method with worker metadata (should be ignored)."""
        mock_model = Mock()
        with patch("nemo_curator.stages.video.caption.caption_enhancement.QwenLM", return_value=mock_model):
            worker_metadata = WorkerMetadata(worker_id="test")
            self.stage.setup(worker_metadata)
            assert hasattr(self.stage, "model")

    def _create_test_video_task_with_captions(self) -> VideoTask:
        """Create a test VideoTask with sample data including captions."""
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))

        # Create clips with windows that have captions
        clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 10.0), buffer=b"test_buffer_1")

        # Add windows with existing captions
        window1 = _Window(start_frame=0, end_frame=5)
        window1.caption = {"qwen": "A person walks down the street"}
        window1.enhanced_caption = {}

        window2 = _Window(start_frame=5, end_frame=10)
        window2.caption = {"qwen": "The person enters a building"}
        window2.enhanced_caption = {}

        clip1.windows = [window1, window2]

        clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(10.0, 20.0), buffer=b"test_buffer_2")

        # Add window with caption
        window3 = _Window(start_frame=10, end_frame=15)
        window3.caption = {"qwen": "Cars drive by on the road"}
        window3.enhanced_caption = {}
        clip2.windows = [window3]

        video.clips = [clip1, clip2]
        return VideoTask(task_id="test", dataset_name="test", data=video)

    def _create_test_video_task_empty_clips(self) -> VideoTask:
        """Create a test VideoTask with empty clips."""
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))

        # Create clip with no windows
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 10.0), buffer=b"test_buffer")
        clip.windows = []
        video.clips = [clip]
        return VideoTask(task_id="test", dataset_name="test", data=video)

    def _create_test_video_task_no_captions(self) -> VideoTask:
        """Create a test VideoTask with windows but no captions."""
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))

        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 10.0), buffer=b"test_buffer")

        # Add window without caption
        window = _Window(start_frame=0, end_frame=5)
        window.caption = None
        window.enhanced_caption = {}
        clip.windows = [window]

        video.clips = [clip]
        return VideoTask(task_id="test", dataset_name="test", data=video)

    @patch("nemo_curator.stages.video.caption.caption_enhancement.logger")
    def test_process_with_valid_captions(self, _mock_logger: Mock):  # noqa: PT019
        """Test process method with valid captions."""
        mock_model = Mock()
        # Since batch_size=2, with 3 inputs we get 2 batches: [0,1] and [2]
        mock_model.generate.side_effect = [
            [
                "Enhanced: A person walks confidently down a busy city street with tall buildings",
                "Enhanced: The person enters a modern glass building through the main entrance",
            ],  # First batch
            ["Enhanced: Multiple cars and vehicles drive by on the busy urban road with traffic"],  # Second batch
        ]
        self.stage.model = mock_model
        self.stage.prompt = "Enhance this caption:"

        task = self._create_test_video_task_with_captions()
        result = self.stage.process(task)

        # Check that generate was called twice due to batching (batch_size=2, 3 inputs)
        assert mock_model.generate.call_count == 2

        # Verify the batched calls
        first_batch_call = mock_model.generate.call_args_list[0][0][0]
        second_batch_call = mock_model.generate.call_args_list[1][0][0]

        # First batch should have 2 items
        assert len(first_batch_call) == 2
        # Second batch should have 1 item
        assert len(second_batch_call) == 1

        # Verify enhanced captions were set
        assert "qwen_lm" in task.data.clips[0].windows[0].enhanced_caption
        assert "qwen_lm" in task.data.clips[0].windows[1].enhanced_caption
        assert "qwen_lm" in task.data.clips[1].windows[0].enhanced_caption

        # Verify the result is the same task
        assert result is task

    @patch("nemo_curator.stages.video.caption.caption_enhancement.logger")
    def test_process_with_empty_clips(self, mock_logger: Mock):
        """Test process method with clips that have no windows."""
        mock_model = Mock()
        mock_model.generate.return_value = []
        self.stage.model = mock_model

        task = self._create_test_video_task_empty_clips()
        result = self.stage.process(task)

        # Verify warning was logged for empty windows
        mock_logger.warning.assert_called_once()
        assert "has no windows" in str(mock_logger.warning.call_args)

        # Verify error was set on clip
        assert "windows" in task.data.clips[0].errors
        assert task.data.clips[0].errors["windows"] == "empty"

        # Verify model.generate was not called (no valid inputs)
        mock_model.generate.assert_not_called()

        # Verify the result is the same task
        assert result is task

    @patch("nemo_curator.stages.video.caption.caption_enhancement.logger")
    def test_process_with_no_captions(self, mock_logger: Mock):
        """Test process method with windows that have no captions."""
        mock_model = Mock()
        mock_model.generate.return_value = []
        self.stage.model = mock_model

        task = self._create_test_video_task_no_captions()
        result = self.stage.process(task)

        # Verify error was logged for missing caption
        mock_logger.error.assert_called()
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        assert any("has no caption" in call for call in error_calls)

        # Verify error was set on clip
        assert "window-0" in task.data.clips[0].errors
        assert task.data.clips[0].errors["window-0"] == "empty"

        # Verify model.generate was not called (no valid inputs)
        mock_model.generate.assert_not_called()

        # Verify the result is the same task
        assert result is task

    def test_process_batch_processing(self):
        """Test process method handles batching correctly."""
        mock_model = Mock()
        # Create a large number of captions to test batching
        mock_model.generate.side_effect = [
            ["Enhanced caption 1", "Enhanced caption 2"],  # First batch
            ["Enhanced caption 3"],  # Second batch
        ]
        self.stage.model = mock_model
        self.stage.model_batch_size = 2  # Small batch size to force multiple batches
        self.stage.prompt = "Enhance:"

        # Create task with 3 windows (will require 2 batches)
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 30.0), buffer=b"test")

        for i in range(3):
            window = _Window(start_frame=i * 10, end_frame=(i + 1) * 10)
            window.caption = {"qwen": f"Caption {i + 1}"}
            window.enhanced_caption = {}
            clip.windows.append(window)

        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        self.stage.process(task)

        # Verify model.generate was called twice (batching)
        assert mock_model.generate.call_count == 2

        # Verify all enhanced captions were set
        for i, window in enumerate(task.data.clips[0].windows):
            assert "qwen_lm" in window.enhanced_caption
            assert window.enhanced_caption["qwen_lm"] == f"Enhanced caption {i + 1}"

    def test_process_returns_same_task(self):
        """Test that process method returns the same task object."""
        mock_model = Mock()
        # Since batch_size=2, with 3 inputs we get 2 batches: [0,1] and [2]
        mock_model.generate.side_effect = [
            ["Enhanced caption 1", "Enhanced caption 2"],  # First batch
            ["Enhanced caption 3"],  # Second batch
        ]
        self.stage.model = mock_model

        task = self._create_test_video_task_with_captions()
        result = self.stage.process(task)

        assert result is task

    @patch("nemo_curator.stages.video.caption.caption_enhancement.logger")
    def test_process_with_verbose_logging(self, mock_logger: Mock):
        """Test process method with verbose logging enabled."""
        mock_model = Mock()
        mock_model.generate.return_value = ["Enhanced caption"]
        self.stage.model = mock_model
        self.stage.verbose = True
        self.stage.prompt = "Enhance:"

        # Create simple task with one caption
        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 10.0), buffer=b"test")
        window = _Window(start_frame=0, end_frame=5)
        window.caption = {"qwen": "Original caption"}
        window.enhanced_caption = {}
        clip.windows = [window]
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        self.stage.process(task)

        # Verify verbose logging occurred
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Caption for clip" in call for call in info_calls)
        assert any("Enhanced QwenLM Caption" in call for call in info_calls)


class TestGetEnhancePrompt:
    """Test cases for _get_enhance_prompt function."""

    def test_get_enhance_prompt_with_prompt_text(self):
        """Test _get_enhance_prompt with custom prompt text."""
        custom_prompt = "Custom enhancement prompt"
        result = _get_enhance_prompt("default", custom_prompt)
        assert result == custom_prompt

    def test_get_enhance_prompt_default_variant(self):
        """Test _get_enhance_prompt with default variant."""
        result = _get_enhance_prompt("default", None)
        assert result == _ENHANCE_PROMPTS["default"]

    def test_get_enhance_prompt_av_surveillance_variant(self):
        """Test _get_enhance_prompt with av-surveillance variant."""
        result = _get_enhance_prompt("av-surveillance", None)
        assert result == _ENHANCE_PROMPTS["av-surveillance"]

    def test_get_enhance_prompt_invalid_variant(self):
        """Test _get_enhance_prompt with invalid variant."""
        with pytest.raises(ValueError, match="Invalid prompt variant: invalid"):
            _get_enhance_prompt("invalid", None)

    @patch("nemo_curator.stages.video.caption.caption_enhancement.logger")
    def test_get_enhance_prompt_with_verbose(self, mock_logger: Mock):
        """Test _get_enhance_prompt with verbose logging."""
        result = _get_enhance_prompt("default", None, verbose=True)

        # Verify debug logging occurred
        mock_logger.debug.assert_called_once()
        debug_call = str(mock_logger.debug.call_args)
        assert "Enhance Captioning prompt" in debug_call
        assert result == _ENHANCE_PROMPTS["default"]

    def test_enhance_prompts_constants(self):
        """Test that enhancement prompt constants are properly defined."""
        assert "default" in _ENHANCE_PROMPTS
        assert "av-surveillance" in _ENHANCE_PROMPTS

        default_prompt = _ENHANCE_PROMPTS["default"]
        av_prompt = _ENHANCE_PROMPTS["av-surveillance"]

        assert "enhances video caption inputs" in default_prompt
        assert "longer than the provided input" in default_prompt
        assert "surveillance cameras" in av_prompt
        assert "longer than the provided input" in av_prompt
