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
import torch

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.video.caption.caption_preparation import (
    _ENHANCE_PROMPTS,
    _PROMPTS,
    CaptionPreparationStage,
    _get_prompt,
)
from nemo_curator.tasks.video import Clip, Video, VideoTask
from nemo_curator.utils.windowing_utils import WindowFrameInfo


class TestGetPromptFunction:
    """Test cases for the _get_prompt helper function."""

    def test_get_prompt_with_custom_text(self):
        """Test _get_prompt with custom prompt text."""
        custom_prompt = "Custom prompt text"
        result = _get_prompt("default", custom_prompt)
        assert result == custom_prompt

    def test_get_prompt_default_variant(self):
        """Test _get_prompt with default variant."""
        result = _get_prompt("default", None)
        expected = _PROMPTS["default"]
        assert result == expected

    def test_get_prompt_av_variant(self):
        """Test _get_prompt with av variant."""
        result = _get_prompt("av", None)
        expected = _PROMPTS["av"]
        assert result == expected

    def test_get_prompt_av_surveillance_variant(self):
        """Test _get_prompt with av-surveillance variant."""
        result = _get_prompt("av-surveillance", None)
        expected = _PROMPTS["av-surveillance"]
        assert result == expected

    def test_get_prompt_invalid_variant(self):
        """Test _get_prompt with invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Invalid prompt variant: invalid"):
            _get_prompt("invalid", None)

    def test_get_prompt_custom_text_overrides_variant(self):
        """Test that custom text takes precedence over variant."""
        custom_prompt = "Override prompt"
        result = _get_prompt("av", custom_prompt)
        assert result == custom_prompt
        assert result != _PROMPTS["av"]


class TestCaptionPreparationStage:
    """Test cases for CaptionPreparationStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = CaptionPreparationStage(
            model_variant="qwen",
            prompt_variant="default",
            prompt_text="Custom test prompt",
            verbose=True,
            sampling_fps=1.0,
            window_size=128,
            remainder_threshold=64,
            model_does_preprocess=True,
            preprocess_dtype="float16",
            generate_previews=False,
        )

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = CaptionPreparationStage()
        assert stage.model_variant == "qwen"
        assert stage.prompt_variant == "default"
        assert stage.prompt_text is None
        assert stage.verbose is False
        assert stage.sampling_fps == 2.0
        assert stage.window_size == 256
        assert stage.remainder_threshold == 128
        assert stage.model_does_preprocess is False
        assert stage.preprocess_dtype == "float32"
        assert stage.generate_previews is True
        assert stage._name == "caption_preparation"

    def test_inputs(self):
        """Test inputs method returns correct format."""
        inputs = self.stage.inputs()
        assert inputs == ([], [])

    def test_outputs(self):
        """Test outputs method returns correct format."""
        outputs = self.stage.outputs()
        assert outputs == ([], [])

    @patch("nemo_curator.stages.video.caption.caption_preparation.PromptFormatter")
    def test_setup(self, mock_prompt_formatter: Mock):
        """Test setup method initializes PromptFormatter."""
        mock_formatter = Mock()
        mock_prompt_formatter.return_value = mock_formatter

        self.stage.setup()

        mock_prompt_formatter.assert_called_once_with("qwen")
        assert self.stage.prompt_formatter == mock_formatter

    def test_setup_with_worker_metadata(self):
        """Test setup method with worker metadata (should be ignored)."""
        with patch("nemo_curator.stages.video.caption.caption_preparation.PromptFormatter") as mock_formatter:
            worker_metadata = WorkerMetadata(worker_id="test")
            self.stage.setup(worker_metadata)
            mock_formatter.assert_called_once_with("qwen")

    def _create_test_video_task(self) -> VideoTask:
        """Create a test VideoTask with sample data."""
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))

        clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 10.0), buffer=b"test_video_buffer_1")
        # Mock attributes for original code bugs/quirks
        clip1.id = clip1.uuid

        clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(10.0, 20.0), buffer=b"test_video_buffer_2")
        # Mock attributes for original code bugs/quirks
        clip2.id = clip2.uuid

        video.clips = [clip1, clip2]

        return VideoTask(task_id="test", dataset_name="test", data=video)

    @patch("nemo_curator.stages.video.caption.caption_preparation.windowing_utils.split_video_into_windows")
    @patch("nemo_curator.stages.video.caption.caption_preparation._get_prompt")
    def test_process_successful(self, mock_get_prompt: Mock, mock_split_video: Mock):
        """Test successful caption preparation process."""
        # Setup mock formatter
        mock_formatter = Mock()
        mock_formatter.generate_inputs.return_value = {
            "prompt": "test formatted prompt",
            "multi_modal_data": {"video": torch.randn(10, 3, 224, 224)},
        }
        self.stage.prompt_formatter = mock_formatter

        # Setup mock prompt
        test_prompt = "Test prompt for preparation"
        mock_get_prompt.return_value = test_prompt

        # Setup mock windowing - return 2 windows for each clip
        window_info_1 = WindowFrameInfo(start=0, end=10)
        window_info_2 = WindowFrameInfo(start=10, end=20)

        mock_split_video.side_effect = [
            # First clip returns
            (
                [b"window1_bytes", b"window2_bytes"],
                [torch.randn(5, 3, 224, 224), torch.randn(5, 3, 224, 224)],
                [window_info_1, window_info_2],
            ),
            # Second clip returns
            ([b"window3_bytes"], [torch.randn(8, 3, 224, 224)], [WindowFrameInfo(start=0, end=8)]),
        ]

        # Mock resources
        with patch.object(self.stage, "_resources") as mock_resources:
            mock_resources.cpus = 4

            task = self._create_test_video_task()
            result = self.stage.process(task)

            # Verify windowing was called for each clip
            assert mock_split_video.call_count == 2

            # Verify calls for first clip
            first_call = mock_split_video.call_args_list[0]
            assert first_call[0][0] == b"test_video_buffer_1"  # buffer
            assert first_call[1]["window_size"] == 128
            assert first_call[1]["remainder_threshold"] == 64
            assert first_call[1]["sampling_fps"] == 1.0
            assert first_call[1]["model_does_preprocess"] is True
            assert first_call[1]["preprocess_dtype"] == "float16"
            assert first_call[1]["return_bytes"] is False
            assert first_call[1]["num_threads"] == 4

            # Verify prompt generation was called
            assert mock_get_prompt.call_count == 3  # 2 windows from clip1 + 1 window from clip2
            mock_get_prompt.assert_called_with("default", "Custom test prompt")

            # Verify formatter was called for each window
            assert mock_formatter.generate_inputs.call_count == 3

            # Verify windows were created
            assert len(result.data.clips[0].windows) == 2
            assert len(result.data.clips[1].windows) == 1

            # Check first window
            window1 = result.data.clips[0].windows[0]
            assert window1.start_frame == 0
            assert window1.end_frame == 10
            assert window1.mp4_bytes == b"window1_bytes"
            # Check structure of qwen_llm_input (can't compare tensors directly)
            assert "prompt" in window1.qwen_llm_input
            assert "multi_modal_data" in window1.qwen_llm_input
            assert window1.qwen_llm_input["prompt"] == "test formatted prompt"
            assert "video" in window1.qwen_llm_input["multi_modal_data"]

    @patch("nemo_curator.stages.video.caption.caption_preparation.logger")
    def test_process_clip_without_buffer(self, mock_logger: Mock):
        """Test process method with clip that has no buffer."""
        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(
            uuid=uuid4(),
            source_video="test.mp4",
            span=(0.0, 5.0),
            buffer=None,  # No buffer
        )
        # Mock the id attribute since original code uses clip.id but Clip only has uuid
        clip.id = clip.uuid
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        # Setup formatter
        self.stage.prompt_formatter = Mock()

        self.stage.process(task)

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(f"No buffer found for clip {clip.id}")
        assert clip.errors["buffer"] == "empty"

        # Verify no windows were created
        assert len(clip.windows) == 0

    @patch("nemo_curator.stages.video.caption.caption_preparation.windowing_utils.split_video_into_windows")
    @patch("nemo_curator.stages.video.caption.caption_preparation._get_prompt")
    @patch("nemo_curator.stages.video.caption.caption_preparation.logger")
    def test_process_formatter_error(self, mock_logger: Mock, mock_get_prompt: Mock, mock_split_video: Mock):
        """Test process method when formatter raises exception."""
        # Setup mock formatter that raises exception
        mock_formatter = Mock()
        mock_formatter.generate_inputs.side_effect = ValueError("Formatter error")
        self.stage.prompt_formatter = mock_formatter

        # Setup mock windowing
        mock_get_prompt.return_value = "test prompt"
        mock_split_video.return_value = (
            [b"window_bytes"],
            [torch.randn(5, 3, 224, 224)],
            [WindowFrameInfo(start=0, end=5)],
        )

        # Mock resources
        with patch.object(self.stage, "_resources") as mock_resources:
            mock_resources.cpus = 2

            task = self._create_test_video_task()
            result = self.stage.process(task)

            # Verify error was logged (called for each clip that fails)
            assert mock_logger.error.call_count == 2
            mock_logger.error.assert_called_with("Error in Caption preparation: Formatter error")

            # Verify error was set on clip
            assert result.data.clips[0].errors["qwen_input"] == "Formatter error"

            # Verify no windows were created for the failed clip
            assert len(result.data.clips[0].windows) == 0

    @patch("nemo_curator.stages.video.caption.caption_preparation.windowing_utils.split_video_into_windows")
    @patch("nemo_curator.stages.video.caption.caption_preparation._get_prompt")
    def test_process_with_generate_previews_enabled(self, mock_get_prompt: Mock, mock_split_video: Mock):
        """Test process method with generate_previews enabled."""
        # Enable previews
        self.stage.generate_previews = True

        # Setup mocks
        mock_formatter = Mock()
        mock_formatter.generate_inputs.return_value = {"prompt": "test", "multi_modal_data": {"video": None}}
        self.stage.prompt_formatter = mock_formatter

        mock_get_prompt.return_value = "test prompt"
        mock_split_video.return_value = (
            [b"preview_bytes"],
            [torch.randn(3, 3, 224, 224)],
            [WindowFrameInfo(start=0, end=3)],
        )

        # Mock resources
        with patch.object(self.stage, "_resources") as mock_resources:
            mock_resources.cpus = 1

            task = self._create_test_video_task()
            self.stage.process(task)

            # Verify windowing was called with return_bytes=True
            call_args = mock_split_video.call_args_list[0]
            assert call_args[1]["return_bytes"] is True

    @patch("nemo_curator.stages.video.caption.caption_preparation.windowing_utils.split_video_into_windows")
    @patch("nemo_curator.stages.video.caption.caption_preparation._get_prompt")
    def test_process_multiple_windows_different_frame_ranges(self, mock_get_prompt: Mock, mock_split_video: Mock):
        """Test process method creates windows with correct frame ranges."""
        # Setup mocks
        mock_formatter = Mock()
        mock_formatter.generate_inputs.return_value = {"prompt": "test", "multi_modal_data": {"video": None}}
        self.stage.prompt_formatter = mock_formatter

        mock_get_prompt.return_value = "test prompt"

        # Return multiple windows with different frame ranges
        mock_split_video.return_value = (
            [b"window1", b"window2", b"window3"],
            [torch.randn(5, 3, 224, 224), torch.randn(8, 3, 224, 224), torch.randn(3, 3, 224, 224)],
            [WindowFrameInfo(start=0, end=5), WindowFrameInfo(start=5, end=13), WindowFrameInfo(start=13, end=16)],
        )

        # Mock resources
        with patch.object(self.stage, "_resources") as mock_resources:
            mock_resources.cpus = 2

            # Create task with single clip
            import pathlib

            video = Video(input_video=pathlib.Path("test.mp4"))
            clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 20.0), buffer=b"test_buffer")
            # Mock attributes for original code bugs/quirks
            clip.id = clip.uuid
            video.clips = [clip]
            task = VideoTask(task_id="test", dataset_name="test", data=video)

            result = self.stage.process(task)

            # Verify 3 windows were created with correct frame ranges
            assert len(result.data.clips[0].windows) == 3

            windows = result.data.clips[0].windows
            assert windows[0].start_frame == 0
            assert windows[0].end_frame == 5
            assert windows[1].start_frame == 5
            assert windows[1].end_frame == 13
            assert windows[2].start_frame == 13
            assert windows[2].end_frame == 16

    def test_process_returns_same_task(self):
        """Test that process method returns the same task object."""
        # Setup minimal mocks
        self.stage.prompt_formatter = Mock()

        task = self._create_test_video_task()
        # Set buffers to None to avoid processing and mock id attribute
        for clip in task.data.clips:
            clip.buffer = None
            clip.id = clip.uuid  # Mock id attribute

        result = self.stage.process(task)

        assert result is task

    def test_stage_name(self):
        """Test that stage has correct name."""
        assert self.stage._name == "caption_preparation"

    def test_prompts_constants(self):
        """Test that prompt constants are defined correctly."""
        assert "default" in _PROMPTS
        assert "av" in _PROMPTS
        assert "av-surveillance" in _PROMPTS

        assert "default" in _ENHANCE_PROMPTS
        assert "av-surveillance" in _ENHANCE_PROMPTS

        # Verify they contain expected content
        assert "Elaborate on the visual and narrative elements" in _PROMPTS["default"]
        assert "camera mounted on a car" in _PROMPTS["av"]
        assert "surveillance camera" in _PROMPTS["av-surveillance"]

    @patch("nemo_curator.stages.video.caption.caption_preparation.windowing_utils.split_video_into_windows")
    def test_process_empty_windowing_result(self, mock_split_video: Mock):
        """Test process method when windowing returns empty results."""
        # Setup formatter
        mock_formatter = Mock()
        self.stage.prompt_formatter = mock_formatter

        # Mock windowing to return empty results
        mock_split_video.return_value = ([], [], [])

        # Mock resources
        with patch.object(self.stage, "_resources") as mock_resources:
            mock_resources.cpus = 1

            task = self._create_test_video_task()
            result = self.stage.process(task)

            # Verify no windows were created
            for clip in result.data.clips:
                assert len(clip.windows) == 0

            # Verify formatter was not called
            mock_formatter.generate_inputs.assert_not_called()
