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

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.tasks.video import Clip, Video, VideoTask, _Window


class TestCaptionGenerationStage:
    """Test cases for CaptionGenerationStage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stage = CaptionGenerationStage(
            model_dir="test/models",
            model_variant="qwen",
            caption_batch_size=2,
            fp8=False,
            max_output_tokens=256,
            model_does_preprocess=True,
            disable_mmcache=True,
            verbose=True,
            generate_stage2_caption=False,
        )

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = CaptionGenerationStage()
        assert stage.model_variant == "qwen"
        assert stage.caption_batch_size == 16
        assert stage.fp8 is False
        assert stage.max_output_tokens == 512
        assert stage.model_does_preprocess is False
        assert stage.disable_mmcache is False
        assert stage.verbose is False
        assert stage.generate_stage2_caption is False
        assert stage.stage2_prompt_text is None
        assert stage._name == "caption_generation"

    def test_inputs(self):
        """Test inputs method returns correct format."""
        inputs = self.stage.inputs()
        assert inputs == (["data"], ["clips"])

    def test_outputs(self):
        """Test outputs method returns correct format."""
        outputs = self.stage.outputs()
        assert outputs == (["data"], ["clips"])

    @patch("nemo_curator.stages.video.caption.caption_generation.QwenVL")
    def test_setup_qwen_variant(self, mock_qwen_vl: Mock):
        """Test setup method with qwen variant."""
        mock_model = Mock()
        mock_qwen_vl.return_value = mock_model

        self.stage.setup()

        mock_qwen_vl.assert_called_once_with(
            model_dir="test/models",
            model_variant="qwen",
            caption_batch_size=2,
            fp8=False,
            max_output_tokens=256,
            model_does_preprocess=True,
            disable_mmcache=True,
        )
        mock_model.setup.assert_called_once()
        assert self.stage.model == mock_model

    def test_setup_unsupported_variant(self):
        """Test setup method with unsupported model variant."""
        stage = CaptionGenerationStage(model_variant="unsupported")

        with pytest.raises(ValueError, match="Unsupported model variant: unsupported"):
            stage.setup()

    def test_post_init_resources(self):
        """Test __post_init__ sets correct resources."""
        stage = CaptionGenerationStage()
        assert stage.resources.gpus == 1

    def _create_test_video_task(self) -> VideoTask:
        """Create a test VideoTask with sample data."""
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))

        # Create clips with windows
        clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 10.0), buffer=b"test_buffer_1")

        # Add windows with llm_inputs
        window1 = _Window(
            start_frame=0,
            end_frame=10,
            llm_inputs={"qwen": {"prompt": "test prompt 1", "multi_modal_data": {"video": "test_data_1"}}},
        )
        window2 = _Window(
            start_frame=10,
            end_frame=20,
            llm_inputs={"qwen": {"prompt": "test prompt 2", "multi_modal_data": {"video": "test_data_2"}}},
        )
        clip1.windows = [window1, window2]

        clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(10.0, 20.0), buffer=b"test_buffer_2")

        # Add window with llm_inputs
        window3 = _Window(
            start_frame=0,
            end_frame=15,
            llm_inputs={"qwen": {"prompt": "test prompt 3", "multi_modal_data": {"video": "test_data_3"}}},
        )
        clip2.windows = [window3]

        video.clips = [clip1, clip2]

        return VideoTask(task_id="test", dataset_name="test", data=video)

    def test_process_successful_generation(self):
        """Test successful caption generation process."""
        # Setup mock model
        mock_model = Mock()
        mock_model.generate.return_value = ["Caption 1", "Caption 2", "Caption 3"]
        self.stage.model = mock_model

        task = self._create_test_video_task()

        result = self.stage.process(task)

        # Verify model.generate was called with correct inputs
        mock_model.generate.assert_called_once()
        args, kwargs = mock_model.generate.call_args

        inputs = args[0]
        assert len(inputs) == 3  # 3 windows total
        assert kwargs["generate_stage2_caption"] is False
        assert kwargs["batch_size"] == 2

        # Verify captions were assigned
        assert result.data.clips[0].windows[0].caption["qwen"] == "Caption 1"
        assert result.data.clips[0].windows[1].caption["qwen"] == "Caption 2"
        assert result.data.clips[1].windows[0].caption["qwen"] == "Caption 3"

        # Verify cleanup
        for clip in result.data.clips:
            for window in clip.windows:
                assert "qwen" not in window.llm_inputs
                assert window.mp4_bytes is None

    def test_process_with_verbose_logging(self):
        """Test process method with verbose logging enabled."""
        mock_model = Mock()
        mock_model.generate.return_value = ["Verbose caption"]
        self.stage.model = mock_model
        self.stage.verbose = True

        # Create simple task with one window
        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), buffer=b"test_buffer")
        window = _Window(
            start_frame=0,
            end_frame=5,
            llm_inputs={"qwen": {"prompt": "test", "multi_modal_data": {"video": "data"}}},
        )
        clip.windows = [window]
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        self.stage.process(task)

    @patch("nemo_curator.stages.video.caption.caption_generation.logger")
    def test_process_empty_windows(self, mock_logger: Mock):
        """Test process method with clips that have no windows."""
        mock_model = Mock()
        mock_model.generate.return_value = []
        self.stage.model = mock_model

        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), buffer=b"test_buffer")
        # No windows added
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        self.stage.process(task)

        # Verify warning was logged and error was set
        mock_logger.warning.assert_called_once_with(f"Clip {clip.uuid} has no windows")
        assert clip.errors["windows"] == "empty"

        # Model should not be called if no valid inputs
        mock_model.generate.assert_called_once_with([], generate_stage2_caption=False, batch_size=2)

    @patch("nemo_curator.stages.video.caption.caption_generation.logger")
    def test_process_window_without_input(self, mock_logger: Mock):
        """Test process method with windows that have no llm_inputs."""
        mock_model = Mock()
        mock_model.generate.return_value = []
        self.stage.model = mock_model

        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), buffer=b"test_buffer")
        # Window without llm_inputs
        window = _Window(start_frame=0, end_frame=5)
        clip.windows = [window]
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        self.stage.process(task)

        # Verify error was logged and set
        mock_logger.error.assert_called_once_with(f"Clip {clip.uuid} window 0 has no prepared inputs for qwen.")
        assert clip.errors["window-0"] == "no_qwen_input"

    def test_assign_captions(self):
        """Test _assign_captions method."""
        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 10.0))
        clip1.windows = [_Window(start_frame=0, end_frame=10), _Window(start_frame=10, end_frame=20)]
        clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(10.0, 20.0))
        clip2.windows = [_Window(start_frame=0, end_frame=15)]
        video.clips = [clip1, clip2]

        mapping = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
        captions = [
            (0, "Caption for clip1 window1"),
            (1, "Caption for clip1 window2"),
            (2, "Caption for clip2 window1"),
        ]

        self.stage._assign_captions(video, mapping, captions)

        assert video.clips[0].windows[0].caption["qwen"] == "Caption for clip1 window1"
        assert video.clips[0].windows[1].caption["qwen"] == "Caption for clip1 window2"
        assert video.clips[1].windows[0].caption["qwen"] == "Caption for clip2 window1"

    @patch("nemo_curator.stages.video.caption.caption_generation.logger")
    def test_assign_captions_with_logging(self, mock_logger: Mock):
        """Test _assign_captions method logs summary information."""
        import pathlib

        video = Video(input_video=pathlib.Path("test_video.mp4"))
        video.clip_chunk_index = 0
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0))
        clip.windows = [_Window(start_frame=0, end_frame=5)]
        video.clips = [clip]

        mapping = {0: (0, 0)}
        captions = [(0, "Test caption")]

        # Turn off verbose logging for this test
        original_verbose = self.stage.verbose
        self.stage.verbose = False

        self.stage._assign_captions(video, mapping, captions)

        # Restore original verbose setting
        self.stage.verbose = original_verbose

        mock_logger.info.assert_called_once_with("Generated 1 captions for video test_video.mp4 chunk-0 with 1 clips")

    def test_process_with_stage2_caption(self):
        """Test process method with stage2 caption generation enabled."""
        mock_model = Mock()
        mock_model.generate.return_value = ["Enhanced caption"]
        self.stage.model = mock_model
        self.stage.generate_stage2_caption = True

        # Create simple task with one window
        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), buffer=b"test_buffer")
        window = _Window(
            start_frame=0,
            end_frame=5,
            llm_inputs={"qwen": {"prompt": "test", "multi_modal_data": {"video": "data"}}},
        )
        clip.windows = [window]
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        self.stage.process(task)

        # Verify model.generate was called with stage2 flag
        mock_model.generate.assert_called_once_with(
            [{"prompt": "test", "multi_modal_data": {"video": "data"}}], generate_stage2_caption=True, batch_size=2
        )

    def test_process_returns_same_task(self):
        """Test that process method returns the same task object."""
        mock_model = Mock()
        mock_model.generate.return_value = []
        self.stage.model = mock_model

        task = self._create_test_video_task()
        result = self.stage.process(task)

        assert result is task

    def test_stage_name(self):
        """Test that stage has correct name."""
        assert self.stage._name == "caption_generation"

    def test_stage_with_worker_metadata(self):
        """Test setup method with worker metadata (should be ignored)."""
        mock_model = Mock()
        with patch("nemo_curator.stages.video.caption.caption_generation.QwenVL", return_value=mock_model):
            worker_metadata = WorkerMetadata(worker_id="test")
            self.stage.setup(worker_metadata)
            assert hasattr(self.stage, "model")

    @patch("nemo_curator.stages.video.caption.caption_generation.NemotronHVL")
    def test_setup_nemotron_variant(self, mock_nemotron_vl: Mock):
        """Test setup method with nemotron variant."""
        mock_model = Mock()
        mock_nemotron_vl.return_value = mock_model

        stage = CaptionGenerationStage(
            model_dir="test/models",
            model_variant="nemotron",
            caption_batch_size=2,
            max_output_tokens=256,
            verbose=True,
        )
        stage.setup()

        mock_nemotron_vl.assert_called_once_with(
            model_dir="test/models",
            model_variant="nemotron",
            caption_batch_size=2,
            max_output_tokens=256,
            stage2_prompt_text=None,
            verbose=True,
        )
        mock_model.setup.assert_called_once()
        assert stage.model == mock_model

    def test_process_nemotron_with_generic_llm_inputs(self):
        """Test process method with nemotron using generic llm_inputs."""
        mock_model = Mock()
        mock_model.generate.return_value = ["Nemotron caption 1", "Nemotron caption 2"]

        stage = CaptionGenerationStage(
            model_dir="test/models",
            model_variant="nemotron",
            caption_batch_size=2,
        )
        stage.model = mock_model

        import pathlib

        video = Video(input_video=pathlib.Path("test.mp4"))
        clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 10.0), buffer=b"test_buffer")

        # Create windows with nemotron llm_inputs
        window1 = _Window(
            start_frame=0,
            end_frame=10,
            llm_inputs={"nemotron": {"prompt": "nemotron prompt 1", "multi_modal_data": {"video": "data1"}}},
        )

        window2 = _Window(
            start_frame=10,
            end_frame=20,
            llm_inputs={"nemotron": {"prompt": "nemotron prompt 2", "multi_modal_data": {"video": "data2"}}},
        )

        clip.windows = [window1, window2]
        video.clips = [clip]
        task = VideoTask(task_id="test", dataset_name="test", data=video)

        result = stage.process(task)

        # Verify captions were assigned with nemotron key
        assert result.data.clips[0].windows[0].caption["nemotron"] == "Nemotron caption 1"
        assert result.data.clips[0].windows[1].caption["nemotron"] == "Nemotron caption 2"

        # Verify cleanup - nemotron inputs should be deleted
        assert "nemotron" not in result.data.clips[0].windows[0].llm_inputs
        assert "nemotron" not in result.data.clips[0].windows[1].llm_inputs


# ---------------------------------------------------------------------------
# Integration fixtures (real model weights + GPU required)
# ---------------------------------------------------------------------------


def _make_task(video_bytes: bytes, task_id: str = "integration-test") -> VideoTask:
    """Build a minimal VideoTask with one clip whose buffer is *video_bytes*."""
    clip = Clip(
        uuid=uuid4(),
        source_video=task_id,
        span=(0.0, 3.0),
        buffer=video_bytes,
    )
    video = Video(input_video=Path(task_id + ".mp4"))
    video.clips = [clip]
    return VideoTask(task_id=task_id, dataset_name="integration", data=video)


@pytest.fixture(scope="module")
def preparation_stage() -> CaptionPreparationStage:
    """Instantiate and set up CaptionPreparationStage once per module."""
    from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage

    stage = CaptionPreparationStage(
        model_variant="qwen",
        prompt_variant="default",
        sampling_fps=2.0,
        window_size=256,
        remainder_threshold=4,
        model_does_preprocess=False,
        generate_previews=False,
        verbose=False,
    )
    stage.setup()
    return stage


@pytest.fixture(scope="class")
def generation_stage():
    """Instantiate and set up CaptionGenerationStage once per class."""
    model_dir = os.environ.get("CURATOR_TEST_MODEL_DIR", "")
    stage = CaptionGenerationStage(
        model_dir=model_dir,
        model_variant="qwen",
        caption_batch_size=1,
        fp8=False,
        max_output_tokens=64,
        model_does_preprocess=False,
        disable_mmcache=True,
        vllm_kwargs={"enforce_eager": True},
        verbose=False,
        generate_stage2_caption=False,
    )
    stage.setup()
    yield stage
    # Release GPU memory so a subsequent model can load in the same session
    import gc

    import torch

    del stage.model.model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestQwenCaptionPipelineIntegration:
    """End-to-end integration tests for the Qwen caption pipeline.

    A single autouse fixture runs the full prep→generation pipeline once;
    individual tests assert on the stored snapshots.
    """

    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(
        self,
        request: pytest.FixtureRequest,
        preparation_stage: CaptionPreparationStage,
        generation_stage: CaptionGenerationStage,
        video_fixture_path: Path,
    ) -> None:
        """Run prep→generation once and store state on the class for all tests."""
        video_bytes = video_fixture_path.read_bytes()
        task = _make_task(video_bytes, task_id="pipeline-test")

        # --- preparation stage ---
        task = preparation_stage.process(task)
        clip = task.data.clips[0]
        request.cls.prep_clip = clip
        # Capture raw vLLM inputs before generation clears them
        request.cls.raw_inputs = [w.llm_inputs["qwen"] for w in clip.windows if "qwen" in w.llm_inputs]

        # --- generation stage ---
        task = generation_stage.process(task)
        request.cls.gen_clip = task.data.clips[0]

    def test_preparation_stage_populates_windows(self) -> None:
        """CaptionPreparationStage must produce at least one window with a
        non-None qwen_llm_input dict that vLLM can consume."""
        assert len(self.prep_clip.errors) == 0, f"Preparation stage set clip errors: {self.prep_clip.errors}"
        assert len(self.raw_inputs) > 0, "No windows with vLLM inputs were created"

        for i, llm_input in enumerate(self.raw_inputs):
            assert "prompt" in llm_input, f"Window {i}: missing 'prompt' key"
            assert "multi_modal_data" in llm_input, f"Window {i}: missing 'multi_modal_data' key"
            assert isinstance(llm_input["prompt"], str), f"Window {i}: 'prompt' is not a str"
            assert len(llm_input["prompt"]) > 0, f"Window {i}: 'prompt' is empty"
            video_tensor = llm_input["multi_modal_data"].get("video")
            assert video_tensor is not None, f"Window {i}: 'multi_modal_data.video' is None"

    def test_generation_stage_returns_captions(self) -> None:
        """Full Qwen caption pipeline must produce a non-empty string caption
        for every window, with no unhandled exceptions."""
        clip = self.gen_clip
        assert len(clip.errors) == 0, f"Generation stage set clip errors: {clip.errors}"
        assert len(clip.windows) == len(self.raw_inputs), "Window count changed after generation"

        for i, window in enumerate(clip.windows):
            caption = window.caption.get("qwen")
            assert caption is not None, f"Window {i}: caption key 'qwen' not set"
            assert isinstance(caption, str), f"Window {i}: caption is not a str"
            assert len(caption.strip()) > 0, f"Window {i}: caption is blank"

        for i, window in enumerate(clip.windows):
            assert "qwen" not in window.llm_inputs, f"Window {i}: qwen_llm_input not cleared after generation"
