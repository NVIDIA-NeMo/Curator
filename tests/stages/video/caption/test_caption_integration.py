# modality: video

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Integration tests for the video caption pipeline (vLLM + HF transformers).

These tests exercise the real QwenVL model stack — no mocks — to catch
API-level breakage between Curator and its external dependencies (vLLM,
HF transformers).  They require a GPU and the Qwen2.5-VL-7B model weights.

Run with::

    pytest -m integration tests/stages/video/caption/test_caption_integration.py

Model weights root is required for generation tests. Either pass it via CLI::

    pytest -m integration --model-dir /path/to/models tests/stages/video/caption/test_caption_integration.py

Or set the environment variable (same directory must contain ``Qwen/Qwen2.5-VL-7B-Instruct/``)::

    QWEN_MODEL_DIR=/path/to/models pytest -m integration tests/stages/video/caption/test_caption_integration.py

If neither is set, tests that need the model are skipped.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from nemo_curator.tasks.video import Clip, Video, VideoTask, _Window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(video_bytes: bytes, task_id: str = "integration-test") -> VideoTask:
    """Build a minimal VideoTask with one clip whose buffer is *video_bytes*.

    We bypass VideoReaderStage / FixedStrideExtractorStage / ClipTranscodingStage
    intentionally: the integration boundary under test starts at
    CaptionPreparationStage (HF AutoProcessor) and ends at
    CaptionGenerationStage (vLLM).
    """
    clip = Clip(
        uuid=uuid4(),
        source_video=task_id,
        span=(0.0, 3.0),
        buffer=video_bytes,
    )
    video = Video(input_video=Path(task_id + ".mp4"))
    video.clips = [clip]
    return VideoTask(task_id=task_id, dataset_name="integration", data=video)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def preparation_stage() -> CaptionPreparationStage:
    """Instantiate and set up CaptionPreparationStage once per module.

    setup() loads the HF AutoProcessor — the first real external-library call.
    """
    stage = CaptionPreparationStage(
        model_variant="qwen",
        prompt_variant="default",
        sampling_fps=2.0,
        window_size=256,
        remainder_threshold=4,  # low threshold so the 30-frame fixture creates windows
        model_does_preprocess=False,
        generate_previews=False,  # skip ffmpeg window-splitting; not under test
        verbose=False,
    )
    stage.setup()
    return stage


@pytest.fixture(scope="class")
def generation_stage(qwen_model_dir: str):
    """Instantiate and set up CaptionGenerationStage once per module.

    setup() loads vLLM's LLM() — the main integration boundary.
    """
    from nemo_curator.models.qwen_vl import _QWEN_VARIANTS_INFO

    weight_path = Path(qwen_model_dir) / _QWEN_VARIANTS_INFO["qwen"]
    if not weight_path.exists():
        pytest.skip(
            f"Qwen weights not found at {weight_path}. "
            f"Pass --model-dir or set $QWEN_MODEL_DIR to a directory "
            f"containing Qwen/Qwen2.5-VL-7B-Instruct/."
        )

    stage = CaptionGenerationStage(
        model_dir=qwen_model_dir,
        model_variant="qwen",
        caption_batch_size=1,
        fp8=False,
        max_output_tokens=64,  # short output keeps the test fast
        model_does_preprocess=False,
        disable_mmcache=True,
        vllm_kwargs={"enforce_eager": True},  # skip CUDA graph capture — ~10s load vs 30+ min
        verbose=False,
        generate_stage2_caption=False,
    )
    stage.setup()
    yield stage
    # Release GPU memory so the 14B enhancement model can load in the same session
    import gc

    import torch

    del stage.model.model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
class TestQwenCaptionPipelineIntegration:
    """End-to-end integration tests for the Qwen caption pipeline.

    Each test asserts *pipeline integrity*, not caption content:
    - No unhandled exceptions
    - Data flows from one stage to the next without loss
    - Output fields are populated and have the expected types / shapes
    """

    def test_preparation_stage_populates_windows(
        self,
        preparation_stage: CaptionPreparationStage,
        video_fixture_path: Path,
    ) -> None:
        """CaptionPreparationStage must produce at least one window with a
        non-None qwen_llm_input dict that vLLM can consume."""
        video_bytes = video_fixture_path.read_bytes()
        task = _make_task(video_bytes, task_id="prep-stage-test")

        result = preparation_stage.process(task)

        clip = result.data.clips[0]
        assert len(clip.errors) == 0, f"Preparation stage set clip errors: {clip.errors}"
        assert len(clip.windows) > 0, "No windows were created — check windowing_utils or clip.buffer"

        for i, window in enumerate(clip.windows):
            assert window.qwen_llm_input is not None, f"Window {i} has no qwen_llm_input"
            llm_input = window.qwen_llm_input

            # Structural contract: vLLM expects exactly these two top-level keys
            assert "prompt" in llm_input, f"Window {i}: missing 'prompt' key"
            assert "multi_modal_data" in llm_input, f"Window {i}: missing 'multi_modal_data' key"

            # Prompt must be a non-empty string (chat-template applied)
            assert isinstance(llm_input["prompt"], str), f"Window {i}: 'prompt' is not a str"
            assert len(llm_input["prompt"]) > 0, f"Window {i}: 'prompt' is empty"

            # Video tensor must be present
            video_tensor = llm_input["multi_modal_data"].get("video")
            assert video_tensor is not None, f"Window {i}: 'multi_modal_data.video' is None"

    def test_generation_stage_returns_captions(
        self,
        preparation_stage: CaptionPreparationStage,
        generation_stage: CaptionGenerationStage,
        video_fixture_path: Path,
    ) -> None:
        """Full Qwen caption pipeline must produce a non-empty string caption
        for every window, with no unhandled exceptions."""
        video_bytes = video_fixture_path.read_bytes()
        task = _make_task(video_bytes, task_id="gen-stage-test")

        # Stage 1: prepare inputs (HF AutoProcessor)
        task = preparation_stage.process(task)

        clip = task.data.clips[0]
        assert len(clip.windows) > 0, "Preparation stage produced no windows; cannot test generation"
        expected_windows = len(clip.windows)

        # Stage 2: generate captions (vLLM)
        result = generation_stage.process(task)

        clip = result.data.clips[0]
        assert len(clip.errors) == 0, f"Generation stage set clip errors: {clip.errors}"
        assert len(clip.windows) == expected_windows, "Window count changed after generation"

        for i, window in enumerate(clip.windows):
            caption = window.caption.get("qwen")
            assert caption is not None, f"Window {i}: caption key 'qwen' not set"
            assert isinstance(caption, str), f"Window {i}: caption is not a str (got {type(caption)})"
            assert len(caption.strip()) > 0, f"Window {i}: caption is blank"

        # Inputs must be cleaned up after generation
        for i, window in enumerate(clip.windows):
            assert window.qwen_llm_input is None, f"Window {i}: qwen_llm_input not cleared after generation"

    def test_generation_stage_vllm_accepts_prepared_input_shape(
        self,
        preparation_stage: CaptionPreparationStage,
        generation_stage: CaptionGenerationStage,
        video_fixture_path: Path,
    ) -> None:
        """Verify the tensor dtype/shape contract between CaptionPreparationStage
        and vLLM's multimodal processor.

        This test is the primary HF-transformers ↔ vLLM regression catcher: if
        the HF AutoProcessor output shape or dtype changes across versions,
        vLLM's generate() will raise before producing any output.
        """
        video_bytes = video_fixture_path.read_bytes()
        task = _make_task(video_bytes, task_id="shape-contract-test")

        task = preparation_stage.process(task)

        clip = task.data.clips[0]
        assert len(clip.windows) > 0

        # Collect raw inputs before generation clears them
        raw_inputs = [w.qwen_llm_input for w in clip.windows if w.qwen_llm_input is not None]
        assert len(raw_inputs) > 0

        # vLLM must not raise on these inputs — no exception == pass
        captions = generation_stage.model.generate(
            raw_inputs,
            generate_stage2_caption=False,
            batch_size=len(raw_inputs),
        )

        assert len(captions) == len(raw_inputs), f"vLLM returned {len(captions)} captions for {len(raw_inputs)} inputs"
        for i, cap in enumerate(captions):
            assert isinstance(cap, str), f"Input {i}: vLLM output is not a str"


def _make_task_with_captions(captions: list[str], task_id: str = "enhancement-test") -> VideoTask:
    """Build a VideoTask with pre-populated window captions, bypassing all video stages.

    CaptionEnhancementStage only needs window.caption["qwen"] to be set — no
    video bytes, no HF preprocessing, no vLLM VL inference required.
    """
    windows = [
        _Window(start_frame=i * 10, end_frame=(i + 1) * 10, caption={"qwen": cap}) for i, cap in enumerate(captions)
    ]
    clip = Clip(uuid=uuid4(), source_video=task_id, span=(0.0, float(len(captions) * 10)))
    clip.windows = windows
    video = Video(input_video=Path(task_id + ".mp4"))
    video.clips = [clip]
    return VideoTask(task_id=task_id, dataset_name="integration", data=video)


# ---------------------------------------------------------------------------
# Caption Enhancement tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
class TestQwenLMCaptionEnhancementIntegration:
    """Integration tests for the QwenLM caption enhancement stage.

    Exercises the real Qwen2.5-14B-Instruct text model via vLLM.  No video
    bytes are needed — the stage works purely on existing caption strings.
    """

    def test_enhancement_stage_produces_enhanced_captions(
        self,
        enhancement_stage: CaptionEnhancementStage,
    ) -> None:
        """CaptionEnhancementStage must write a non-empty enhanced_caption["qwen_lm"]
        for every window that has a qwen caption, with no unhandled exceptions."""
        captions = [
            "A herd of cattle walking slowly through a wide green field.",
            "A person riding a bicycle on a busy city street.",
        ]
        task = _make_task_with_captions(captions)

        result = enhancement_stage.process(task)

        clip = result.data.clips[0]
        assert len(clip.errors) == 0, f"Enhancement stage set clip errors: {clip.errors}"
        assert len(clip.windows) == len(captions), "Window count changed after enhancement"

        for i, window in enumerate(clip.windows):
            enhanced = window.enhanced_caption.get("qwen_lm")
            assert enhanced is not None, f"Window {i}: enhanced_caption key 'qwen_lm' not set"
            assert isinstance(enhanced, str), f"Window {i}: enhanced_caption is not a str"
            assert len(enhanced.strip()) > 0, f"Window {i}: enhanced_caption is blank"
