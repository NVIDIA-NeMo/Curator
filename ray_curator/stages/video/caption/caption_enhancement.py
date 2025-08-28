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

from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.qwen_lm import QwenLM
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, Video, VideoTask, _Window

_ENHANCE_PROMPTS = {
    "default": """
        You are a chatbot that enhances video caption inputs, adding more color and details to the text.
        The output should be longer than the provided input caption.
    """,
    "av-surveillance": """
        You are a chatbot that enhances video captions from vehicle dashboard cameras or surveillance cameras.
        Add more details and generate a summary from the original text.
        The output should be longer than the provided input caption.
    """,
}


@dataclass
class CaptionEnhancementStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that enhances video captions using language models.

    This stage takes existing captions and uses LLM (e.g. Qwen) to generate
    more detailed and refined descriptions of the video content.
    """

    model_dir: str = "models/qwen"
    model_variant: str = "qwen"
    prompt_variant: str = "default"
    prompt_text: str | None = None
    model_batch_size: int = 128
    fp8: bool = False
    max_output_tokens: int = 512
    verbose: bool = False
    _name: str = "caption_enhancement"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def __post_init__(self) -> None:
        self._resources = Resources(gpus=1)
        self.prompt = _get_enhance_prompt(
            self.prompt_variant,
            self.prompt_text,
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.model_variant == "qwen":
            self.model = QwenLM(
                model_dir=self.model_dir,
                caption_batch_size=self.model_batch_size,
                fp8=self.fp8,
                max_output_tokens=self.max_output_tokens,
            )
        else:
            msg = f"Unsupported model variant: {self.model_variant}"
            raise ValueError(msg)
        self.model.setup()

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the QwenLM model on the node."""
        QwenLM.download_weights_on_node(self.model_dir)

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        mapping, inputs = self._prepare_caption_inputs(video)

        if len(inputs) > 0:
            self._generate_and_assign_captions(video, mapping, inputs)

        return task

    def _prepare_caption_inputs(self, video: Video) -> tuple[dict[int, tuple[int, int]], list[dict[str, Any]]]:
        """Prepare caption inputs from video clips and windows."""
        mapping: dict[int, tuple[int, int]] = {}
        inputs: list[dict[str, Any]] = []
        idx = 0

        for clip_idx, clip in enumerate(video.clips):
            if len(clip.windows) == 0:
                logger.warning(f"Clip {clip.uuid} has no windows")
                clip.errors["windows"] = "empty"
                continue

            for window_idx, window in enumerate(clip.windows):
                if not self._is_valid_window_caption(clip, window, window_idx):
                    continue

                caption = window.caption["qwen"]
                if caption is None:
                    logger.error(f"Clip {clip.uuid} window {window_idx} has no caption")
                    continue

                mapping[idx] = (clip_idx, window_idx)
                caption_input = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": caption},
                ]
                inputs.append(caption_input)
                idx += 1

        return mapping, inputs

    def _is_valid_window_caption(self, clip: Clip, window: _Window, window_idx: int) -> bool:
        """Check if window has valid caption data."""
        if window.caption is None:
            logger.error(f"Clip {clip.uuid} window {window_idx} has no caption")
            clip.errors[f"window-{window_idx}"] = "empty"
            return False

        return "qwen" in window.caption

    def _generate_and_assign_captions(
        self, video: Video, mapping: dict[int, tuple[int, int]], inputs: list[dict[str, Any]]
    ) -> None:
        """Generate enhanced captions and assign them to video windows."""
        captions = []
        for i in range(0, len(inputs), self.model_batch_size):
            captions.extend(self.model.generate(inputs[i : i + self.model_batch_size]))

        if len(captions) != len(inputs):
            logger.error(f"Caption generation failed: expected {len(inputs)} captions, got {len(captions)}")
            return

        for idx, result in enumerate(captions):
            clip_idx, window_idx = mapping[idx]
            original_caption = video.clips[clip_idx].windows[window_idx].caption["qwen"]
            video.clips[clip_idx].windows[window_idx].enhanced_caption["qwen_lm"] = result

            if self.verbose:
                logger.info(
                    f"Caption for clip {video.clips[clip_idx].uuid} window {window_idx}: {original_caption}",
                )
                logger.info(
                    f"Enhanced QwenLM Caption for clip {video.clips[clip_idx].uuid} window {window_idx}: {result}",
                )


def _get_enhance_prompt(prompt_variant: str, prompt_text: str | None, *, verbose: bool = False) -> str:
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _ENHANCE_PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _ENHANCE_PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Enhance Captioning prompt: {prompt}")
    return prompt
