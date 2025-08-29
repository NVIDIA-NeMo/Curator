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
from itertools import zip_longest

from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.prompt_formatter import PromptFormatter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.video import VideoTask, _Window
from nemo_curator.utils import windowing_utils

_PROMPTS = {
    "default": """
        Elaborate on the visual and narrative elements of the video in detail.
    """,
    "av": """
        The video depicts the view from a camera mounted on a car as it is driving.
        Pay special attention to the motion of the cars, including the primary car
        whose point-of-view we observe in the video. Also note important factors
        that would relate to driving safety like the relative positions of pedestrians,
        lane markers, road signs, traffic signals, and any aggressive driving behavior
        of other vehicles. Also pay attention to interesting landmarks and describe
        them in detail.
    """,
    "av-surveillance": """
        The video depicts the view from a surveillance camera. Pay special attention
        to the motion of the cars and other important factors that would relate to
        driving safety like the relative positions of pedestrians, lane markers,
        road signs, traffic signals, and any aggressive driving behavior of vehicles.
        Also pay attention to interesting landmarks and describe them in detail.
    """,
}

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


# Use with Captioning Stage
def _get_prompt(
    prompt_variant: str,
    prompt_text: str | None,
) -> str:
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _PROMPTS[prompt_variant]
    return prompt


@dataclass
class CaptionPreparationStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that prepares captions for video processing."""

    model_variant: str = "qwen"
    prompt_variant: str = "default"
    prompt_text: str | None = None
    verbose: bool = False
    sampling_fps: float = 2.0
    window_size: int = 256
    remainder_threshold: int = 128
    model_does_preprocess: bool = False
    preprocess_dtype: str = "float32"
    generate_previews: bool = True
    _name: str = "caption_preparation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self.prompt_formatter = PromptFormatter(self.model_variant)

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data

        for clip in video.clips:
            if clip.buffer is None:
                logger.warning(f"No buffer found for clip {clip.id}")
                clip.errors["buffer"] = "empty"
                continue

            for window_bytes, window_frames, window_frame_info in zip_longest(
                *windowing_utils.split_video_into_windows(
                    clip.buffer,
                    window_size=self.window_size,
                    remainder_threshold=self.remainder_threshold,
                    sampling_fps=self.sampling_fps,
                    model_does_preprocess=self.model_does_preprocess,
                    preprocess_dtype=self.preprocess_dtype,
                    return_bytes=self.generate_previews,
                    num_threads=max(int(self.resources.cpus), 1),
                ),
            ):
                prompt = _get_prompt(
                    self.prompt_variant,
                    self.prompt_text,
                )
                try:
                    llm_input = self.prompt_formatter.generate_inputs(
                        prompt=prompt,
                        video_inputs=window_frames,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error in Caption preparation: {e}")
                    clip.errors[f"{self.model_variant}_input"] = str(e)
                    continue

                clip.windows.append(
                    _Window(
                        window_frame_info.start,
                        window_frame_info.end,
                        mp4_bytes=window_bytes,
                        qwen_llm_input=llm_input,
                    ),
                )

        return task
