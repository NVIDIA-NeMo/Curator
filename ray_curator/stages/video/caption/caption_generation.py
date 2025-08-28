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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.qwen_vl import QwenVL
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Video, VideoTask


@dataclass
class CaptionGenerationStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that generates captions for video windows using specified VL model.

    This stage processes prepared video windows through the specified vision-language model to
    generate descriptive captions, with support for both synchronous and asynchronous processing.
    """

    model_dir: str = "models/qwen"
    model_variant: str = "qwen"
    caption_batch_size: int = 16
    fp8: bool = False
    max_output_tokens: int = 512
    model_does_preprocess: bool = False
    disable_mmcache: bool = False
    verbose: bool = False
    generate_stage2_caption: bool = False
    stage2_prompt_text: str | None = None
    _name: str = "caption_generation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.model_variant == "qwen":
            self.model = QwenVL(
                model_dir=self.model_dir,
                model_variant=self.model_variant,
                caption_batch_size=self.caption_batch_size,
                fp8=self.fp8,
                max_output_tokens=self.max_output_tokens,
                model_does_preprocess=self.model_does_preprocess,
                disable_mmcache=self.disable_mmcache,
            )
        else:
            msg = f"Unsupported model variant: {self.model_variant}"
            raise ValueError(msg)
        self.model.setup()

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the QwenVL model on the node."""
        QwenVL.download_weights_on_node(self.model_dir)

    def __post_init__(self) -> None:
        self._resources = Resources(gpus=1)

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        mapping: dict[int, tuple[int, int]] = {}
        inputs: list[dict[str, Any]] = []
        idx = 0
        for clip_idx, clip in enumerate(video.clips):
            if len(clip.windows) == 0:
                logger.warning(f"Clip {clip.uuid} has no windows")
                clip.errors["windows"] = "empty"
            for window_idx, window in enumerate(clip.windows):
                if window.qwen_llm_input is None:
                    logger.error(f"Clip {clip.uuid} window {window_idx} has no prepared inputs.")
                    clip.errors[f"window-{window_idx}"] = "empty"
                    continue
                mapping[idx] = (clip_idx, window_idx)
                inputs.append(window.qwen_llm_input)
                idx += 1

        captions = self.model.generate(
            inputs,
            generate_stage2_caption=self.generate_stage2_caption,
            batch_size=self.caption_batch_size,
        )

        self._assign_captions(video, mapping, enumerate(captions))

        return task

    def _assign_captions(
        self,
        video: Video,
        mapping: dict[int, tuple[int, int]],
        captions: Iterable[tuple[int, str]],
    ) -> None:
        _captions = list(captions)
        for req_id, caption in _captions:
            clip_idx, window_idx = mapping[req_id]
            video.clips[clip_idx].windows[window_idx].caption[self.model_variant] = caption
            if self.verbose:
                logger.info(f"Caption for clip {video.clips[clip_idx].uuid} window {window_idx}: {caption}")

        logger.info(
            f"Generated {len(_captions)} captions for video {video.input_path} "
            f"chunk-{video.clip_chunk_index} with {len(video.clips)} clips",
        )

        # Clear caption inputs
        for clip in video.clips:
            for window in clip.windows:
                window.qwen_llm_input = None
                window.mp4_bytes = None
