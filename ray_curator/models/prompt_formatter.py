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

from typing import Any

import torch
from transformers import AutoProcessor

VARIANT_MAPPING = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}


class PromptFormatter:
    def __init__(self, prompt_variant: str):
        if prompt_variant not in VARIANT_MAPPING:
            msg = f"Invalid prompt variant: {prompt_variant}. Valid variants are: {', '.join(VARIANT_MAPPING.keys())}"
            raise ValueError(msg)

        self.prompt_variant = prompt_variant
        self.text_prompt = None
        self.processor = AutoProcessor.from_pretrained(VARIANT_MAPPING[self.prompt_variant])

    def generate_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | None = None,
        *,
        override_text_prompt: bool = False,
    ) -> dict[str, Any]:
        """Generate inputs for video and text data based on prompt_variant.

        Processes video and text inputs to create the input for the model. It handles both video and
        image inputs, decoding video and applying preprocessing if needed, and creates a structured
        input dictionary containing the processed prompt and multimodal data.

        Args:
            prompt: Text prompt to be included with the input.
            fps: Frames per second of the input video.
            preprocess_dtype: Data type to use for preprocessing the video/image inputs.
            num_frames_to_use: Number of frames to extract from the video. If 0, uses all frames.
            flip_input: Whether to flip the input video/image horizontally.
            video_inputs: Pre-processed video inputs. If None, and video data is to be passed to
                          the model, then video cannot be None.
            override_text_prompt: whether the text prompt should be overridden

        Returns:
            dict containing:
                - "prompt": The processed text prompt with chat template applied
                - "multi_modal_data": Dictionary containing processed "image" and/or "video" inputs

        """
        message = self.create_message(prompt)
        if self.text_prompt is None or override_text_prompt:
            self.text_prompt = self.processor.apply_chat_template(
                message,
                tokenizer=False,
                add_generation_prompt=True,
            )
        return {
            "prompt": self.text_prompt,
            "multi_modal_data": {"video": video_inputs},
        }

    def create_message(self, prompt: str) -> list[dict[str, Any]]:
        """Create a message.

        Args:
            text_input: The text input to create a message for.

        Returns:
            List of messages for the VLM model including the text prompt and video.

        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
