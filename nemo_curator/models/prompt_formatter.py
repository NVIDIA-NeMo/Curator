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

import numpy as np
import torch
from transformers import AutoProcessor

from nemo_curator.models.nemotron_h_vl import _NEMOTRON_VARIANTS_INFO

# Mapping of variants to their HuggingFace model IDs
VARIANT_MAPPING: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    **_NEMOTRON_VARIANTS_INFO,
}


class PromptFormatter:
    """Unified prompt formatter for VLM models using HuggingFace AutoProcessor.

    Supports both Qwen and Nemotron model variants. Uses AutoProcessor.from_pretrained()
    to load the appropriate tokenizer and chat template from HuggingFace Hub.
    """

    def __init__(self, prompt_variant: str):
        """Initialize the prompt formatter.

        Args:
            prompt_variant: Model variant to use (e.g., "qwen", "nemotron", "nemotron-fp8").
        """
        if prompt_variant not in VARIANT_MAPPING:
            msg = f"Invalid prompt variant: {prompt_variant}. Valid variants are: {', '.join(VARIANT_MAPPING.keys())}"
            raise ValueError(msg)

        self.prompt_variant = prompt_variant
        self.text_prompt = None

        # Load processor from HuggingFace (auto-downloads and caches)
        hf_model_id = VARIANT_MAPPING[prompt_variant]
        self.processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)

    def generate_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | np.ndarray | None = None,
        *,
        override_text_prompt: bool = False,
        fps: float = 2.0,
    ) -> dict[str, Any]:
        """Generate inputs for video and text data based on prompt_variant.

        Args:
            prompt: Text prompt to be included with the input.
            video_inputs: Pre-processed video inputs (tensor or numpy array).
            override_text_prompt: Whether to regenerate the text prompt even if cached.
            fps: Frames per second of the input video (used for Nemotron metadata).

        Returns:
            dict containing:
                - "prompt": The processed text prompt with chat template applied
                - "multi_modal_data": Dictionary containing processed "video" inputs

        """
        if self.prompt_variant == "qwen":
            return self._generate_qwen_inputs(prompt, video_inputs, override_text_prompt)

        if self.prompt_variant.startswith("nemotron"):
            return self._generate_nemotron_inputs(prompt, video_inputs, fps)

        msg = f"Unsupported prompt variant: {self.prompt_variant}"
        raise ValueError(msg)

    def _generate_qwen_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | None,
        override_text_prompt: bool,
    ) -> dict[str, Any]:
        """Generate inputs for Qwen models."""
        message = self._create_qwen_message(prompt)
        if self.text_prompt is None or override_text_prompt:
            self.text_prompt = self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        return {
            "prompt": self.text_prompt,
            "multi_modal_data": {"video": video_inputs},
        }

    def _generate_nemotron_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | np.ndarray | None,
        fps: float,
    ) -> dict[str, Any]:
        """Generate inputs for Nemotron models.

        Nemotron requires video metadata (fps, frames_indices) for vLLM processing.
        """
        # Format messages for Nemotron
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": f"<video>\n{prompt}"}]},
        ]

        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Handle video metadata (vLLM's Nemotron processor requires this tuple format)
        video_with_metadata = None
        if video_inputs is not None:
            video_np = self._convert_to_numpy(video_inputs)
            num_frames = video_np.shape[0]
            video_with_metadata = (
                video_np,
                {"fps": fps, "frames_indices": list(range(num_frames))},
            )

        return {
            "prompt": formatted_prompt,
            "multi_modal_data": {"video": video_with_metadata},
        }

    def _convert_to_numpy(self, video_inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert video inputs to numpy array in (T, H, W, C) format."""
        if isinstance(video_inputs, torch.Tensor):
            # Assume tensor is (T, C, H, W), convert to (T, H, W, C)
            video_np = video_inputs.permute(0, 2, 3, 1).cpu().numpy()
        else:
            video_np = video_inputs

        # Normalize to uint8 if needed
        if video_np.dtype != np.uint8:
            if np.issubdtype(video_np.dtype, np.floating) and video_np.max() <= 1.0:
                video_np = (video_np * 255).astype(np.uint8)
            else:
                video_np = video_np.astype(np.uint8)

        return video_np

    def _create_qwen_message(self, prompt: str) -> list[dict[str, Any]]:
        """Create a message for Qwen models."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
