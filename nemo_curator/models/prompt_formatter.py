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

from nemo_curator.models.nemotron_3_nano_omni import _HF_MODEL_ID as _NEMOTRON_3_NANO_OMNI_HF_ID
from nemo_curator.models.nemotron_h_vl import _NEMOTRON_VARIANTS_INFO

_RAW_VIDEO_NDIMS = 4
_RAW_VIDEO_MAX_VALUE = 255.0
_RAW_VIDEO_CHANNEL_COUNTS = {1, 3, 4}

# Mapping of variants to their HuggingFace model IDs
VARIANT_MAPPING: dict[str, str] = {
    "qwen2.5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    **_NEMOTRON_VARIANTS_INFO,
    "nemotron-3-nano-omni": _NEMOTRON_3_NANO_OMNI_HF_ID,
}


class PromptFormatter:
    """Unified prompt formatter for VLM models using HuggingFace AutoProcessor.

    Supports both Qwen and Nemotron model variants. Uses AutoProcessor.from_pretrained()
    to load the appropriate tokenizer and chat template from HuggingFace Hub or a local path.
    """

    def __init__(self, prompt_variant: str):
        """Initialize the prompt formatter.

        Args:
            prompt_variant: Model variant to use (e.g., "qwen", "nemotron", "nemotron-fp8", "nemotron-3-nano-omni").
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
        if self.prompt_variant in {"qwen2.5", "qwen3"}:
            return self._generate_qwen_inputs(prompt, video_inputs, override_text_prompt, fps)

        if self.prompt_variant.startswith("nemotron"):
            return self._generate_nemotron_inputs(prompt, video_inputs, fps)

        msg = f"Unsupported prompt variant: {self.prompt_variant}"
        raise ValueError(msg)

    def _generate_qwen_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | np.ndarray | None,
        override_text_prompt: bool,
        fps: float = 2.0,
    ) -> dict[str, Any]:
        """Generate inputs for Qwen models."""
        message = self._create_qwen_message(prompt)
        if self.text_prompt is None or override_text_prompt:
            self.text_prompt = self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        video_data = video_inputs
        if video_inputs is not None:
            video_np = self._format_raw_video_frames(video_inputs)
            num_frames = video_np.shape[0]
            video_data = (
                video_np,
                {"fps": fps, "frames_indices": list(range(num_frames)), "total_num_frames": num_frames},
            )
        return {
            "prompt": self.text_prompt,
            "multi_modal_data": {"video": video_data},
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
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": f"<video>\n{prompt}"}]},
        ]

        # Omni model has a thinking-enabled chat template; disable it for captioning
        template_kwargs: dict[str, Any] = {}
        if self.prompt_variant == "nemotron-3-nano-omni":
            template_kwargs["enable_thinking"] = False

        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

        # Handle video metadata (vLLM's Nemotron processor requires this tuple format)
        video_with_metadata = None
        if video_inputs is not None:
            video_np = self._format_raw_video_frames(video_inputs)
            num_frames = video_np.shape[0]
            video_with_metadata = (
                video_np,
                {"fps": fps, "frames_indices": list(range(num_frames))},
            )

        return {
            "prompt": formatted_prompt,
            "multi_modal_data": {"video": video_with_metadata},
        }

    def _format_raw_video_frames(self, video_inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Format raw decoded frames for vLLM video inputs as a contiguous uint8 array.

        Torch tensors are expected in decoder format (T, C, H, W). NumPy arrays
        are expected to already be channel-last (T, H, W, C).
        """
        if isinstance(video_inputs, torch.Tensor):
            # Decoder tensors are (T, C, H, W); vLLM processors expect (T, H, W, C).
            video_tensor = video_inputs.detach().permute(0, 2, 3, 1).cpu()
            if video_tensor.dtype == torch.bfloat16:
                video_tensor = video_tensor.float()
            video_np = video_tensor.numpy()
        else:
            # NumPy callers must provide vLLM-ready channel-last frames.
            video_np = video_inputs

        if video_np.ndim != _RAW_VIDEO_NDIMS:
            msg = f"Expected raw video frames with 4 dimensions, got shape {video_np.shape}"
            raise ValueError(msg)

        if not isinstance(video_inputs, torch.Tensor) and video_np.shape[-1] not in _RAW_VIDEO_CHANNEL_COUNTS:
            msg = (
                "Expected NumPy raw video frames in channel-last (T, H, W, C) format, "
                f"got shape {video_np.shape}"
            )
            raise ValueError(msg)

        if video_np.dtype == np.uint8:
            return np.ascontiguousarray(video_np)

        min_value = float(np.nanmin(video_np))
        max_value = float(np.nanmax(video_np))
        if min_value < 0:
            msg = "Captioning expects raw video frames, but got normalized frames with negative values."
            raise ValueError(msg)

        if max_value > _RAW_VIDEO_MAX_VALUE:
            msg = f"Raw video frame values exceed uint8 range: max={max_value}"
            raise ValueError(msg)

        return np.ascontiguousarray(video_np.astype(np.uint8))

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
