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

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from jinja2 import Template
from loguru import logger

# Constants for tensor dimensions and channel counts
EXPECTED_TENSOR_DIMENSIONS = 4
EXPECTED_NUMPY_DIMENSIONS = 4
EXPECTED_CHANNELS = 3


class NemotronPromptFormatter:
    """Prompt formatter for NemotronH models using Jinja2 chat templates."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        checkpoint_path = self._resolve_checkpoint_path(model_path)

        template_path = checkpoint_path / "chat_template.jinja"
        if not template_path.exists():
            msg = (
                f"chat_template.jinja not found at {template_path}.\n"
                f"NemotronH models require this template file for proper prompt formatting.\n"
                f"Please ensure your checkpoint directory contains chat_template.jinja."
            )
            raise FileNotFoundError(msg)

        with open(template_path) as f:
            self.chat_template = f.read()
        logger.info(f"Loaded chat template from: {template_path}")

    def _resolve_checkpoint_path(self, model_path: str) -> Path:
        path = Path(model_path)

        if (path / "config.json").exists():
            return path

        return path

    def generate_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | npt.NDArray[np.uint8] | None = None,
        add_generation_prompt: bool = True,
    ) -> dict[str, Any]:
        video_np = self._convert_video_format(video_inputs)
        formatted_prompt = self._apply_chat_template(prompt, add_generation_prompt)

        return {
            "prompt": formatted_prompt,
            "multi_modal_data": {"video": video_np},
        }

    def _convert_tensor_to_numpy(self, tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
        """Convert torch.Tensor (T, C, H, W) to numpy (T, H, W, C)."""
        if tensor.ndim != EXPECTED_TENSOR_DIMENSIONS:
            msg = f"Expected 4D torch.Tensor (T, C, H, W), got shape {tensor.shape}"
            raise ValueError(msg)

        video_np = tensor.permute(0, 2, 3, 1).cpu().numpy()
        return self._normalize_dtype(video_np)

    def _validate_numpy_array(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Validate and normalize numpy array format."""
        if array.ndim != EXPECTED_NUMPY_DIMENSIONS:
            msg = f"Expected 4D numpy array (T, H, W, C), got shape {array.shape}"
            raise ValueError(msg)

        if array.shape[-1] != EXPECTED_CHANNELS:
            msg = f"Expected channels-last format (T, H, W, 3), got shape {array.shape}."
            raise ValueError(msg)

        return self._normalize_dtype(array)

    def _normalize_dtype(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Normalize array dtype to uint8."""
        if array.dtype != np.uint8:
            if array.dtype in (np.float32, np.float16) and array.max() <= 1.0:
                return (array * 255).astype(np.uint8)
            return array.astype(np.uint8)
        return array

    def _convert_video_format(
        self,
        video_inputs: torch.Tensor | npt.NDArray[np.uint8] | None,
    ) -> npt.NDArray[np.uint8] | None:
        """Convert torch.Tensor (T, C, H, W) or np.ndarray to vLLM format (T, H, W, C)."""
        if video_inputs is None:
            return None

        if isinstance(video_inputs, torch.Tensor):
            return self._convert_tensor_to_numpy(video_inputs)

        if isinstance(video_inputs, np.ndarray):
            return self._validate_numpy_array(video_inputs)

        msg = f"Expected torch.Tensor or np.ndarray, got {type(video_inputs)}"
        raise TypeError(msg)

    def _apply_chat_template(
        self,
        prompt_text: str,
        add_generation_prompt: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "text", "text": f"<video>\n{prompt_text}"}],
            },
        ]

        template = Template(self.chat_template)
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

