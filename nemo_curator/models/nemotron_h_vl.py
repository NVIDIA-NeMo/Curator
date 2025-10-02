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

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils import grouping
from nemo_curator.utils.hf_download_utils import download_model_from_hf

# Constants for prompt processing
VIDEO_TAG_SPLIT_MAX = 1
EXPECTED_VIDEO_TAG_PARTS = 2

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


# HuggingFace model IDs (will be updated when models are published)
_NEMOTRON_H_NANO_MODEL_ID = None
_NEMOTRON_H_NANO_MODEL_REVISION = None


class NemotronHVL(ModelInterface):
    """NemotronH hybrid Mamba-Attention VLM for video captioning."""

    def __init__(  # noqa: PLR0913
        self,
        model_dir: str,
        model_variant: str = "nemotron",
        caption_batch_size: int = 8,
        max_output_tokens: int = 512,
        stage2_prompt_text: str | None = None,
        verbose: bool = False,
    ):
        self.model_dir = model_dir
        self.model_variant = model_variant
        self.caption_batch_size = caption_batch_size
        self.max_output_tokens = max_output_tokens
        self.stage2_prompt = stage2_prompt_text if stage2_prompt_text else "Please refine this caption: "
        self.verbose = verbose

        if _NEMOTRON_H_NANO_MODEL_ID is not None:
            self.weight_file = str(Path(model_dir) / _NEMOTRON_H_NANO_MODEL_ID)
        else:
            # Local checkpoint: model_dir is the checkpoint path itself
            self.weight_file = str(Path(model_dir))

    @property
    def model_id_names(self) -> list[str]:
        """Return model ID from config.json or HuggingFace ID."""
        if _NEMOTRON_H_NANO_MODEL_ID is not None:
            return [_NEMOTRON_H_NANO_MODEL_ID]

        # Read from config.json if available
        try:
            config_path = Path(self.weight_file) / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            return [config.get("_name_or_path", Path(self.weight_file).name)]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return [Path(self.weight_file).name]

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for NemotronHVL but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        # Use V0 engine to avoid flashinfer issues for now
        os.environ["VLLM_USE_V1"] = "0"
        logger.info("Using vLLM V0 engine.")

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.model = LLM(
            model=self.weight_file,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
        )

        self.sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=self.max_output_tokens,
            top_p=0.95,
            stop=["</s>", "<|endoftext|>", "<SPECIAL_12>", "</think>"],
        )

        logger.info(
            f"NemotronHVL initialized: variant={self.model_variant}, "
            f"TP=1, GPU_util=0.9, max_len=32768"
        )

    def _refine_caption_prompt(self, original_prompt: str, refinement_text: str) -> str:
        """Create a refined prompt for stage 2 captioning."""
        if "<video>" not in original_prompt:
            return refinement_text

        parts = original_prompt.split("<video>", VIDEO_TAG_SPLIT_MAX)
        if len(parts) != EXPECTED_VIDEO_TAG_PARTS:
            return refinement_text

        prefix = parts[0] + "<video>"

        # Find where the user message ends
        suffix_markers = ["<SPECIAL_11>Assistant", "<|im_end|>", "</s>"]
        suffix_start = len(parts[1])
        for marker in suffix_markers:
            if marker in parts[1]:
                suffix_start = parts[1].index(marker)
                break

        suffix = parts[1][suffix_start:]
        return prefix + "\n" + refinement_text + suffix

    def generate(
        self,
        videos: list[dict[str, Any]],
        generate_stage2_caption: bool = False,
        batch_size: int = 16,
    ) -> list[str]:
        generated_text = []

        for batch_videos in grouping.split_by_chunk_size(videos, batch_size):
            model_inputs = list(batch_videos)
            try:
                # PASS 1: Generate initial captions
                outputs = self.model.generate(
                    model_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                # PASS 2: Refine captions if requested
                if generate_stage2_caption:
                    for i, out in enumerate(outputs):
                        initial_caption = out.outputs[0].text
                        refinement_text = self.stage2_prompt + initial_caption
                        original_prompt = model_inputs[i]["prompt"]
                        model_inputs[i]["prompt"] = self._refine_caption_prompt(
                            original_prompt, refinement_text
                        )

                    outputs = self.model.generate(
                        model_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )

                generated_text.extend(out.outputs[0].text for out in outputs)

                if self.verbose:
                    for i, out in enumerate(outputs):
                        logger.info(f"Generated caption {i}: {out.outputs[0].text[:100]}...")

            except Exception as e:
                logger.error(f"Error generating caption for batch: {e}")
                raise

        return generated_text

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download or verify NemotronH weights on the node (follows Qwen pattern)."""
        # If HuggingFace model ID is configured, download from HF
        if _NEMOTRON_H_NANO_MODEL_ID is not None:
            model_dir_path = Path(model_dir) / _NEMOTRON_H_NANO_MODEL_ID
            model_dir_path.mkdir(parents=True, exist_ok=True)

            # Check if already downloaded
            if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
                logger.info(f"NemotronH checkpoint already exists at: {model_dir_path}")
                return

            # Download from HuggingFace
            download_model_from_hf(
                model_id=_NEMOTRON_H_NANO_MODEL_ID,
                local_dir=model_dir_path,
                revision=_NEMOTRON_H_NANO_MODEL_REVISION,
            )
            logger.info(f"NemotronH weights downloaded to: {model_dir_path}")
        else:
            # Local checkpoint mode: validate path exists
            model_path = Path(model_dir)
            if not model_path.exists():
                msg = f"NemotronH checkpoint path does not exist: {model_dir}"
                raise FileNotFoundError(msg)

            # Verify it contains model files
            if not any(model_path.glob("*.safetensors")) and not any(model_path.glob("*.bin")):
                msg = (
                    f"No model files (.safetensors or .bin) found in: {model_dir}\n"
                    f"Please ensure this directory contains a valid NemotronH checkpoint."
                )
                raise FileNotFoundError(msg)

            logger.info(f"Using local NemotronH checkpoint: {model_dir}")

