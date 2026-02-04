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

import os
from pathlib import Path
from typing import Any, Final, Literal

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


# HuggingFace model IDs for Nemotron Nano V2 VL variants
# Available variants: BF16 (default), FP8, NVFP4-QAD
_NEMOTRON_VARIANTS_INFO: Final = {
    "nemotron": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",  # Default BF16 variant
    "nemotron-bf16": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "nemotron-fp8": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
    "nemotron-nvfp4": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
}

_NEMOTRON_REVISION_INFO: Final = {
    "nemotron": "5d250e2e111dc5e1434131bdf3d590c27a878ade",  # BF16 default
    "nemotron-bf16": "5d250e2e111dc5e1434131bdf3d590c27a878ade",
    "nemotron-fp8": "7394488badb786e1decc0e00e308de1cab9560e6",
    "nemotron-nvfp4": "b8d3c170d9ee3a078917ef9bfd508eff988d6de7",
}

NemotronVariant = Literal["nemotron", "nemotron-bf16", "nemotron-fp8", "nemotron-nvfp4"]


class NemotronHVL(ModelInterface):
    """NemotronH hybrid Mamba-Attention VLM for video captioning.

    Supports multiple checkpoint variants from HuggingFace:
    - nemotron / nemotron-bf16: BF16 precision (default)
    - nemotron-fp8: FP8 quantized
    - nemotron-nvfp4: NVFP4 quantized

    Models are automatically downloaded from HuggingFace on first use.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_dir: str,
        model_variant: NemotronVariant = "nemotron",
        caption_batch_size: int = 8,
        max_output_tokens: int = 512,
        stage2_prompt_text: str | None = None,
        verbose: bool = False,
    ):
        """Initialize NemotronHVL model.

        Args:
            model_dir: Base directory for model weights. Models will be downloaded
                to subdirectories named after the HuggingFace model ID.
            model_variant: Model variant to use. Options:
                - "nemotron" or "nemotron-bf16": BF16 precision (default)
                - "nemotron-fp8": FP8 quantized
                - "nemotron-nvfp4": NVFP4 quantized
            caption_batch_size: Batch size for caption generation.
            max_output_tokens: Maximum number of tokens to generate.
            stage2_prompt_text: Optional prompt text for stage 2 caption refinement.
            verbose: Whether to enable verbose logging.
        """
        # Normalize variant name - treat "nemotron" as "nemotron-bf16"
        if model_variant == "nemotron":
            self._normalized_variant: NemotronVariant = "nemotron-bf16"
        else:
            self._normalized_variant = model_variant

        if self._normalized_variant not in _NEMOTRON_VARIANTS_INFO:
            valid_variants = ", ".join(_NEMOTRON_VARIANTS_INFO.keys())
            msg = f"Invalid model_variant: {model_variant}. Valid options are: {valid_variants}"
            raise ValueError(msg)

        self.model_dir = model_dir
        self.model_variant = model_variant
        self.caption_batch_size = caption_batch_size
        self.max_output_tokens = max_output_tokens
        self.stage2_prompt = stage2_prompt_text if stage2_prompt_text else "Please refine this caption: "
        self.verbose = verbose

        # Set weight file path using HuggingFace model ID
        self._hf_model_id = _NEMOTRON_VARIANTS_INFO[self._normalized_variant]
        self.weight_file = str(Path(model_dir) / self._hf_model_id)

    @property
    def model_id_names(self) -> list[str]:
        """Return HuggingFace model ID for the selected variant."""
        return [self._hf_model_id]

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for NemotronHVL but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Determine quantization and dtype based on variant
        # See: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8
        #      https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD
        quantization = None
        dtype = "bfloat16"  # BF16 variant requires explicit dtype
        if self._normalized_variant == "nemotron-fp8":
            quantization = "modelopt"  # vllm serve uses: --quantization modelopt
            dtype = "auto"  # FP8 determines dtype from quantization
        elif self._normalized_variant == "nemotron-nvfp4":
            quantization = "modelopt_fp4"  # vllm serve uses: --quantization modelopt_fp4
            dtype = "auto"  # FP4 determines dtype from quantization

        self.model = LLM(
            model=self.weight_file,
            trust_remote_code=True,
            dtype=dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
            quantization=quantization,
            video_pruning_rate=0,  # Disable video pruning
        )

        self.sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=self.max_output_tokens,
            top_p=0.95,
            stop=["</s>", "<|endoftext|>", "<SPECIAL_12>", "</think>"],
        )

        logger.info(
            f"NemotronHVL initialized: variant={self.model_variant}, "
            f"quantization={quantization}, TP=1, GPU_util=0.9, max_len=32768"
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
                        model_inputs[i]["prompt"] = self._refine_caption_prompt(original_prompt, refinement_text)

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
    def download_weights_on_node(
        cls,
        model_dir: str,
        variant: NemotronVariant = "nemotron",
    ) -> None:
        """Download NemotronH VL weights from HuggingFace.

        Models are automatically downloaded from HuggingFace Hub on first use.
        Supports multiple quantization variants for different performance/memory tradeoffs.

        Args:
            model_dir: Base directory for model weights. The model will be downloaded
                to a subdirectory named after the HuggingFace model ID.
            variant: Model variant to download. Options:
                - "nemotron" or "nemotron-bf16": BF16 precision (default)
                - "nemotron-fp8": FP8 quantized
                - "nemotron-nvfp4": NVFP4 quantized
        """
        # Normalize variant name
        normalized_variant: NemotronVariant = "nemotron-bf16" if variant == "nemotron" else variant

        if normalized_variant not in _NEMOTRON_VARIANTS_INFO:
            valid_variants = ", ".join(_NEMOTRON_VARIANTS_INFO.keys())
            msg = f"Invalid variant: {variant}. Valid options are: {valid_variants}"
            raise ValueError(msg)

        hf_model_id = _NEMOTRON_VARIANTS_INFO[normalized_variant]
        revision = _NEMOTRON_REVISION_INFO.get(normalized_variant)

        model_dir_path = Path(model_dir) / hf_model_id
        model_dir_path.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            logger.info(f"NemotronH {variant} checkpoint already exists at: {model_dir_path}")
            return

        # Download from HuggingFace
        logger.info(f"Downloading NemotronH {variant} from HuggingFace: {hf_model_id}")
        download_model_from_hf(
            model_id=hf_model_id,
            local_dir=model_dir_path,
            revision=revision,
        )
        logger.info(f"NemotronH {variant} weights downloaded to: {model_dir_path}")
