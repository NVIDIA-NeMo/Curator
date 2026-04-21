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

import multiprocessing
from pathlib import Path
from typing import Any, Final

from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils import grouping
from nemo_curator.utils.hf_download_utils import download_model_from_hf

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


# HuggingFace model ID — set to None until the model is publicly released.
# Once released, fill in the HF repo ID (e.g. "nvidia/Nemotron-3-Nano-Omni-...") and
# optionally pin a revision below. Everything else (download, weight path, processor
# resolution in PromptFormatter) will work automatically.
_HF_MODEL_ID: Final[str | None] = None
_HF_REVISION: Final[str | None] = None

# Constants for stage-2 prompt refinement
_VIDEO_TAG_SPLIT_MAX = 1
_EXPECTED_VIDEO_TAG_PARTS = 2


class Nemotron3NanoOmni(ModelInterface):
    """Nemotron 3 Nano Omni multimodal VLM for video captioning.

    Currently loaded from a local checkpoint; will support automatic HuggingFace
    download once the model is publicly released (_HF_MODEL_ID is set).

    When _HF_MODEL_ID is None (local mode):
        model_dir  — path to the checkpoint directory itself
    When _HF_MODEL_ID is set (HF mode):
        model_dir  — base directory; weights are stored at model_dir/<hf_model_id>/
    """

    def __init__(
        self,
        model_dir: str,
        caption_batch_size: int = 8,
        max_output_tokens: int = 512,
        stage2_prompt_text: str | None = None,
        verbose: bool = False,
    ):
        self.model_dir = model_dir
        self.caption_batch_size = caption_batch_size
        self.max_output_tokens = max_output_tokens
        self.stage2_prompt = stage2_prompt_text or "Please refine this caption: "
        self.verbose = verbose

        if _HF_MODEL_ID is None:
            self.weight_file = model_dir
        else:
            self.weight_file = str(Path(model_dir) / _HF_MODEL_ID)

    @property
    def model_id_names(self) -> list[str]:
        return [_HF_MODEL_ID] if _HF_MODEL_ID is not None else []

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = (
                "vllm is required for Nemotron3NanoOmni but is not installed. "
                "Please install vllm: pip install vllm"
            )
            raise ImportError(msg)

        # vLLM v1 spawns an EngineCore subprocess. When running inside a Ray GPU actor,
        # Ray sets the global multiprocessing start method to 'fork' and initializes CUDA
        # in the actor process. PyTorch then refuses to re-initialize CUDA in a forked
        # subprocess. Forcing 'spawn' creates a fresh subprocess that initializes CUDA cleanly.
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        self.model = LLM(
            model=self.weight_file,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
            video_pruning_rate=0,
        )

        # Omni uses <|im_end|> as the turn terminator; thinking is disabled so no </think>
        self.sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=self.max_output_tokens,
            top_p=0.95,
            stop=["<|im_end|>"],
        )

        logger.info("Nemotron3NanoOmni initialized: TP=1, GPU_util=0.9, max_len=32768")

    def _refine_caption_prompt(self, original_prompt: str, refinement_text: str) -> str:
        if "<video>" not in original_prompt:
            return refinement_text

        parts = original_prompt.split("<video>", _VIDEO_TAG_SPLIT_MAX)
        if len(parts) != _EXPECTED_VIDEO_TAG_PARTS:
            return refinement_text

        prefix = parts[0] + "<video>"
        suffix_start = len(parts[1])
        for marker in ["<|im_end|>", "</s>"]:
            if marker in parts[1]:
                suffix_start = parts[1].index(marker)
                break

        return prefix + "\n" + refinement_text + parts[1][suffix_start:]

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
                outputs = self.model.generate(
                    model_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                if generate_stage2_caption:
                    for i, out in enumerate(outputs):
                        initial_caption = out.outputs[0].text
                        refinement_text = self.stage2_prompt + initial_caption
                        model_inputs[i]["prompt"] = self._refine_caption_prompt(
                            model_inputs[i]["prompt"], refinement_text
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
        """Download Nemotron3NanoOmni weights.

        No-op while _HF_MODEL_ID is None (local checkpoint mode). Once the model is
        publicly released and _HF_MODEL_ID is set, this will download from HuggingFace.
        """
        if _HF_MODEL_ID is None:
            logger.info(
                f"Nemotron3NanoOmni is a local checkpoint at {model_dir!r}, skipping HF download"
            )
            return

        model_dir_path = Path(model_dir) / _HF_MODEL_ID
        model_dir_path.mkdir(parents=True, exist_ok=True)

        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            logger.info(f"Nemotron3NanoOmni checkpoint already exists at: {model_dir_path}")
            return

        logger.info(f"Downloading Nemotron3NanoOmni from HuggingFace: {_HF_MODEL_ID}")
        download_model_from_hf(
            model_id=_HF_MODEL_ID,
            local_dir=model_dir_path,
            revision=_HF_REVISION,
        )
        logger.info(f"Nemotron3NanoOmni weights downloaded to: {model_dir_path}")
