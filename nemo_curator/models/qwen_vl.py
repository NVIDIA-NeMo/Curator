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

import pathlib
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.utils.hf_download_utils import download_model_from_hf

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    # Create dummy classes for type hints when vllm is not available
    class LLM:
        pass

    class SamplingParams:
        pass


from nemo_curator.models.base import ModelInterface
from nemo_curator.utils import grouping

_QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
_QWEN2_5_VL_MODEL_REVISION = "cc59489"

_QWEN_VARIANTS_INFO = {
    "qwen": _QWEN2_5_VL_MODEL_ID,
}


class QwenVL(ModelInterface):
    def __init__(  # noqa: PLR0913
        self,
        model_dir: str,
        model_variant: str,
        caption_batch_size: int,
        fp8: bool = True,
        max_output_tokens: int = 512,
        model_does_preprocess: bool = False,
        disable_mmcache: bool = False,
        stage2_prompt_text: str | None = None,
        verbose: bool = False,
    ):
        self.model_dir = model_dir
        self.model_variant = model_variant
        self.caption_batch_size = caption_batch_size
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.model_does_preprocess = model_does_preprocess
        self.disable_mmcache = disable_mmcache
        self.stage2_prompt = stage2_prompt_text
        self.verbose = verbose
        self.weight_file = str(pathlib.Path(model_dir) / _QWEN_VARIANTS_INFO[model_variant])
        # Default pattern for stage2 caption generation - matches (.*)(user_prompt)(.*)
        self.pattern = r"(.*)(user_prompt)(.*)"

    @property
    def model_id_names(self) -> list[str]:
        return [_QWEN_VARIANTS_INFO[self.model_variant]]

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for QwenVL model but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        mm_processor_kwargs = {
            "do_resize": self.model_does_preprocess,
            "do_rescale": self.model_does_preprocess,
            "do_normalize": self.model_does_preprocess,
        }
        self.model = LLM(
            model=self.weight_file,
            limit_mm_per_prompt={"image": 0, "video": 1},
            quantization="fp8" if self.fp8 else None,
            max_seq_len_to_capture=32768,
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            mm_processor_kwargs=mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mmcache,
            max_num_batched_tokens=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )
        logger.info(
            "CUDA graph enabled for sequences smaller than 16k tokens; adjust accordingly for even longer sequences"
        )

    def generate(
        self, videos: list[dict[str, Any]], generate_stage2_caption: bool = False, batch_size: int = 16
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
                        out_caption = out.outputs[0].text
                        updated_prompt = self.stage2_prompt + out_caption
                        model_inputs[i]["prompt"] = re.sub(
                            self.pattern,
                            rf"\1{updated_prompt}\3",
                            model_inputs[i]["prompt"],
                            flags=re.DOTALL,
                        )
                    outputs = self.model.generate(
                        model_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                generated_text.extend(out.outputs[0].text for out in outputs)
            except Exception as e:
                logger.error(f"Error generating caption for batch: {e}")
                raise
        return generated_text

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download the weights for the QwenVL model on the node."""
        model_dir_path = Path(model_dir) / _QWEN2_5_VL_MODEL_ID
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            return
        download_model_from_hf(
            model_id=_QWEN2_5_VL_MODEL_ID,
            local_dir=model_dir_path,
            revision=_QWEN2_5_VL_MODEL_REVISION,
        )
        logger.info(f"QwenVL weights downloaded to: {model_dir_path}")
