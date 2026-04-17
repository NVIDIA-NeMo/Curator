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
from pathlib import Path
from typing import Any, Final

from loguru import logger
from transformers import AutoTokenizer

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

_QWEN_LM_VARIANTS_INFO: Final = {
    "qwen2.5": ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b"),
    "qwen3": ("Qwen/Qwen3-14B", "f8c293d"),
}


def _weights_complete(model_dir_path: Path) -> bool:
    single_file = model_dir_path / "model.safetensors"
    if single_file.exists():
        return True
    index_file = model_dir_path / "model.safetensors.index.json"
    if not index_file.exists():
        return False
    try:
        with open(index_file) as f:
            index_data = json.load(f)
        shard_names = set(index_data.get("weight_map", {}).values())
        return bool(shard_names) and all((model_dir_path / s).exists() for s in shard_names)
    except (json.JSONDecodeError, OSError):
        return False


class QwenLM(ModelInterface):
    """Qwen language model."""

    def model_id_names(self) -> list[str]:
        model_id, _ = _QWEN_LM_VARIANTS_INFO[self.model_variant]
        return [model_id]

    def __init__(
        self,
        model_dir: str = "",
        model_variant: str = "qwen2.5",
        caption_batch_size: int = 1,
        fp8: bool = False,
        max_output_tokens: int = 512,
        **vllm_kwargs,
    ):
        self.model_dir = model_dir
        self.model_variant = model_variant
        self.caption_batch_size = caption_batch_size
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.vllm_kwargs = vllm_kwargs

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for QwenLM model but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        model_id, _ = _QWEN_LM_VARIANTS_INFO[self.model_variant]
        self.weight_file = str(Path(self.model_dir) / model_id)
        self.llm = LLM(
            model=self.weight_file,
            quantization="fp8" if self.fp8 else None,
            **self.vllm_kwargs,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_file)

    def generate(self, inputs: list[dict[str, Any]]) -> list[str]:
        formatted_inputs = self.tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)
        results = self.llm.generate(formatted_inputs, sampling_params=self.sampling_params)
        return [result.outputs[0].text for result in results]

    @classmethod
    def download_weights_on_node(cls, model_dir: str, variant: str = "qwen2.5") -> None:
        """Download the weights for the QwenLM model on the node."""
        model_id, revision = _QWEN_LM_VARIANTS_INFO[variant]
        model_dir_path = Path(model_dir) / model_id
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if _weights_complete(model_dir_path):
            return
        download_model_from_hf(
            model_id=model_id,
            local_dir=model_dir_path,
            revision=revision,
        )
        logger.info(f"QwenLM weights downloaded to: {model_dir_path}")
