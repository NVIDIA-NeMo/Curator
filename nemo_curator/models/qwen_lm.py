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

_QWEN_LM_VARIANTS: dict[str, dict] = {
    "qwen": {"model_id": "Qwen/Qwen3.5-14B", "revision": None},
    "qwen3.5": {"model_id": "Qwen/Qwen3.5-14B", "revision": None},
    "qwen3": {"model_id": "Qwen/Qwen3-14B", "revision": "8268fe3"},
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-14B-Instruct", "revision": "cf98f3b"},
}


class QwenLM(ModelInterface):
    """Qwen language model."""

    DEFAULT_MODEL_ID = "Qwen/Qwen3.5-14B"
    DEFAULT_MODEL_REVISION = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    def __init__(  # noqa: PLR0913
        self,
        model_dir: str = "",
        caption_batch_size: int = 1,
        fp8: bool = False,
        max_output_tokens: int = 512,
        model_variant: str = "qwen",
        model_id: str | None = None,
        model_revision: str | None = None,
        **vllm_kwargs,
    ):
        variant_info = _QWEN_LM_VARIANTS.get(model_variant, {})
        model_id = model_id or variant_info.get("model_id") or self.DEFAULT_MODEL_ID
        model_revision = model_revision or variant_info.get("revision") or self.DEFAULT_MODEL_REVISION
        if not (Path(model_id).is_absolute() or model_id.startswith(("./", "../"))) and not model_id.startswith(
            "Qwen/"
        ):
            msg = f"model_id '{model_id}' is not a Qwen model. For a local path use an absolute path or './' prefix."
            raise ValueError(msg)
        self.model_dir = model_dir
        self.caption_batch_size = caption_batch_size
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.model_id = model_id
        self.model_revision = model_revision
        self.vllm_kwargs = vllm_kwargs

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for QwenLM model but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        self.weight_file = str(Path(self.model_dir) / self.model_id)
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
    def download_weights_on_node(
        cls, model_dir: str, model_id: str | None = None, model_revision: str | None = None
    ) -> None:
        """Download the weights for the QwenLM model on the node."""
        model_id = model_id or cls.DEFAULT_MODEL_ID
        model_revision = model_revision or cls.DEFAULT_MODEL_REVISION
        model_dir_path = Path(model_dir) / model_id
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            logger.info(f"QwenLM weights already exist at: {model_dir_path}")
            return
        download_model_from_hf(
            model_id=model_id,
            local_dir=model_dir_path,
            revision=model_revision,
        )
        logger.info(f"QwenLM weights downloaded to: {model_dir_path}")
