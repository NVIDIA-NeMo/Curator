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
from transformers import AutoConfig, AutoTokenizer

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

_QWEN_LM_MODEL_ID = "Qwen/Qwen3-14B"
_QWEN_LM_MODEL_REVISION = None


def _validate_qwen_model(model_name: str) -> None:
    if not model_name.startswith("Qwen/"):
        msg = f"model_name must be a Qwen model (start with 'Qwen/'). Got: '{model_name}'"
        raise ValueError(msg)


def _check_vllm_supports_model(model_name: str) -> None:
    if not VLLM_AVAILABLE:
        return
    config = AutoConfig.from_pretrained(model_name)
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        return
    try:
        from vllm.model_executor.models import ModelRegistry

        unsupported = [arch for arch in architectures if not ModelRegistry.is_model_supported(arch)]
        if len(unsupported) == len(architectures):
            msg = f"Model '{model_name}' has architecture(s) {architectures} not supported by vLLM"
            raise ValueError(msg)
    except ImportError:
        pass  # vLLM registry not accessible, skip check


class QwenLM(ModelInterface):
    """Qwen language model."""

    def model_id_names(self) -> list[str]:
        return [self.model_name]

    def __init__(  # noqa: PLR0913
        self,
        model_dir: str,
        caption_batch_size: int,
        fp8: bool,
        max_output_tokens: int,
        model_name: str = _QWEN_LM_MODEL_ID,
        model_revision: str | None = _QWEN_LM_MODEL_REVISION,
    ):
        _validate_qwen_model(model_name)
        self.model_dir = model_dir
        self.caption_batch_size = caption_batch_size
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.model_name = model_name
        self.model_revision = model_revision

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for QwenLM model but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        _check_vllm_supports_model(self.model_name)

        self.weight_file = str(Path(self.model_dir) / self.model_name)
        weight_path = Path(self.weight_file)
        if not weight_path.exists() or not any(weight_path.glob("*.safetensors")):
            weight_path.mkdir(parents=True, exist_ok=True)
            download_model_from_hf(model_id=self.model_name, local_dir=weight_path, revision=self.model_revision)

        self.llm = LLM(
            model=self.weight_file,
            quantization="fp8" if self.fp8 else None,
            enforce_eager=False,
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
        cls,
        model_dir: str,
        model_name: str = _QWEN_LM_MODEL_ID,
        model_revision: str | None = _QWEN_LM_MODEL_REVISION,
    ) -> None:
        """Download the weights for the QwenLM model on the node."""
        _validate_qwen_model(model_name)
        model_dir_path = Path(model_dir) / model_name
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            return
        download_model_from_hf(model_id=model_name, local_dir=model_dir_path, revision=model_revision)
        logger.info(f"QwenLM weights downloaded to: {model_dir_path}")
