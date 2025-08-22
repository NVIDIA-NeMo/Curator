from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

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


from ray_curator.models.base import ModelInterface

_QWEN_LM_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


class QwenLM(ModelInterface):
    """Qwen language model."""

    def model_id_names(self) -> list[str]:
        return [_QWEN_LM_MODEL_ID]

    def __init__(self, model_dir: str, caption_batch_size: int, fp8: bool, max_output_tokens: int):
        self.model_dir = model_dir
        self.caption_batch_size = caption_batch_size
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens

    def setup(self) -> None:
        if not VLLM_AVAILABLE:
            msg = "vllm is required for QwenLM model but is not installed. Please install vllm: pip install vllm"
            raise ImportError(msg)

        self.weight_file = str(Path(self.model_dir) / _QWEN_LM_MODEL_ID)
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
