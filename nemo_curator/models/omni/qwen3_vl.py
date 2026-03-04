"""Qwen3-VL model implementations."""

from nemo_curator.models.omni.base import (
    VLLMModel,
    VLLMModelConfig,
)


# DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"


class Qwen3VL(VLLMModel):
    """Qwen3-VL model using vLLM backend."""

    def __init__(
        self,
        model_config: VLLMModelConfig,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        """Initialize Qwen3-VL with vLLM.

        Args:
            model_config: vLLM-specific configuration.
            model_id: HuggingFace model identifier.
        """
        model_config.extra_kwargs = {"limit_mm_per_prompt": {"image": 1}}
        super().__init__(model_id, model_config)
