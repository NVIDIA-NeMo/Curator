"""Base model classes for VLM inference."""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from loguru import logger
from PIL import Image


@dataclass(kw_only=True)
class InferenceConfig:
    """Configuration for inference."""

    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = False
    priority_mode: bool = False


@dataclass(kw_only=True)
class ModelConfig:
    """Model-specific configuration."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.gpu_memory_utilization < 0.0 or self.gpu_memory_utilization > 1.0:
            msg = "GPU memory utilization must be between 0.0 and 1.0"
            raise ValueError(msg)
        if self.tensor_parallel_size < 1:
            msg = "Tensor parallel size must be greater than 0"
            raise ValueError(msg)


class Model(ABC):
    """Abstract base class for models."""

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return False

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
        ...

    @abstractmethod
    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        self._model = None


class VLMModel(Model):
    """Abstract base class for vision-language models."""

    def __init__(self, model_id: str, model_config: ModelConfig) -> None:
        """Initialize VLM model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self.model_id = model_id
        self._model: Any = None
        self.model_config = model_config

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
        ...

    @abstractmethod
    def preload(self) -> None:
        """Preload the model into the local disk cache."""
        ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        """Generate responses for a batch of prompts, optionally with images.

        Args:
            prompts: List of text prompts.
            images: List of PIL images corresponding to prompts, or None for text-only.
            inference_config: Inference configuration.

        Returns:
            List of generated response strings.
        """
        ...

    def unload(self) -> None:
        """Unload model and free resources."""
        self._model = None


@dataclass(kw_only=True)
class NVInferenceModelConfig(ModelConfig):
    """NVIDIA Inference API-specific configuration."""

    base_url: str = "https://inference-api.nvidia.com"
    api_key_env_var: str = "NVINFERENCE_API_KEY"

    def __post_init__(self) -> None:
        pass


class NVInferenceModel(VLMModel):
    """Base class for models using NVIDIA Inference API."""

    model_config: NVInferenceModelConfig

    def __init__(
        self,
        model_id: str,
        model_config: NVInferenceModelConfig,
    ) -> None:
        """Initialize NVIDIA Inference API model.

        Args:
            model_id: Model identifier for NVIDIA Inference API.
            model_config: NVIDIA Inference API-specific configuration.
        """
        super().__init__(model_id, model_config)
        self._client: Any = None

    @property
    def is_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None

    def load(self) -> None:
        """Initialize the NVIDIA Inference API client."""
        if self.is_loaded:
            return

        from nemo_curator.models.client.nvinference_client import (
            create_openai_client,
            get_nvinference_api_key,
        )

        logger.info(f"Initializing NVIDIA Inference client for model {self.model_id}")
        api_key = get_nvinference_api_key(self.model_config.api_key_env_var)
        self._client = create_openai_client(api_key=api_key, base_url=self.model_config.base_url)
        logger.info("NVIDIA Inference client initialized")

    def preload(self) -> None:
        """Preload is not needed for API-based models."""
        logger.info(f"Preload not required for API-based model {self.model_id}")

    def unload(self) -> None:
        """Unload client and free resources."""
        self._client = None
        logger.info("NVIDIA Inference client unloaded")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64-encoded image string.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_message_content(self, prompt: str, image: Image.Image | str | None) -> list[dict[str, Any]]:
        """Build message content for OpenAI-compatible API.

        Args:
            prompt: Text prompt.
            image: Optional PIL Image or URI string.

        Returns:
            List of content dictionaries.
        """
        content: list[dict[str, Any]] = []

        content.append({"type": "text", "text": prompt})
        if image is not None:
            if isinstance(image, str):
                image_url = image
            else:
                image_b64 = self._encode_image_to_base64(image)
                image_url = f"data:image/png;base64,{image_b64}"

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return content

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        """Generate responses using NVIDIA Inference API.

        Args:
            prompts: List of text prompts.
            images: List of PIL images or URI strings corresponding to prompts, or None for text-only.
            inference_config: Inference configuration.

        Returns:
            List of generated response strings.
        """
        if not self.is_loaded:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        from nemo_curator.models.client.nvinference_client import stream_chat_completion_text

        results = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            content = self._build_message_content(prompt, image)
            messages = [{"role": "user", "content": content}]

            extra_headers = (
                {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"} if inference_config.priority_mode else None
            )

            try:
                response = stream_chat_completion_text(
                    self._client,
                    model=self.model_id,
                    messages=messages,
                    temperature=inference_config.temperature,
                    top_p=inference_config.top_p,
                    max_tokens=self.model_config.max_tokens or 2048,
                    extra_headers=extra_headers,
                )
                results.append(response if response else "")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error generating response for prompt {i}: {e}")
                results.append("")

        return results
