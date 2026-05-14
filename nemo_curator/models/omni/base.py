"""NVIDIA Inference API client model.

Implements the codebase's :class:`ModelInterface` contract (``setup()`` +
``model_id_names``). All sampling and connection kwargs live on the instance;
``generate`` takes only the per-call inputs.
"""

import base64
from io import BytesIO
from typing import Any

from loguru import logger
from PIL import Image

from nemo_curator.models.base import ModelInterface


class NVInferenceModel(ModelInterface):
    """Wraps an OpenAI-compatible chat-completion endpoint on NVIDIA Inference API."""

    def __init__(  # noqa: PLR0913
        self,
        model_id: str,
        *,
        max_tokens: int = 2048,
        base_url: str = "https://inference-api.nvidia.com",
        api_key_env_var: str = "NVINFERENCE_API_KEY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        priority_mode: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key_env_var = api_key_env_var
        self.temperature = temperature
        self.top_p = top_p
        self.priority_mode = priority_mode
        self._client: Any = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def setup(self) -> None:
        if self.is_loaded:
            return

        from nemo_curator.models.client.nvinference_client import (
            create_openai_client,
            get_nvinference_api_key,
        )

        logger.info(f"Initializing NVIDIA Inference client for model {self.model_id}")
        api_key = get_nvinference_api_key(self.api_key_env_var)
        self._client = create_openai_client(api_key=api_key, base_url=self.base_url)
        logger.info("NVIDIA Inference client initialized")

    def unload(self) -> None:
        self._client = None
        logger.info("NVIDIA Inference client unloaded")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_message_content(self, prompt: str, image: Image.Image | str | None) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        if image is not None:
            if isinstance(image, str):
                image_url = image
            else:
                image_b64 = self._encode_image_to_base64(image)
                image_url = f"data:image/png;base64,{image_b64}"

            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": prompt})
        return content

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None = None,
    ) -> list[str]:
        if not self.is_loaded:
            msg = "Model not loaded. Call setup() first."
            raise RuntimeError(msg)

        from nemo_curator.models.client.nvinference_client import stream_chat_completion_text

        extra_headers = {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"} if self.priority_mode else None

        results = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            content = self._build_message_content(prompt, image)
            messages = [{"role": "user", "content": content}]

            try:
                response = stream_chat_completion_text(
                    self._client,
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_headers=extra_headers,
                )
                results.append(response if response else "")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error generating response for prompt {i}: {e}")
                results.append("")

        return results
