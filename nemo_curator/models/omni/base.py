"""NVIDIA Inference API client model.

Streaming is required because the non-stream response shape from reasoning
models on NVIDIA Inference (e.g. Nemotron-Nano-Omni-Reasoning) is not
deserialized cleanly by the OpenAI SDK; this wrapper reassembles the
streamed chunks into a single string.
"""

import asyncio
import base64
import concurrent.futures
import os
from io import BytesIO
from typing import Any

from loguru import logger
from PIL import Image

from nemo_curator.models.base import ModelInterface
from nemo_curator.models.client import AsyncOpenAIClient, OpenAIClient


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
        use_async: bool = True,
        max_concurrent_requests: int = 10,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key_env_var = api_key_env_var
        self.temperature = temperature
        self.top_p = top_p
        self.priority_mode = priority_mode
        self.use_async = use_async
        self.max_concurrent_requests = max_concurrent_requests
        self._client: OpenAIClient | AsyncOpenAIClient | None = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def setup(self) -> None:
        if self.is_loaded:
            return

        api_key = os.environ.get(self.api_key_env_var, "").strip()
        if not api_key:
            msg = f"{self.api_key_env_var} is not set"
            raise RuntimeError(msg)

        logger.info(f"Initializing NVIDIA Inference client for model {self.model_id} (async={self.use_async})")
        if self.use_async:
            self._client = AsyncOpenAIClient(
                base_url=self.base_url,
                api_key=api_key,
                max_concurrent_requests=self.max_concurrent_requests,
            )
        else:
            self._client = OpenAIClient(base_url=self.base_url, api_key=api_key)
        self._client.setup()
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

    def _stream_completion_text(self, messages: list[dict[str, Any]]) -> str:
        """Issue one streaming chat completion and reassemble ``delta.content`` into a string.

        Reasoning models on NV Inference split output: ``delta.content`` carries
        the final answer, ``delta.reasoning_content`` carries chain-of-thought.
        Reading only ``content`` keeps the thinking out of the parsed response
        without any prompt-side stripping.
        """
        extra_headers = {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"} if self.priority_mode else None
        completion = self._client.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
            extra_headers=extra_headers,
        )
        parts: list[str] = []
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta is not None:
                parts.append(delta)
        return "".join(parts)

    async def _astream_completion_text(self, messages: list[dict[str, Any]]) -> str:
        """Async twin of :meth:`_stream_completion_text` for ``AsyncOpenAIClient``."""
        extra_headers = {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"} if self.priority_mode else None
        completion = await self._client.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
            extra_headers=extra_headers,
        )
        parts: list[str] = []
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta is not None:
                parts.append(delta)
        return "".join(parts)

    def _messages_for(self, prompt: str, image: Image.Image | str | None) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self._build_message_content(prompt, image)}]

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None = None,
    ) -> list[str]:
        if not self.is_loaded:
            msg = "Model not loaded. Call setup() first."
            raise RuntimeError(msg)
        return self._generate_async(prompts, images) if self.use_async else self._generate_sync(prompts, images)

    def _generate_sync(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None,
    ) -> list[str]:
        results: list[str] = []
        for i, prompt in enumerate(prompts):
            image = images[i] if images else None
            try:
                results.append(self._stream_completion_text(self._messages_for(prompt, image)))
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error generating response for prompt {i}: {e}")
                results.append("")
        return results

    def _generate_async(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None,
    ) -> list[str]:
        """Fan prompts out concurrently.

        Uses ``asyncio.run`` in the normal case; if a loop is already running
        (e.g. inside a Ray async actor) we run in a dedicated thread with its
        own loop, mirroring ``QAMultilingualSyntheticStage``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._agenerate_all(prompts, images))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, self._agenerate_all(prompts, images)).result()

    async def _agenerate_all(
        self,
        prompts: list[str],
        images: list[Image.Image | str] | None,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def _one(i: int, prompt: str) -> str:
            image = images[i] if images else None
            async with semaphore:
                try:
                    return await self._astream_completion_text(self._messages_for(prompt, image))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error generating response for prompt {i}: {e}")
                    return ""

        return list(await asyncio.gather(*(_one(i, prompt) for i, prompt in enumerate(prompts))))
