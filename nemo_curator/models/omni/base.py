"""NVIDIA Inference API client for reasoning VLMs.

Reasoning models on NVIDIA Inference (e.g. Nemotron-Nano-Omni-Reasoning) split
their output into ``delta.reasoning_content`` (chain-of-thought) and
``delta.content`` (the final answer), and their non-stream response shape is not
deserialized cleanly by the OpenAI SDK. This client therefore streams and
reassembles only ``delta.content``.
"""

import os
from collections.abc import Iterable

from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import ConversationFormatter, GenerationConfig

_PRIORITY_HEADER = {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}


class NVInferenceClient(AsyncOpenAIClient):
    """Async OpenAI-compatible client that streams reasoning-model output.

    Resolves the API key from ``api_key_env_var`` at ``setup()`` time (so the
    key is read on the worker, not serialized from the driver), then reassembles
    ``delta.content`` from a streaming completion.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key_env_var: str = "NVINFERENCE_API_KEY",
        priority_mode: bool = False,
        max_concurrent_requests: int = 10,
        timeout: int = 120,
    ) -> None:
        super().__init__(max_concurrent_requests=max_concurrent_requests, base_url=base_url, timeout=timeout)
        self.api_key_env_var = api_key_env_var
        self.priority_mode = priority_mode

    def setup(self) -> None:
        if getattr(self, "client", None) is not None:
            return
        api_key = os.environ.get(self.api_key_env_var, "").strip()
        if not api_key:
            msg = f"{self.api_key_env_var} is not set"
            raise RuntimeError(msg)
        self.openai_kwargs["api_key"] = api_key
        super().setup()

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,  # noqa: ARG002
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        if not hasattr(self, "client"):
            self.setup()

        create_kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_tokens,
            "stop": generation_config.stop,
            "stream": True,
            "timeout": self.timeout,
        }
        if self.priority_mode:
            create_kwargs["extra_headers"] = _PRIORITY_HEADER
        # extra_kwargs wins on overlap, matching AsyncOpenAIClient.
        if generation_config.extra_kwargs:
            create_kwargs.update(generation_config.extra_kwargs)

        completion = await self.client.chat.completions.create(**create_kwargs)
        parts: list[str] = []
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta is not None:
                parts.append(delta)
        return ["".join(parts)]
