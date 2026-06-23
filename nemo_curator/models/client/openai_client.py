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

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from nemo_curator.models.client.llm_client import AsyncLLMClient, ConversationFormatter, GenerationConfig, LLMClient


@dataclass(frozen=True)
class OpenAIChatCompletionResult:
    """OpenAI-compatible chat completion content and aggregate usage."""

    contents: list[str]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class OpenAIClient(LLMClient):
    """
    A wrapper around OpenAI's Python client for querying models
    """

    def __init__(self, **kwargs) -> None:
        # Extract timeout if provided, default to 120 for backward compatibility
        self.timeout = kwargs.pop("timeout", 120)
        self.openai_kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the client.
        """
        self.client = OpenAI(**self.openai_kwargs)

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        return self.query_model_with_usage(
            messages=messages,
            model=model,
            conversation_formatter=conversation_formatter,
            generation_config=generation_config,
        ).contents

    def query_model_with_usage(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> OpenAIChatCompletionResult:
        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an OpenAIClient", stacklevel=2)

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        if generation_config.top_k is not None:
            warnings.warn("top_k is not used in an OpenAIClient", stacklevel=2)

        create_kwargs = {
            "messages": messages,
            "model": model,
            "max_tokens": generation_config.max_tokens,
            "n": generation_config.n,
            "seed": generation_config.seed,
            "stop": generation_config.stop,
            "stream": generation_config.stream,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "timeout": self.timeout,
        }
        if generation_config.extra_kwargs:
            overlapping = set(generation_config.extra_kwargs) & set(create_kwargs)
            if overlapping:
                logger.warning(f"extra_kwargs will overwrite existing parameter(s): {overlapping}")
            create_kwargs.update(generation_config.extra_kwargs)

        if not hasattr(self, "client"):
            self.setup()

        response = self.client.chat.completions.create(**create_kwargs)

        return _completion_result_from_response(response)


class AsyncOpenAIClient(AsyncLLMClient):
    """
    A wrapper around OpenAI's Python async client for querying models
    """

    def __init__(
        self, max_concurrent_requests: int = 5, max_retries: int = 3, base_delay: float = 1.0, **kwargs
    ) -> None:
        """
        Initialize the AsyncOpenAI client.

        Args:
            max_concurrent_requests: Maximum number of concurrent requests
            max_retries: Maximum number of retry attempts for rate-limited requests
            base_delay: Base delay for exponential backoff (in seconds)
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(max_concurrent_requests, max_retries, base_delay)
        # Extract timeout if provided, default to 120 for backward compatibility
        self.timeout = kwargs.pop("timeout", 120)
        self.openai_kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the client.
        """
        self.client = AsyncOpenAI(**self.openai_kwargs)

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        """
        Internal implementation of query_model without retry/concurrency logic.
        """
        result = await self._query_model_with_usage_impl(
            messages=messages,
            model=model,
            conversation_formatter=conversation_formatter,
            generation_config=generation_config,
        )
        return result.contents

    async def _query_model_with_usage_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> OpenAIChatCompletionResult:
        """
        Internal implementation of query_model_with_usage without retry/concurrency logic.
        """
        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an AsyncOpenAIClient", stacklevel=2)

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        if generation_config.top_k is not None:
            warnings.warn("top_k is not used in an AsyncOpenAIClient", stacklevel=2)

        create_kwargs = {
            "messages": messages,
            "model": model,
            "max_tokens": generation_config.max_tokens,
            "n": generation_config.n,
            "seed": generation_config.seed,
            "stop": generation_config.stop,
            "stream": generation_config.stream,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "timeout": self.timeout,
        }
        if generation_config.extra_kwargs:
            overlapping = set(generation_config.extra_kwargs) & set(create_kwargs)
            if overlapping:
                logger.warning(f"extra_kwargs will overwrite existing parameter(s): {overlapping}")
            create_kwargs.update(generation_config.extra_kwargs)

        if not hasattr(self, "client"):
            self.setup()

        response = await self.client.chat.completions.create(**create_kwargs)

        return _completion_result_from_response(response)

    async def query_model_with_usage(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> OpenAIChatCompletionResult:
        """
        Query the model and keep OpenAI-compatible usage counters when the server returns them.
        """
        generation_config = self._coerce_generation_config(generation_config)
        return await self._run_with_retry_and_concurrency(
            lambda: self._query_model_with_usage_impl(
                messages=messages,
                model=model,
                conversation_formatter=conversation_formatter,
                generation_config=generation_config,
            )
        )


def _completion_result_from_response(response: Any) -> OpenAIChatCompletionResult:  # noqa: ANN401
    usage = getattr(response, "usage", None)
    return OpenAIChatCompletionResult(
        contents=[choice.message.content for choice in response.choices],
        prompt_tokens=_usage_int(usage, "prompt_tokens"),
        completion_tokens=_usage_int(usage, "completion_tokens"),
        total_tokens=_usage_int(usage, "total_tokens"),
    )


def _usage_int(usage: Any, field: str) -> int | None:  # noqa: ANN401
    if usage is None:
        return None
    value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
