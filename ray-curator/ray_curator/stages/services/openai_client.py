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

from openai import AsyncOpenAI, OpenAI

from ray_curator.stages.services.conversation_formatter import ConversationFormatter

from .model_client import AsyncLLMClient, LLMClient


class OpenAIClient(LLMClient):
    """
    A wrapper around OpenAI's Python client for querying models
    """

    def __init__(self, **kwargs) -> None:
        # Extract timeout if provided, default to 120 for backward compatibility
        self.timeout = kwargs.pop('timeout', 120)
        self.openai_kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the client.
        """
        self.client = OpenAI(**self.openai_kwargs)

    def query_model(  # noqa: PLR0913
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        max_tokens: int | None = 2048,
        n: int | None = 1,
        seed: int | None = 0,
        stop: str | None | list[str] = None,
        stream: bool = False,
        temperature: float | None = 0.0,
        top_k: int | None = None,
        top_p: float | None = 0.95,
    ) -> list[str]:
        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an OpenAIClient", stacklevel=2)
        if top_k is not None:
            warnings.warn("top_k is not used in an OpenAIClient", stacklevel=2)

        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            timeout=self.timeout,
        )

        return [choice.message.content for choice in response.choices]


class AsyncOpenAIClient(AsyncLLMClient):
    """
    A wrapper around OpenAI's Python async client for querying models
    """

    def __init__(self, max_concurrent_requests: int = 5, max_retries: int = 3, base_delay: float = 1.0, **kwargs) -> None:
        """
        Initialize the AsyncOpenAIClient.

        Args:
            max_concurrent_requests: Maximum number of concurrent requests
            max_retries: Maximum number of retry attempts for rate-limited requests
            base_delay: Base delay for exponential backoff (in seconds)
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(max_concurrent_requests, max_retries, base_delay)
        # Extract timeout if provided, default to 120 for backward compatibility
        self.timeout = kwargs.pop('timeout', 120)
        self.openai_kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the client.
        """
        self.client = AsyncOpenAI(**self.openai_kwargs)

    async def _query_model_impl(  # noqa: PLR0913
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        max_tokens: int | None = 2048,
        n: int | None = 1,
        seed: int | None = 0,
        stop: str | None | list[str] = None,
        stream: bool = False,
        temperature: float | None = 0.0,
        top_k: int | None = None,
        top_p: float | None = 0.95,
    ) -> list[str]:
        """
        Internal implementation of query_model without retry/concurrency logic.
        """
        if conversation_formatter is not None:
            warnings.warn("conversation_formatter is not used in an AsyncOpenAIClient", stacklevel=2)
        if top_k is not None:
            warnings.warn("top_k is not used in an AsyncOpenAIClient", stacklevel=2)

        response = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            timeout=self.timeout,
        )

        return [choice.message.content for choice in response.choices]
