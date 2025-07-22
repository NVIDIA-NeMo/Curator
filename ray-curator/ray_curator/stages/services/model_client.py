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
import asyncio
import secrets
from abc import ABC, abstractmethod
from collections.abc import Iterable

from ray_curator.stages.services.conversation_formatter import ConversationFormatter


class LLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests synchronously
    """

    @abstractmethod
    def setup(self) -> None:
        """
        Setup the client.
        """

    @abstractmethod
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
        _timeout: float | None = None,
        _presence_penalty: float | None = 1.0,
        _logprobs: bool = False,
        _top_logprobs: int | None = None,
    ) -> list[str]:
        msg = "Subclass of LLMClient must implement 'query_model'"
        raise NotImplementedError(msg)


class AsyncLLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests asynchronously
    """

    def __init__(self, max_concurrent_requests: int = 5, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize the async client with concurrency and retry settings.

        Args:
            max_concurrent_requests: Maximum number of concurrent requests
            max_retries: Maximum number of retry attempts for rate-limited requests
            base_delay: Base delay for exponential backoff (in seconds)
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._semaphore = None
        self._semaphore_loop = None

    @abstractmethod
    def setup(self) -> None:
        """
        Setup the client.
        """

    @abstractmethod
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
        Subclasses should implement this method instead of query_model.
        """
        msg = "Subclass of AsyncLLMClient must implement '_query_model_impl'"
        raise NotImplementedError(msg)

    async def query_model(  # noqa: PLR0913
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
        timeout: float | None = None,
        presence_penalty: float | None = 1.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> list[str]:
        """
        Query the model with automatic retry and concurrency control.
        """
        # Initialize semaphore if not already done or if we're in a different event loop
        current_loop = asyncio.get_event_loop()
        if self._semaphore is None or self._semaphore_loop != current_loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self._semaphore_loop = current_loop

        async with self._semaphore:  # Limit concurrent requests
            # Retry logic with exponential backoff
            last_exception = None

            for attempt in range(self.max_retries + 1):
                # Check if this is a retry attempt and if we should delay
                if attempt > 0 and last_exception:
                    is_rate_limit = "429" in str(last_exception) or "rate" in str(last_exception).lower()
                    if is_rate_limit:
                        print(f"⚠️  WARNING: Rate limit error (429) detected. Attempt {attempt + 1}/{self.max_retries + 1}. Retrying in {self.base_delay * (2 ** (attempt - 1)):.1f}s...")
                        # Exponential backoff with jitter
                        delay = self.base_delay * (2 ** (attempt - 1)) + secrets.randbelow(100) / 100.0
                        await asyncio.sleep(delay)
                    else:
                        # Re-raise if not a rate limit error
                        raise last_exception

                # Attempt the query
                try:
                    return await self._query_model_impl(
                        messages=messages,
                        model=model,
                        conversation_formatter=conversation_formatter,
                        max_tokens=max_tokens,
                        n=n,
                        seed=seed,
                        stop=stop,
                        stream=stream,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        timeout=timeout,
                        presence_penalty=presence_penalty,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                    )
                except Exception as e:
                    last_exception = e
                    # If this is the last attempt, re-raise
                    if attempt == self.max_retries:
                        raise
                    # Otherwise, continue to next iteration
                    continue

            # This line should never be reached due to the raise in the except block
            # but if we get here, re-raise the last exception
            if last_exception:
                raise last_exception
            return []
