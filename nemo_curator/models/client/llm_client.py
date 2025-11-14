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
from dataclasses import dataclass

from loguru import logger


class ConversationFormatter(ABC):
    """
    Represents a way of formatting a conversation with an LLM
    such that it can response appropriately
    """

    @abstractmethod
    def format_conversation(self, conv: list[dict]) -> str:
        msg = "format_conversation must be implemented by subclasses"
        raise NotImplementedError(msg)


@dataclass
class GenerationConfig:
    """Configuration class for LLM generation parameters."""

    max_tokens: int | None = 2048
    n: int | None = 1
    seed: int | None = 0
    stop: str | None | list[str] = None
    stream: bool = False
    temperature: float | None = 0.0
    top_k: int | None = None
    top_p: float | None = 0.95


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
    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
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
        # Semaphore for controlling concurrent requests
        self._semaphore = None
        self._semaphore_loop = None

    @abstractmethod
    def setup(self) -> None:
        """
        Setup the client.
        """

    @abstractmethod
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
        Subclasses should implement this method instead of query_model.
        """
        msg = "Subclass of AsyncLLMClient must implement '_query_model_impl'"
        raise NotImplementedError(msg)

    async def query_model(  # noqa: C901, PLR0912
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        """
        Query the model with automatic retry and concurrency control.
        """
        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        # Initialize semaphore if not already done or if we're in a different event loop
        current_loop = asyncio.get_running_loop()
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
                    is_connection_error = (
                        "connection" in str(last_exception).lower()
                        or "ReadError" in str(last_exception)
                        or "BrokenResourceError" in str(last_exception)
                        or "APIConnectionError" in str(last_exception)
                        or "httpx.ReadError" in str(last_exception)
                    )

                    if is_rate_limit or is_connection_error:
                        if is_rate_limit:
                            logger.warning(
                                f"Rate limit error (429) detected. Attempt {attempt + 1}/{self.max_retries + 1}. Retrying in {self.base_delay * (2 ** (attempt - 1)):.1f}s..."
                            )
                        else:
                            logger.warning(
                                f"Connection error detected. Attempt {attempt + 1}/{self.max_retries + 1}. Retrying in {self.base_delay * (2 ** (attempt - 1)):.1f}s..."
                            )
                            logger.warning(f"Error details: {str(last_exception)[:200]}...")
                            if "localhost" in str(last_exception):
                                logger.warning(
                                    "Local API server issue - consider reducing --max-concurrent-requests or checking server resources"
                                )

                        # Exponential backoff with jitter
                        delay = self.base_delay * (2 ** (attempt - 1)) + secrets.randbelow(100) / 100.0
                        await asyncio.sleep(delay)
                    else:
                        # Re-raise if not a retryable error
                        raise last_exception

                # Attempt the query
                try:
                    return await self._query_model_impl(
                        messages=messages,
                        model=model,
                        conversation_formatter=conversation_formatter,
                        generation_config=generation_config,
                    )
                except Exception as e:
                    last_exception = e
                    # If this is the last attempt, provide helpful error message
                    if attempt == self.max_retries:
                        if "connection" in str(e).lower() or "ReadError" in str(e):
                            logger.error(f"Connection error after {self.max_retries + 1} attempts!")
                            logger.error(f"Final error: {str(e)[:200]}...")
                            if "localhost" in str(e):
                                logger.error("Suggestions for local API server:")
                                logger.error("- Check if server is running and has sufficient resources")
                                logger.error("- Reduce concurrent requests: --max-concurrent-requests 1")
                                logger.error("- Increase timeout: --timeout 900")
                                logger.error("- Check server logs for memory/GPU issues")
                        raise
                    # Otherwise, continue to next iteration
                    continue

            # This line should never be reached due to the raise in the except block
            # but if we get here, re-raise the last exception
            if last_exception:
                raise last_exception

            # This should never be reached, but add explicit return for linter
            logger.warning(
                "Unexpected code path: AsyncLLMClient.query_model completed without returning a result or raising an exception"
            )
            return []
