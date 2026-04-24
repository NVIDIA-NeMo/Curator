# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Base classes for non-LLM translation backends."""

import asyncio
from abc import ABC, abstractmethod

from loguru import logger

from ..utils.async_utils import run_async_safe
from ._retry import retry_with_backoff


class TranslationBackend(ABC):
    """Backend ABC for non-LLM translation (Google, AWS, NMT).

    This interface operates on in-memory text lists and returns translated
    text lists. It does not manage file I/O.

    All subclasses must implement:
        - setup(): Initialize client connections and async infrastructure.
        - translate_batch_async(): Asynchronous batch translation.
        - check_server(): Verify backend service is available.
        - close(): Cleanup resources (optional override).

    Constructor Parameters:
        max_concurrent_requests: Maximum number of concurrent translation
            requests. Controls the asyncio.Semaphore size. Default 32.
    """

    def __init__(self, max_concurrent_requests: int = 32) -> None:
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore: asyncio.Semaphore | None = None

    @abstractmethod
    def setup(self) -> None:
        """Initialize client connections.

        Subclasses should call ``super().setup()`` for any future base-class
        initialization.  The concurrency semaphore is created lazily inside
        ``translate_batch_async()`` so that it always belongs to the correct
        event loop.
        """
        pass

    @abstractmethod
    def check_server(self) -> bool:
        """Check if the translation server/service is available.

        Each backend implements its own health check logic:
        - Google: test translate "Hello"
        - AWS: test translate "Hello"
        - NMT: GET to ``/health`` endpoint

        Returns:
            True if backend is reachable/healthy, False otherwise.
        """
        ...

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a batch of texts synchronously.

        Args:
            texts: Source texts to translate.
            source_lang: ISO 639-1 source language code.
            target_lang: ISO 639-1 target language code.

        Returns:
            Translated texts in the same order as input.
        """
        return run_async_safe(
            lambda: self.translate_batch_async(texts, source_lang, target_lang)
        )

    @abstractmethod
    async def translate_batch_async(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a batch of texts asynchronously.

        Args:
            texts: Source texts to translate.
            source_lang: ISO 639-1 source language code.
            target_lang: ISO 639-1 target language code.

        Returns:
            Translated texts in the same order as input.
        """
        ...

    def close(self) -> None:
        """Cleanup resources (e.g., close HTTP sessions, API clients).

        Override in subclasses that hold open connections.
        """
        pass

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the per-backend semaphore, creating it lazily per event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._semaphore


class ExecutorTranslationBackend(TranslationBackend):
    """Common base for backends with a synchronous single-text SDK call.

    AWS Translate and Google Cloud Translate both expose synchronous client
    methods. This class centralizes the common async bridge, retry wrapper,
    and lightweight health check so those backends only define setup and the
    actual single-text translation call.
    """

    backend_name: str = "backend"
    health_check_text: str = "Hello"
    health_check_source_lang: str = "en"
    health_check_target_lang: str = "es"

    def check_server(self) -> bool:
        """Check backend reachability with a tiny translation request."""
        try:
            result = self._translate_single_sync(
                self.health_check_text,
                self.health_check_source_lang,
                self.health_check_target_lang,
            )
        except Exception as exc:
            logger.warning("{} health check failed: {}", self.backend_name, exc)
            return False

        if result:
            logger.info("{} health check passed", self.backend_name)
            return True

        logger.warning("{} health check returned empty result", self.backend_name)
        return False

    async def translate_batch_async(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate texts concurrently via the sync single-text SDK call."""
        if not texts:
            return []

        tasks = [
            self._translate_single_async(text, source_lang, target_lang)
            for text in texts
        ]
        return list(await asyncio.gather(*tasks))

    async def _translate_single_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text using an executor-backed sync SDK call."""
        if not text or not text.strip():
            return ""

        loop = asyncio.get_running_loop()
        semaphore = self._get_semaphore()

        async def _attempt() -> str:
            async with semaphore:
                return await loop.run_in_executor(
                    None,
                    self._translate_single_sync,
                    text,
                    source_lang,
                    target_lang,
                )

        return await retry_with_backoff(
            _attempt,
            backend_name=self.backend_name,
            non_retryable=self._non_retryable_exceptions(),
        )

    def _non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return exception types that should bypass retry/backoff."""
        return ()

    @abstractmethod
    def _translate_single_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate one text synchronously."""
        ...
