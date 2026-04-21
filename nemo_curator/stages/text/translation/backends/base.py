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

"""
Abstract base class for non-LLM translation backends (Google, AWS, NMT).

Unlike Speaker's TranslationBackend which manages file I/O, this ABC operates
purely on lists of strings. It translates text in memory and returns results
directly, fitting Curator's DataFrame-based processing model.
"""

import asyncio
from abc import ABC, abstractmethod


class TranslationBackend(ABC):
    """Backend ABC for non-LLM translation (Google, AWS, NMT).

    Unlike Speaker's version, this does NOT do file I/O.
    It translates lists of strings and returns lists of strings.

    All subclasses must implement:
        - setup(): Initialize client connections and async infrastructure.
        - translate_batch(): Synchronous batch translation.
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

    @abstractmethod
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
        ...

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
