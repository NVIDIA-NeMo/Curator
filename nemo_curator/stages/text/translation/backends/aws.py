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
AWS Translate backend for NeMo Curator.

Uses Amazon Translate for translation.  The sync boto3 client is wrapped in
``asyncio.get_running_loop().run_in_executor()`` for async support.

Setup:
    Configure AWS credentials via one of:
    - AWS CLI: ``aws configure``
    - Environment variables: ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``
    - IAM role (EC2 / ECS / Lambda)

Dependencies:
    ``pip install boto3``

Notes:
    AWS Translate enforces a 10 000-byte UTF-8 limit per ``TranslateText``
    request.  Texts exceeding this limit raise ``ValueError`` -- callers
    should chunk upstream.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from ..utils.async_utils import run_async_safe
from ._retry import retry_with_backoff
from .base import TranslationBackend

# AWS Translate hard limit per TranslateText call (bytes, UTF-8).
AWS_MAX_BYTES_PER_REQUEST = 10_000


class AWSTranslationBackend(TranslationBackend):
    """AWS Translate backend.

    Args:
        region: AWS region.  Resolved in order: explicit value ->
            ``AWS_REGION`` env var -> ``AWS_DEFAULT_REGION`` env var ->
            ``"us-east-2"`` fallback.
        max_concurrent_requests: Semaphore size for async concurrency.
    """

    def __init__(
        self,
        region: str | None = None,
        max_concurrent_requests: int = 32,
    ) -> None:
        super().__init__(max_concurrent_requests=max_concurrent_requests)
        self._region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-2"
        )
        self._client = None  # Initialized in setup()

    # --------------------------------------------------------------------- #
    #  Lifecycle
    # --------------------------------------------------------------------- #

    def setup(self) -> None:
        """Initialize the boto3 Translate client.

        Raises:
            ImportError: If ``boto3`` is not installed.
        """
        super().setup()

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for the AWS backend: "
                "pip install boto3"
            )

        self._client = boto3.client(
            "translate",
            region_name=self._region,
        )
        logger.info(
            "AWS Translate client initialized (region={})",
            self._region,
        )

    def close(self) -> None:
        """Release client resources."""
        self._client = None

    def check_server(self) -> bool:
        """Check if the AWS Translate service is reachable.

        Performs a test translation of "Hello" to verify credentials and
        API access are configured correctly.

        Returns:
            True if the test translation succeeds, False otherwise.
        """
        try:
            result = self._translate_single_sync("Hello", "en", "es")
            if result:
                logger.info("AWS Translate health check passed")
                return True
            logger.warning("AWS Translate health check returned empty result")
            return False
        except Exception as exc:
            logger.warning(
                "AWS Translate health check failed: {}",
                exc,
            )
            return False

    # --------------------------------------------------------------------- #
    #  Synchronous interface
    # --------------------------------------------------------------------- #

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a batch of texts synchronously.

        Delegates to :meth:`translate_batch_async` through the safe async bridge.
        """
        return run_async_safe(
            lambda: self.translate_batch_async(texts, source_lang, target_lang)
        )

    # --------------------------------------------------------------------- #
    #  Asynchronous interface
    # --------------------------------------------------------------------- #

    async def translate_batch_async(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a batch of texts asynchronously.

        Creates one task per text, gated by the concurrency semaphore.
        """
        if not texts:
            return []

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        tasks = [
            self._translate_single_async(text, source_lang, target_lang)
            for text in texts
        ]
        return list(await asyncio.gather(*tasks))

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    async def _translate_single_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text with semaphore gating and retry logic.

        The boto3 client is synchronous, so the actual API call is wrapped
        in ``run_in_executor``.

        Raises:
            ValueError: If input text exceeds the AWS 10KB UTF-8 limit
                (non-retryable).
        """
        if not text or not text.strip():
            return ""

        loop = asyncio.get_running_loop()

        async def _attempt() -> str:
            async with self._semaphore:
                return await loop.run_in_executor(
                    None,
                    self._translate_single_sync,
                    text,
                    source_lang,
                    target_lang,
                )

        return await retry_with_backoff(
            _attempt, backend_name="AWS", non_retryable=(ValueError,)
        )

    def _translate_single_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Synchronous single-text translation (called via executor).

        Raises:
            ValueError: If the UTF-8 encoded text exceeds 10 000 bytes.
        """
        text_bytes = len(text.encode("utf-8"))
        if text_bytes > AWS_MAX_BYTES_PER_REQUEST:
            raise ValueError(
                f"AWS TranslateText input too large: {text_bytes} bytes "
                f"(UTF-8), limit is {AWS_MAX_BYTES_PER_REQUEST} bytes. "
                "Please chunk the input text before calling AWS Translate."
            )

        response = self._client.translate_text(
            Text=text,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang,
        )
        return response.get("TranslatedText", "")
