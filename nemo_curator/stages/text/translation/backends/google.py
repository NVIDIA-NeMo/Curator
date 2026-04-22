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
Google Cloud Translation API backend for NeMo Curator.

Supports both Google Cloud Translation API v2 (Basic) and v3 (Advanced).

Setup:
    1. Create a Google Cloud project and enable the Cloud Translation API.
    2. Create a service account and download the JSON key file.
    3. Set the environment variable:
       ``export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json``

Dependencies:
    ``pip install google-cloud-translate``
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from ..utils.async_utils import run_async_safe
from ._retry import retry_with_backoff
from .base import TranslationBackend


class GoogleTranslationBackend(TranslationBackend):
    """Google Cloud Translation backend (v2 and v3 APIs).

    Args:
        project_id: Google Cloud project ID.  Required for v3.  Falls back
            to the ``GOOGLE_CLOUD_PROJECT`` environment variable.
        location: Google Cloud location for v3 (default ``"global"``).
        api_version: ``"v2"`` (default) or ``"v3"``.
        max_concurrent_requests: Semaphore size for async concurrency.
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "global",
        api_version: str = "v2",
        max_concurrent_requests: int = 32,
    ) -> None:
        super().__init__(max_concurrent_requests=max_concurrent_requests)
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location
        self._api_version = api_version.lower()
        self._client = None  # Initialized in setup()
        self._parent: str | None = None  # v3 resource path

    # --------------------------------------------------------------------- #
    #  Lifecycle
    # --------------------------------------------------------------------- #

    def setup(self) -> None:
        """Initialize the Google Cloud Translation client.

        Raises:
            ImportError: If ``google-cloud-translate`` is not installed.
            ValueError: If v3 is selected but no *project_id* is available.
        """
        super().setup()

        try:
            if self._api_version == "v3":
                from google.cloud import translate_v3 as translate
            else:
                from google.cloud import translate_v2 as translate
        except ImportError:
            raise ImportError(
                "Google Cloud Translate required: "
                "pip install google-cloud-translate"
            )

        if self._api_version == "v3":
            self._client = translate.TranslationServiceClient()
            if not self._project_id:
                raise ValueError(
                    "project_id is required for Google Translation API v3. "
                    "Set project_id in backend_config or the "
                    "GOOGLE_CLOUD_PROJECT environment variable."
                )
            self._parent = (
                f"projects/{self._project_id}/locations/{self._location}"
            )
        else:
            self._client = translate.Client()
            self._parent = None

        logger.info(
            "Google Cloud Translation client initialized "
            "(api_version={}, project={})",
            self._api_version,
            self._project_id or "none (v2 mode)",
        )

    def close(self) -> None:
        """Release client resources."""
        self._client = None

    def check_server(self) -> bool:
        """Check if the Google Cloud Translation service is reachable.

        Performs a test translation of "Hello" to verify credentials and
        API access are configured correctly.

        Returns:
            True if the test translation succeeds, False otherwise.
        """
        try:
            result = self._translate_single_sync("Hello", "en", "es")
            if result:
                logger.info("Google Cloud Translation health check passed")
                return True
            logger.warning("Google Cloud Translation health check returned empty result")
            return False
        except Exception as exc:
            logger.warning(
                "Google Cloud Translation health check failed: {}",
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

        The Google client is synchronous, so the actual API call is wrapped
        in ``run_in_executor``.

        Non-retryable errors (400-class: invalid arguments, permission
        denied, etc.) short-circuit the retry loop so misconfigurations
        surface immediately instead of hanging on exponential backoff.
        """
        if not text or not text.strip():
            return ""

        loop = asyncio.get_running_loop()

        # Import google.api_core exceptions lazily so the module still imports
        # cleanly when google-cloud-translate is not installed.
        try:
            from google.api_core import exceptions as gcp_exc

            _NON_RETRYABLE: tuple[type[BaseException], ...] = (
                gcp_exc.InvalidArgument,
                gcp_exc.PermissionDenied,
                gcp_exc.NotFound,
                gcp_exc.Unauthenticated,
                gcp_exc.FailedPrecondition,
            )
        except ImportError:
            _NON_RETRYABLE = ()

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
            _attempt, backend_name="Google", non_retryable=_NON_RETRYABLE
        )

    def _translate_single_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Synchronous single-text translation (called via executor)."""
        if self._api_version == "v3":
            response = self._client.translate_text(
                parent=self._parent,
                contents=[text],
                source_language_code=source_lang,
                target_language_code=target_lang,
                mime_type="text/plain",
            )
            return response.translations[0].translated_text
        else:
            # v2 API
            result = self._client.translate(
                text,
                source_language=source_lang,
                target_language=target_lang,
                format_="text",
            )
            return result["translatedText"]
