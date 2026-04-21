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
NMT (Neural Machine Translation) backend for NeMo Curator.

Communicates with an NMT server (e.g., IndicTrans2) over HTTP.  Unlike the
Google and AWS backends that translate one text per API call, the NMT backend
sends batches of texts in a single HTTP POST for higher throughput.

NMT API contract:
    POST ``{server_url}/translate``
    Request body::

        {"texts": [...], "src_lang": "en", "tgt_lang": "hi"}

    Response body::

        {"translations": [...]}

Dependencies:
    ``pip install aiohttp``  (for async HTTP; ``requests`` used for health check)
"""

from __future__ import annotations

import asyncio

from loguru import logger

from ._retry import retry_with_backoff
from .base import TranslationBackend


class NMTTranslationBackend(TranslationBackend):
    """NMT server backend with batched translation.

    Args:
        server_url: Base URL of the NMT server (e.g., ``"http://localhost:8000"``).
        batch_size: Number of texts per HTTP request.  Default 32.
        timeout: HTTP request timeout in seconds.  Default 120.
        max_concurrent_requests: Semaphore size for async concurrency.
    """

    def __init__(
        self,
        server_url: str,
        batch_size: int = 32,
        timeout: int = 120,
        max_concurrent_requests: int = 32,
    ) -> None:
        super().__init__(max_concurrent_requests=max_concurrent_requests)
        if not server_url or not server_url.strip():
            raise ValueError(
                "NMT backend requires a non-empty server_url. "
                "Example: server_url='http://localhost:8000'"
            )
        self._server_url = server_url.rstrip("/")
        self._batch_size = batch_size
        self._timeout = timeout
        self._session = None  # aiohttp.ClientSession, lazily created

    # --------------------------------------------------------------------- #
    #  Lifecycle
    # --------------------------------------------------------------------- #

    def setup(self) -> None:
        """Validate the server URL and optionally perform a health check.

        Raises:
            ImportError: If ``aiohttp`` is not installed.
        """
        super().setup()

        # Eagerly verify that aiohttp is importable so users get a clear
        # error at setup time rather than mid-translation.
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            raise ImportError(
                "aiohttp is required for the NMT backend: "
                "pip install aiohttp"
            )

        # Optional health check -- log but do not fail.
        self.check_server()

        logger.info(
            "NMT backend initialized: server_url={}, batch_size={}, "
            "timeout={}s, max_concurrent={}",
            self._server_url,
            self._batch_size,
            self._timeout,
            self.max_concurrent_requests,
        )

    def close(self) -> None:
        """Close the aiohttp session if open.

        Handles both cases:
        - Inside a running event loop: schedule close via
          ``loop.create_task`` so it is awaited by the running loop.
        - Outside an event loop: use ``asyncio.run`` to close synchronously.
        """
        if self._session is not None and not self._session.closed:
            try:
                loop = asyncio.get_running_loop()
                # Already inside a running loop -- schedule on it.  The
                # ``loop=`` kwarg was removed from asyncio APIs in 3.10.
                loop.create_task(self._session.close())
            except RuntimeError:
                # No running event loop -- actually await the close.
                asyncio.run(self._session.close())
            self._session = None
            logger.debug("NMT aiohttp session closed")

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

        Splits *texts* into sub-batches of ``self._batch_size``, posts each
        to the NMT server, and collects results.
        """
        return asyncio.run(
            self.translate_batch_async(texts, source_lang, target_lang)
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

        Splits *texts* into sub-batches and sends them concurrently, gated
        by the semaphore.
        """
        if not texts:
            return []

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Split into sub-batches.
        sub_batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        tasks = [
            self._translate_sub_batch(sub, source_lang, target_lang)
            for sub in sub_batches
        ]
        results = await asyncio.gather(*tasks)

        # Flatten the list of lists.
        return [text for batch_result in results for text in batch_result]

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    async def _get_session(self):
        """Lazily create or return the aiohttp session."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for the NMT backend: "
                "pip install aiohttp"
            )

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _translate_sub_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a single sub-batch with semaphore gating and retries."""
        session = await self._get_session()

        payload = {
            "texts": texts,
            "src_lang": source_lang,
            "tgt_lang": target_lang,
        }

        async def _attempt() -> list[str]:
            async with self._semaphore:
                async with session.post(
                    f"{self._server_url}/translate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            translations = result.get("translations", [])
            if len(translations) != len(texts):
                raise RuntimeError(
                    f"Translation count mismatch: sent {len(texts)} "
                    f"texts, received {len(translations)} translations "
                    "from NMT server."
                )
            return translations

        return await retry_with_backoff(_attempt, backend_name="NMT")

    def check_server(self) -> bool:
        """Check if the NMT server is reachable via its ``/health`` endpoint.

        Falls back to a plain GET to the server root URL if ``/health`` is
        not available. Uses synchronous ``requests`` for simplicity.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            import requests
        except ImportError:
            logger.debug(
                "requests not installed; skipping NMT health check"
            )
            return True  # Assume reachable if we cannot check

        try:
            resp = requests.get(
                f"{self._server_url}/health", timeout=10
            )
            resp.raise_for_status()
            logger.info("NMT server health check passed ({})", self._server_url)
            return True
        except Exception:
            try:
                resp = requests.get(self._server_url, timeout=10)
                logger.info(
                    "NMT server reachable at {} (no /health endpoint)",
                    self._server_url,
                )
                return True
            except Exception as exc:
                logger.warning(
                    "NMT server at {} is not reachable: {}. "
                    "Translation calls may fail.",
                    self._server_url,
                    exc,
                )
                return False
