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

"""Unit tests for translation backends (base, factory, Google, AWS, NMT)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_curator.stages.text.translation.backends import get_backend
from nemo_curator.stages.text.translation.backends.aws import (
    AWS_MAX_BYTES_PER_REQUEST,
    AWSTranslationBackend,
)
from nemo_curator.stages.text.translation.backends.base import TranslationBackend
from nemo_curator.stages.text.translation.backends.google import (
    GoogleTranslationBackend,
)
from nemo_curator.stages.text.translation.backends.nmt import NMTTranslationBackend


# ---------------------------------------------------------------------------
# TranslationBackend ABC tests
# ---------------------------------------------------------------------------


class TestTranslationBackendABC:
    """Tests for the abstract base class."""

    def test_backend_abc_cannot_instantiate(self) -> None:
        """TranslationBackend is abstract and cannot be directly instantiated."""
        with pytest.raises(TypeError):
            TranslationBackend()


# ---------------------------------------------------------------------------
# get_backend() factory tests
# ---------------------------------------------------------------------------


class TestGetBackendFactory:
    """Tests for the get_backend() factory function."""

    @patch(
        "nemo_curator.stages.text.translation.backends.google.GoogleTranslationBackend.__init__",
        return_value=None,
    )
    def test_get_backend_google(self, mock_init: MagicMock) -> None:
        """Factory returns GoogleTranslationBackend for 'google'."""
        backend = get_backend("google", {})
        assert isinstance(backend, GoogleTranslationBackend)

    @patch(
        "nemo_curator.stages.text.translation.backends.aws.AWSTranslationBackend.__init__",
        return_value=None,
    )
    def test_get_backend_aws(self, mock_init: MagicMock) -> None:
        """Factory returns AWSTranslationBackend for 'aws'."""
        backend = get_backend("aws", {})
        assert isinstance(backend, AWSTranslationBackend)

    def test_get_backend_nmt(self) -> None:
        """Factory returns NMTTranslationBackend for 'nmt'."""
        backend = get_backend("nmt", {"server_url": "http://localhost:8000"})
        assert isinstance(backend, NMTTranslationBackend)

    def test_get_backend_unknown_raises(self) -> None:
        """Unknown backend type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            get_backend("unknown_backend", {})


# ---------------------------------------------------------------------------
# GoogleTranslationBackend tests
# ---------------------------------------------------------------------------


class TestGoogleTranslationBackend:
    """Tests for GoogleTranslationBackend."""

    def test_google_translate_batch_mock(self) -> None:
        """Mock google-cloud-translate client and verify translation."""
        backend = GoogleTranslationBackend(api_version="v2")

        # Create a mock google translate v2 client.
        mock_client = MagicMock()
        mock_client.translate.return_value = {"translatedText": "Hola mundo"}

        # Inject the mock client directly.
        backend._client = mock_client
        backend._api_version = "v2"
        backend._semaphore = asyncio.Semaphore(32)

        result = backend.translate_batch(["Hello world"], "en", "es")

        assert result == ["Hola mundo"]
        mock_client.translate.assert_called_once_with(
            "Hello world",
            source_language="en",
            target_language="es",
            format_="text",
        )

    def test_google_translate_batch_inside_running_loop(self) -> None:
        """The sync wrapper should still work when an event loop is already running."""
        backend = GoogleTranslationBackend(api_version="v2")
        backend.translate_batch_async = AsyncMock(return_value=["Hola mundo"])  # type: ignore[method-assign]

        async def _call() -> list[str]:
            return backend.translate_batch(["Hello world"], "en", "es")

        result = asyncio.run(_call())

        assert result == ["Hola mundo"]
        backend.translate_batch_async.assert_awaited_once_with(["Hello world"], "en", "es")


# ---------------------------------------------------------------------------
# AWSTranslationBackend tests
# ---------------------------------------------------------------------------


class TestAWSTranslationBackend:
    """Tests for AWSTranslationBackend."""

    def test_aws_translate_batch_mock(self) -> None:
        """Mock boto3 client and verify translation."""
        backend = AWSTranslationBackend(region="us-east-1")

        mock_client = MagicMock()
        mock_client.translate_text.return_value = {
            "TranslatedText": "Hola mundo"
        }

        backend._client = mock_client
        backend._semaphore = asyncio.Semaphore(32)

        result = backend.translate_batch(["Hello world"], "en", "es")

        assert result == ["Hola mundo"]
        mock_client.translate_text.assert_called_once_with(
            Text="Hello world",
            SourceLanguageCode="en",
            TargetLanguageCode="es",
        )

    def test_aws_size_limit(self) -> None:
        """Text exceeding 10KB raises ValueError."""
        backend = AWSTranslationBackend(region="us-east-1")

        mock_client = MagicMock()
        backend._client = mock_client
        backend._semaphore = asyncio.Semaphore(32)

        # Create a text that exceeds 10,000 bytes in UTF-8.
        oversized_text = "a" * (AWS_MAX_BYTES_PER_REQUEST + 1)

        with pytest.raises(ValueError, match="AWS TranslateText input too large"):
            backend.translate_batch([oversized_text], "en", "es")

    def test_aws_translate_batch_inside_running_loop(self) -> None:
        """The sync wrapper should still work when an event loop is already running."""
        backend = AWSTranslationBackend(region="us-east-1")
        backend.translate_batch_async = AsyncMock(return_value=["Hola mundo"])  # type: ignore[method-assign]

        async def _call() -> list[str]:
            return backend.translate_batch(["Hello world"], "en", "es")

        result = asyncio.run(_call())

        assert result == ["Hola mundo"]
        backend.translate_batch_async.assert_awaited_once_with(["Hello world"], "en", "es")


# ---------------------------------------------------------------------------
# NMTTranslationBackend tests
# ---------------------------------------------------------------------------


class TestNMTTranslationBackend:
    """Tests for NMTTranslationBackend."""

    def test_nmt_translate_batch_mock(self) -> None:
        """Mock aiohttp POST and verify batched translation."""
        backend = NMTTranslationBackend(
            server_url="http://localhost:8000",
            batch_size=32,
        )
        backend._semaphore = asyncio.Semaphore(32)

        # Create a mock aiohttp response.
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"translations": ["Hola mundo", "Adios"]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        # Create a mock session.
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        backend._session = mock_session

        result = backend.translate_batch(
            ["Hello world", "Goodbye"], "en", "es"
        )

        assert result == ["Hola mundo", "Adios"]
        mock_session.post.assert_called_once_with(
            "http://localhost:8000/translate",
            json={
                "texts": ["Hello world", "Goodbye"],
                "src_lang": "en",
                "tgt_lang": "es",
            },
        )

    def test_nmt_batch_splitting(self) -> None:
        """Large input split into sub-batches of batch_size."""
        backend = NMTTranslationBackend(
            server_url="http://localhost:8000",
            batch_size=2,
        )
        backend._semaphore = asyncio.Semaphore(32)

        # 5 texts with batch_size=2 should produce 3 sub-batches (2, 2, 1).
        texts = ["text1", "text2", "text3", "text4", "text5"]

        call_payloads: list[dict] = []

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        # Return matching-length translations for each sub-batch.
        async def mock_json():
            # Find the texts from the last call.
            last_call = call_payloads[-1]
            return {
                "translations": [f"translated_{t}" for t in last_call["texts"]]
            }

        mock_response.json = mock_json
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.closed = False

        def capture_post(url, json=None):
            call_payloads.append(json)
            return mock_response

        mock_session.post = capture_post

        backend._session = mock_session

        result = backend.translate_batch(texts, "en", "es")

        assert len(result) == 5
        assert len(call_payloads) == 3
        # First sub-batch has 2 texts.
        assert call_payloads[0]["texts"] == ["text1", "text2"]
        # Second sub-batch has 2 texts.
        assert call_payloads[1]["texts"] == ["text3", "text4"]
        # Third sub-batch has 1 text.
        assert call_payloads[2]["texts"] == ["text5"]
        # Verify the result order is preserved.
        assert result == [
            "translated_text1",
            "translated_text2",
            "translated_text3",
            "translated_text4",
            "translated_text5",
        ]

    def test_nmt_translate_batch_inside_running_loop(self) -> None:
        """The sync wrapper should still work when an event loop is already running."""
        backend = NMTTranslationBackend(server_url="http://localhost:8000")
        backend.translate_batch_async = AsyncMock(return_value=["Hola mundo"])  # type: ignore[method-assign]

        async def _call() -> list[str]:
            return backend.translate_batch(["Hello world"], "en", "es")

        result = asyncio.run(_call())

        assert result == ["Hola mundo"]
        backend.translate_batch_async.assert_awaited_once_with(["Hello world"], "en", "es")


# ---------------------------------------------------------------------------
# Retry / exponential backoff tests
# ---------------------------------------------------------------------------


class TestRetryOnTransientError:
    """Tests for retry behavior with exponential backoff."""

    def test_retry_on_transient_error(self) -> None:
        """Verify exponential backoff retry -- mock one failure then success."""
        backend = GoogleTranslationBackend(api_version="v2")
        backend._semaphore = asyncio.Semaphore(32)

        mock_client = MagicMock()
        # First call raises a transient error, second call succeeds.
        mock_client.translate.side_effect = [
            RuntimeError("Transient API error"),
            {"translatedText": "Hola mundo"},
        ]
        backend._client = mock_client

        # Patch asyncio.sleep to avoid actual delays in tests.
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = backend.translate_batch(["Hello world"], "en", "es")

        assert result == ["Hola mundo"]
        assert mock_client.translate.call_count == 2
        mock_sleep.assert_called_once()
        wait_time = mock_sleep.call_args.args[0]
        assert 0.0 <= wait_time <= 1.0
