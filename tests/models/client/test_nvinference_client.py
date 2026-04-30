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

"""Unit tests for nemo_curator.models.client.nvinference_client."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.models.client.nvinference_client import (
    create_openai_client,
    get_nvinference_api_key,
    stream_chat_completion_text,
)

# ---------------------------------------------------------------------------
# get_nvinference_api_key
# ---------------------------------------------------------------------------


class TestGetNvinferenceApiKey:
    def test_raises_when_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVINFERENCE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            get_nvinference_api_key()

    def test_raises_when_env_var_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVINFERENCE_API_KEY", "   ")
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            get_nvinference_api_key()

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVINFERENCE_API_KEY", "test-key-123")
        assert get_nvinference_api_key() == "test-key-123"

    def test_custom_env_var_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-value")
        assert get_nvinference_api_key("MY_CUSTOM_KEY") == "custom-value"

    def test_custom_env_var_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MY_CUSTOM_KEY is not set"):
            get_nvinference_api_key("MY_CUSTOM_KEY")


# ---------------------------------------------------------------------------
# create_openai_client
# ---------------------------------------------------------------------------


class TestCreateOpenaiClient:
    @patch("openai.OpenAI")
    def test_creates_client_with_correct_args(self, mock_openai_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        client = create_openai_client(api_key="key123", base_url="https://example.com")
        mock_openai_cls.assert_called_once_with(base_url="https://example.com", api_key="key123")
        assert client is mock_instance

    @patch("openai.OpenAI")
    def test_default_base_url(self, mock_openai_cls: MagicMock) -> None:
        create_openai_client(api_key="key")
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == "https://inference-api.nvidia.com"


# ---------------------------------------------------------------------------
# stream_chat_completion_text
# ---------------------------------------------------------------------------


def _make_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


class TestStreamChatCompletionText:
    def test_collects_chunks_into_string(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = [
            _make_chunk("hello"),
            _make_chunk(" world"),
        ]
        result = stream_chat_completion_text(client, model="m", messages=[])
        assert result == "hello world"

    def test_none_delta_skipped(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = [
            _make_chunk("hello"),
            _make_chunk(None),
            _make_chunk("!"),
        ]
        result = stream_chat_completion_text(client, model="m", messages=[])
        assert result == "hello!"

    def test_empty_stream_returns_empty_string(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = []
        result = stream_chat_completion_text(client, model="m", messages=[])
        assert result == ""

    def test_passes_kwargs_to_create(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = []
        stream_chat_completion_text(
            client,
            model="my-model",
            messages=[{"role": "user"}],
            temperature=0.5,
            top_p=0.9,
            max_tokens=512,
        )
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "my-model"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 512
        assert kwargs["stream"] is True

    def test_extra_headers_passed_through(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = []
        headers = {"X-Custom": "value"}
        stream_chat_completion_text(client, model="m", messages=[], extra_headers=headers)
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["extra_headers"] == headers
