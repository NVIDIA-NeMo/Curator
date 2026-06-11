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

"""Unit tests for nemo_curator.models.omni.base.NVInferenceClient."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.omni.base import NVInferenceClient


def _content_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


def _empty_choices_chunk() -> MagicMock:
    chunk = MagicMock()
    chunk.choices = []
    return chunk


class _AsyncStreamIter:
    """Minimal async iterator over a fixed list of chunks (mimics openai.AsyncStream)."""

    def __init__(self, chunks: list[MagicMock]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> "_AsyncStreamIter":
        return self

    async def __anext__(self) -> MagicMock:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _client_with_stream(chunks: list[MagicMock], **kwargs: object) -> NVInferenceClient:
    """An NVInferenceClient whose underlying async create() yields ``chunks``."""
    client = NVInferenceClient(**kwargs)
    inner = MagicMock()
    inner.chat.completions.create = AsyncMock(return_value=_AsyncStreamIter(chunks))
    client.client = inner  # bypass setup()
    return client


def _run_query(client: NVInferenceClient, **kwargs: object) -> list[str]:
    return asyncio.run(client._query_model_impl(model="test/model", **kwargs))


class TestNVInferenceClientSetup:
    def test_setup_raises_when_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVINFERENCE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            NVInferenceClient().setup()

    def test_setup_whitespace_only_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVINFERENCE_API_KEY", "   ")
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            NVInferenceClient().setup()

    def test_setup_constructs_asyncopenai_with_resolved_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVINFERENCE_API_KEY", "key-xyz")  # pragma: allowlist secret
        with patch("nemo_curator.models.client.openai_client.AsyncOpenAI") as AsyncOpenAI:  # noqa: N806
            NVInferenceClient(base_url="https://example.test").setup()
            kwargs = AsyncOpenAI.call_args.kwargs
            assert kwargs["base_url"] == "https://example.test"
            assert kwargs["api_key"] == "key-xyz"  # pragma: allowlist secret


class TestNVInferenceClientStreaming:
    def test_reassembles_stream_skipping_none_and_empty_chunks(self) -> None:
        client = _client_with_stream(
            [
                _content_chunk("hello"),
                _content_chunk(None),  # reasoning models emit None content deltas
                _empty_choices_chunk(),  # final usage chunk has no choices
                _content_chunk(" "),
                _content_chunk("world"),
            ]
        )
        out = _run_query(client, messages=[{"role": "user", "content": "p"}])
        assert out == ["hello world"]
        call = client.client.chat.completions.create.call_args.kwargs
        assert call["model"] == "test/model"
        assert call["stream"] is True
        assert call["messages"] == [{"role": "user", "content": "p"}]

    def test_generation_config_threads_through(self) -> None:
        client = _client_with_stream([_content_chunk("ok")])
        _run_query(
            client,
            messages=[{"role": "user", "content": "p"}],
            generation_config=GenerationConfig(max_tokens=123, temperature=0.5, top_p=0.9),
        )
        call = client.client.chat.completions.create.call_args.kwargs
        assert call["max_tokens"] == 123
        assert call["temperature"] == 0.5
        assert call["top_p"] == 0.9

    @pytest.mark.parametrize(
        ("flag", "expected_header"),
        [
            (True, {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}),
            (False, None),
        ],
    )
    def test_priority_mode_threads_to_extra_headers(self, flag: bool, expected_header: dict | None) -> None:
        client = _client_with_stream([_content_chunk("ok")], priority_mode=flag)
        _run_query(client, messages=[{"role": "user", "content": "p"}])
        assert client.client.chat.completions.create.call_args.kwargs.get("extra_headers") == expected_header
