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

"""Unit tests for the thin NVIDIA Inference HTTP client helpers.

Covers the non-obvious wrapper behavior: whitespace handling on the env-var
read, and stream-chunk reassembly (None deltas, empty-choices chunks).
"""

from unittest.mock import MagicMock

import pytest

from nemo_curator.models.client.nvinference_client import (
    get_nvinference_api_key,
    stream_chat_completion_text,
)


def _content_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


def _empty_choices_chunk() -> MagicMock:
    chunk = MagicMock()
    chunk.choices = []
    return chunk


class TestApiKey:
    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVINFERENCE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            get_nvinference_api_key()

    def test_whitespace_only_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The wrapper strips before checking emptiness — easy to regress.
        monkeypatch.setenv("NVINFERENCE_API_KEY", "   ")
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            get_nvinference_api_key()

    def test_custom_env_var_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MY_CUSTOM_KEY is not set"):
            get_nvinference_api_key("MY_CUSTOM_KEY")


class TestStreamChatCompletionText:
    def test_concatenates_deltas_skipping_none_and_empty_chunks(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = [
            _content_chunk("hello"),
            _content_chunk(None),  # delta with no content — must be skipped
            _empty_choices_chunk(),  # usage chunk: choices=[] — must be skipped
            _content_chunk(" "),
            _content_chunk("world"),
        ]
        result = stream_chat_completion_text(
            client,
            model="my-model",
            messages=[{"role": "user"}],
            temperature=0.5,
            top_p=0.9,
            max_tokens=512,
            extra_headers={"X-Custom": "value"},
        )
        assert result == "hello world"
        # All call-time kwargs reach the upstream client unchanged, plus stream=True.
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["extra_headers"] == {"X-Custom": "value"}
        assert kwargs["max_tokens"] == 512
