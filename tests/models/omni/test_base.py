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

"""Unit tests for nemo_curator.models.omni.base.

The underlying ``openai.OpenAI`` client (held by ``OpenAIClient``) is mocked;
what these tests verify is that the wrapper:
  - reads + validates the env-var API key on setup,
  - builds correct OpenAI-compatible message contents (text-only, embedded
    PIL image, URL string passthrough),
  - reassembles streamed deltas while skipping ``None`` content and
    empty-choices chunks,
  - threads priority_mode through to the extra-headers parameter,
  - swallows per-prompt errors and preserves batch length.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from nemo_curator.models.omni.base import NVInferenceModel


def _content_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


def _empty_choices_chunk() -> MagicMock:
    chunk = MagicMock()
    chunk.choices = []
    return chunk


def _stream(chunks: list[MagicMock]) -> MagicMock:
    """Build a MagicMock OpenAIClient whose chat.completions.create returns ``chunks``."""
    inner_openai = MagicMock()
    inner_openai.chat.completions.create.return_value = chunks
    wrapper = MagicMock()
    wrapper.client = inner_openai
    return wrapper


class TestNVInferenceModel:
    def _model(self, **kwargs: object) -> NVInferenceModel:
        return NVInferenceModel("test/model", **kwargs)

    def test_setup_raises_when_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVINFERENCE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            self._model().setup()

    def test_setup_whitespace_only_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVINFERENCE_API_KEY", "   ")
        with pytest.raises(RuntimeError, match="NVINFERENCE_API_KEY is not set"):
            self._model().setup()

    def test_setup_constructs_openai_client_with_resolved_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVINFERENCE_API_KEY", "key-xyz")  # pragma: allowlist secret
        with patch("nemo_curator.models.omni.base.OpenAIClient") as Client:  # noqa: N806
            m = self._model(base_url="https://example.test")
            m.setup()
            m.setup()  # idempotent — second call should not reconstruct.
            Client.assert_called_once_with(  # pragma: allowlist secret
                base_url="https://example.test", api_key="key-xyz"
            )
            Client.return_value.setup.assert_called_once()
            assert m.is_loaded

    def test_encode_image_to_base64_produces_real_png(self) -> None:
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        decoded = base64.b64decode(self._model()._encode_image_to_base64(img))
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_build_message_content_handles_pil_image_and_url_string(self) -> None:
        # PIL image → real PNG → real base64 → data URL prefix
        pil_content = self._model()._build_message_content("describe", Image.new("RGB", (4, 4)))
        assert pil_content[0]["type"] == "image_url"
        assert pil_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert pil_content[1] == {"type": "text", "text": "describe"}

        # Pre-built URL string passes through untouched (no re-encoding)
        url = "https://example.com/img.jpg"
        url_content = self._model()._build_message_content("describe", url)
        assert url_content[0]["image_url"]["url"] == url

    def test_generate_requires_setup(self) -> None:
        with pytest.raises(RuntimeError, match="not loaded"):
            self._model().generate(["prompt"])

    def test_generate_reassembles_stream_skipping_none_and_empty_chunks(self) -> None:
        m = self._model()
        m._client = _stream(
            [
                _content_chunk("hello"),
                _content_chunk(None),  # skipped — reasoning models emit None deltas
                _empty_choices_chunk(),  # skipped — final usage chunk
                _content_chunk(" "),
                _content_chunk("world"),
            ]
        )
        assert m.generate(["p"], [Image.new("RGB", (4, 4))]) == ["hello world"]
        # The underlying OpenAI client was called with the right model + stream=True.
        call = m._client.client.chat.completions.create.call_args.kwargs
        assert call["model"] == "test/model"
        assert call["stream"] is True
        # First user message has both an image part and a text part.
        first_msg = call["messages"][0]
        assert first_msg["role"] == "user"
        assert any(part["type"] == "image_url" for part in first_msg["content"])

    def test_generate_swallows_per_prompt_errors_as_empty_string(self) -> None:
        m = self._model()
        m._client = _stream([])
        m._client.client.chat.completions.create.side_effect = [
            RuntimeError("transient"),
            iter([_content_chunk("ok")]),
        ]
        assert m.generate(["a", "b"]) == ["", "ok"]

    @pytest.mark.parametrize(
        ("flag", "expected_header"),
        [
            (True, {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}),
            (False, None),
        ],
    )
    def test_priority_mode_threads_to_extra_headers(self, flag: bool, expected_header: dict | None) -> None:
        m = self._model(priority_mode=flag)
        m._client = _stream([_content_chunk("ok")])
        m.generate(["p"])
        assert m._client.client.chat.completions.create.call_args.kwargs["extra_headers"] == expected_header
