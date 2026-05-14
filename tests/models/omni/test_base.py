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

The OpenAI client + ``stream_chat_completion_text`` are mocked because the
class is a thin wrapper over the NVIDIA Inference HTTP API; what these tests
verify is that the wrapper composes the right request and threads kwargs and
errors through correctly.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from nemo_curator.models.omni.base import NVInferenceModel


class TestNVInferenceModel:
    def _model(self, **kwargs: object) -> NVInferenceModel:
        return NVInferenceModel("test/model", **kwargs)

    def test_setup_creates_client_via_factory(self) -> None:
        with (
            patch("nemo_curator.models.client.nvinference_client.create_openai_client") as create,
            patch(
                "nemo_curator.models.client.nvinference_client.get_nvinference_api_key",
                return_value="key-xyz",  # pragma: allowlist secret
            ),
        ):
            m = self._model(base_url="https://example.test")
            m.setup()
            m.setup()  # idempotent — should only build the client once
            create.assert_called_once_with(
                api_key="key-xyz", base_url="https://example.test"
            )  # pragma: allowlist secret
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

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_generate_dispatches_one_call_per_prompt_with_image(self, mock_stream: MagicMock) -> None:
        mock_stream.side_effect = ["r0", "r1"]
        m = self._model()
        m._client = MagicMock()
        results = m.generate(["p0", "p1"], [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))])
        assert results == ["r0", "r1"]
        assert mock_stream.call_count == 2
        # First call's message has the right role and embeds an image part.
        first_msg = mock_stream.call_args_list[0].kwargs["messages"][0]
        assert first_msg["role"] == "user"
        assert any(part["type"] == "image_url" for part in first_msg["content"])

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_generate_swallows_per_prompt_errors_as_empty_string(self, mock_stream: MagicMock) -> None:
        mock_stream.side_effect = [RuntimeError("transient"), "ok"]
        m = self._model()
        m._client = MagicMock()
        assert m.generate(["a", "b"]) == ["", "ok"]

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_priority_mode_adds_vertex_header(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = "ok"
        for flag, expected in [(True, {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}), (False, None)]:
            m = self._model(priority_mode=flag)
            m._client = MagicMock()
            m.generate(["p"])
            assert mock_stream.call_args.kwargs["extra_headers"] == expected
