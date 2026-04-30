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

"""Unit tests for nemo_curator.models.omni.base."""

import base64
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from nemo_curator.models.omni.base import (
    InferenceConfig,
    ModelConfig,
    NVInferenceModel,
    NVInferenceModelConfig,
    VLMModel,
)

# ---------------------------------------------------------------------------
# Concrete VLMModel subclass for testing the abstract base
# ---------------------------------------------------------------------------


class _ConcreteVLMModel(VLMModel):
    def load(self) -> None:
        self._model = "loaded"

    def preload(self) -> None:
        pass

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image] | None,
        inference_config: InferenceConfig,
    ) -> list[str]:
        return [""] * len(prompts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nv_model(model_id: str = "test/model") -> NVInferenceModel:
    return NVInferenceModel(model_id, NVInferenceModelConfig())


# ---------------------------------------------------------------------------
# InferenceConfig defaults
# ---------------------------------------------------------------------------


class TestInferenceConfig:
    def test_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.temperature == 0.0
        assert cfg.top_p == 1.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.do_sample is False
        assert cfg.priority_mode is False


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_valid_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.tensor_parallel_size == 1
        assert cfg.gpu_memory_utilization == pytest.approx(0.9)

    def test_gpu_memory_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=r"0\.0 and 1\.0"):
            ModelConfig(gpu_memory_utilization=-0.1)

    def test_gpu_memory_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match=r"0\.0 and 1\.0"):
            ModelConfig(gpu_memory_utilization=1.1)

    def test_gpu_memory_boundary_values_valid(self) -> None:
        ModelConfig(gpu_memory_utilization=0.0)
        ModelConfig(gpu_memory_utilization=1.0)

    def test_tensor_parallel_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            ModelConfig(tensor_parallel_size=0)

    def test_tensor_parallel_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            ModelConfig(tensor_parallel_size=-1)


# ---------------------------------------------------------------------------
# VLMModel is_loaded / unload
# ---------------------------------------------------------------------------


class TestVLMModelIsLoaded:
    def test_not_loaded_initially(self) -> None:
        model = _ConcreteVLMModel("test-model", ModelConfig())
        assert model.is_loaded is False

    def test_loaded_after_load_call(self) -> None:
        model = _ConcreteVLMModel("test-model", ModelConfig())
        model.load()
        assert model.is_loaded is True

    def test_unload_clears_model(self) -> None:
        model = _ConcreteVLMModel("test-model", ModelConfig())
        model.load()
        model.unload()
        assert model.is_loaded is False


# ---------------------------------------------------------------------------
# NVInferenceModel.is_loaded
# ---------------------------------------------------------------------------


class TestNVInferenceModelIsLoaded:
    def test_not_loaded_initially(self) -> None:
        assert _make_nv_model().is_loaded is False


# ---------------------------------------------------------------------------
# NVInferenceModel.load / unload
# ---------------------------------------------------------------------------


class TestNVInferenceModelLoad:
    @patch("nemo_curator.models.client.nvinference_client.create_openai_client")
    @patch("nemo_curator.models.client.nvinference_client.get_nvinference_api_key")
    def test_load_initializes_client(
        self, mock_get_key: MagicMock, mock_create: MagicMock
    ) -> None:
        mock_get_key.return_value = "key"
        mock_create.return_value = MagicMock()
        model = _make_nv_model()
        model.load()
        assert model.is_loaded is True
        mock_get_key.assert_called_once()
        mock_create.assert_called_once_with(api_key="key", base_url="https://inference-api.nvidia.com")  # pragma: allowlist secret

    @patch("nemo_curator.models.client.nvinference_client.create_openai_client")
    @patch("nemo_curator.models.client.nvinference_client.get_nvinference_api_key")
    def test_load_is_idempotent(
        self, mock_get_key: MagicMock, mock_create: MagicMock
    ) -> None:
        mock_get_key.return_value = "key"
        mock_create.return_value = MagicMock()
        model = _make_nv_model()
        model.load()
        model.load()
        mock_create.assert_called_once()


class TestNVInferenceModelUnload:
    def test_unload_clears_client(self) -> None:
        model = _make_nv_model()
        model._client = MagicMock()
        model.unload()
        assert model.is_loaded is False


# ---------------------------------------------------------------------------
# NVInferenceModel._encode_image_to_base64
# ---------------------------------------------------------------------------


class TestNVInferenceModelEncodeImage:
    def test_returns_valid_base64_png(self) -> None:
        model = _make_nv_model()
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        result = model._encode_image_to_base64(img)
        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# NVInferenceModel._build_message_content
# ---------------------------------------------------------------------------


class TestNVInferenceModelBuildMessageContent:
    def test_text_only(self) -> None:
        model = _make_nv_model()
        content = model._build_message_content("hello", None)
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "hello"}

    def test_with_pil_image_produces_data_url(self) -> None:
        model = _make_nv_model()
        img = Image.new("RGB", (4, 4))
        content = model._build_message_content("describe", img)
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"
        url = content[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_with_uri_string_passed_as_is(self) -> None:
        model = _make_nv_model()
        uri = "https://example.com/img.jpg"
        content = model._build_message_content("describe", uri)
        assert content[0]["image_url"]["url"] == uri


# ---------------------------------------------------------------------------
# NVInferenceModel.generate
# ---------------------------------------------------------------------------


class TestNVInferenceModelGenerate:
    def test_raises_when_not_loaded(self) -> None:
        model = _make_nv_model()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.generate(["prompt"], None, InferenceConfig())

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_calls_stream_per_prompt(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = "response"
        model = _make_nv_model()
        model._client = MagicMock()
        results = model.generate(["p1", "p2"], None, InferenceConfig())
        assert mock_stream.call_count == 2
        assert results == ["response", "response"]

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_error_returns_empty_string(self, mock_stream: MagicMock) -> None:
        mock_stream.side_effect = RuntimeError("API error")
        model = _make_nv_model()
        model._client = MagicMock()
        results = model.generate(["p"], None, InferenceConfig())
        assert results == [""]

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_priority_mode_adds_header(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = "ok"
        model = _make_nv_model()
        model._client = MagicMock()
        model.generate(["p"], None, InferenceConfig(priority_mode=True))
        kwargs = mock_stream.call_args[1]
        assert kwargs["extra_headers"] == {"X-Vertex-AI-LLM-Shared-Request-Type": "priority"}

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_no_priority_mode_no_header(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = "ok"
        model = _make_nv_model()
        model._client = MagicMock()
        model.generate(["p"], None, InferenceConfig(priority_mode=False))
        kwargs = mock_stream.call_args[1]
        assert kwargs["extra_headers"] is None

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_image_included_in_message_content(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = "ok"
        model = _make_nv_model()
        model._client = MagicMock()
        imgs = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
        model.generate(["p1", "p2"], imgs, InferenceConfig())
        first_call_kwargs = mock_stream.call_args_list[0][1]
        content = first_call_kwargs["messages"][0]["content"]
        assert len(content) == 2  # image + text
        assert content[0]["type"] == "image_url"

    @patch("nemo_curator.models.client.nvinference_client.stream_chat_completion_text")
    def test_empty_response_becomes_empty_string(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = ""
        model = _make_nv_model()
        model._client = MagicMock()
        results = model.generate(["p"], None, InferenceConfig())
        assert results == [""]
