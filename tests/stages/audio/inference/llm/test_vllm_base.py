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

"""Unit tests for VLLMInference helper class.

VLLMInference imports ``vllm`` and ``transformers`` at module level, so tests
that don't have those packages installed need the ``_mock_vllm_deps`` fixture
**before** importing the class.  Because pytest fixtures cannot run before
top-level imports, we use ``importlib.import_module`` inside the fixture to
defer the import until after the patches are active.

If ``vllm`` *is* installed in your environment, you can import at the top level
instead and skip the fixture — but for CI without GPUs/vllm, this pattern keeps
everything working.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

MODULE = "nemo_curator.stages.audio.inference.llm.vllm_base"


def _get_vllm_inference_cls() -> type:
    """Import and return VLLMInference (must be called inside a mock context)."""
    mod = importlib.import_module(MODULE)
    return mod.VLLMInference


@pytest.fixture
def _mock_vllm_deps() -> dict:
    """Patch vllm and transformers so VLLMInference can be instantiated without GPU."""
    mock_llm_cls = MagicMock()
    mock_sampling_cls = MagicMock()
    mock_tokenizer_cls = MagicMock()

    with (
        patch(f"{MODULE}.LLM", mock_llm_cls),
        patch(f"{MODULE}.SamplingParams", mock_sampling_cls),
        patch(f"{MODULE}.AutoTokenizer", mock_tokenizer_cls),
    ):
        yield {
            "LLM": mock_llm_cls,
            "SamplingParams": mock_sampling_cls,
            "AutoTokenizer": mock_tokenizer_cls,
        }


@pytest.fixture
def vllm_inference_cls(_mock_vllm_deps: dict) -> type:
    """Return the VLLMInference class with mocked heavy dependencies."""
    return _get_vllm_inference_cls()


class TestVLLMInferenceInit:
    def test_requires_at_least_one_prompt(self, vllm_inference_cls: type) -> None:
        with pytest.raises(ValueError, match="One of"):
            vllm_inference_cls(model={"model": "m"})

    def test_rejects_multiple_prompts(self, vllm_inference_cls: type) -> None:
        with pytest.raises(ValueError, match="more than one"):
            vllm_inference_cls(
                prompt={"user": "{text}"},
                prompt_field="my_prompt",
                model={"model": "m"},
            )

    def test_init_stores_config(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(
            prompt={"user": "Do: {text}"},
            generation_field="gen",
            model={"model": "test-model"},
            inference={"temperature": 0.5},
            apply_chat_template={"tokenize": False},
        )
        assert v.prompt == {"user": "Do: {text}"}
        assert v.generation_field == "gen"
        assert v.model_params["model"] == "test-model"
        assert v.inference_params == {"temperature": 0.5}
        assert v.llm is None
        assert v.tokenizer is None

    def test_prompt_file(self, vllm_inference_cls: type, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.yaml"
        prompt_file.write_text("user: 'Punctuate: {text}'\n")

        v = vllm_inference_cls(
            prompt_file=str(prompt_file),
            model={"model": "m"},
        )
        assert v.prompt == {"user": "Punctuate: {text}"}


# ---------------------------------------------------------------------------
# get_entry_prompt
# ---------------------------------------------------------------------------


class TestVLLMInferenceGetEntryPrompt:
    def test_formats_static_prompt(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(prompt={"user": "Fix: {text}"}, model={"model": "m"})
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        v.tokenizer = mock_tokenizer

        result = v.get_entry_prompt({"text": "hello"})
        assert result == "formatted"
        mock_tokenizer.apply_chat_template.assert_called_once_with([{"role": "user", "content": "Fix: hello"}])

    def test_uses_prompt_field(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(prompt_field="my_prompt", model={"model": "m"})
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "ok"
        v.tokenizer = mock_tokenizer

        entry = {
            "text": "hello",
            "my_prompt": {"system": "You are helpful.", "user": "Fix: {text}"},
        }
        result = v.get_entry_prompt(entry)
        assert result == "ok"

    def test_falls_back_on_template_error(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(prompt={"user": "{text}"}, model={"model": "m"})
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = RuntimeError("bad template")
        v.tokenizer = mock_tokenizer

        result = v.get_entry_prompt({"text": "hi"})
        assert isinstance(result, list)
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------


class TestVLLMInferenceProcessBatch:
    def test_generate_mode(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(prompt={"user": "{text}"}, model={"model": "m"})
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="out")])]
        v.llm = mock_llm
        v.sampling_params = MagicMock()

        results = v.process_batch(["prompt1"])
        mock_llm.generate.assert_called_once()
        assert results[0].outputs[0].text == "out"

    def test_chat_mode_when_use_chat_api_set(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(
            prompt={"user": "{text}"},
            model={"model": "m"},
            use_chat_api=True,
        )
        mock_llm = MagicMock()
        mock_llm.chat.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="chat_out")])]
        v.llm = mock_llm
        v.sampling_params = MagicMock()

        results = v.process_batch(["prompt1"])
        mock_llm.chat.assert_called_once()
        assert results[0].outputs[0].text == "chat_out"


# ---------------------------------------------------------------------------
# clean_up
# ---------------------------------------------------------------------------


class TestVLLMInferenceCleanUp:
    def test_clean_up_releases_llm(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(prompt={"user": "{text}"}, model={"model": "m"})
        v.llm = MagicMock()
        v.device = "cpu"

        with patch(f"{MODULE}.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            v.clean_up()

        assert v.llm is None

    def test_clean_up_handles_none(self, vllm_inference_cls: type) -> None:
        v = vllm_inference_cls(prompt={"user": "{text}"}, model={"model": "m"})
        with patch(f"{MODULE}.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            v.clean_up()
