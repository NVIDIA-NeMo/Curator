# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured requests preserve the shared model's sampling configuration."""

from types import ModuleType, SimpleNamespace

import pytest

from nemo_curator.models.vllm_model import VLLMModel


def test_structured_request_clones_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("vllm.sampling_params")
    module.StructuredOutputsParams = SimpleNamespace
    monkeypatch.setitem(__import__("sys").modules, "vllm.sampling_params", module)
    model = VLLMModel("test")
    original = SimpleNamespace(temperature=0.2, max_tokens=10)
    model._sampling_params = original
    captured = []

    def generate(prompts: list[str], *, sampling_params: object, use_tqdm: bool) -> list:
        captured.append(sampling_params)
        assert not use_tqdm
        return [SimpleNamespace(outputs=[SimpleNamespace(text="{}")]) for _ in prompts]

    model._llm = SimpleNamespace(generate=generate)
    schema = {"type": "object", "additionalProperties": False}
    assert model.generate(["prompt"], json_schema=schema) == ["{}"]
    assert captured[0] is not original
    assert captured[0].structured_outputs.json == schema
    assert captured[0].temperature == original.temperature
    assert not hasattr(original, "structured_outputs")
    assert model.generate(["prompt"]) == ["{}"]
    assert captured[1] is original


def test_generation_requires_initialized_model() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        VLLMModel("test").generate(["prompt"], json_schema={"type": "object"})
