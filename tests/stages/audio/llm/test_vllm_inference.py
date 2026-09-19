# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the dependency-light vLLM conversation stage contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.models.vllm_model import VLLMModel
from nemo_curator.stages.audio.llm.vllm_inference import vLLMInference
from nemo_curator.tasks import AudioTask


class _Tokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        return " | ".join(message["content"] for message in messages)


def _model(**overrides: object) -> VLLMModel:
    params: dict[str, object] = {"model": "test-model", "tensor_parallel_size": 1}
    params.update(overrides)
    return VLLMModel(**params)


def _task(topic: str = "weather") -> AudioTask:
    task = AudioTask(dataset_name="topics", data={"topic": topic}, task_id="source-task")
    task._metadata = {"source_files": ["topics.jsonl"]}
    task._source_id = "source-id"
    return task


def _stage(model: VLLMModel | dict | None = None, **kwargs: object) -> vLLMInference:
    return vLLMInference(prompt="system:Create a conversation about $topic", model=model or _model(), **kwargs)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_model_len", 4096),
        ("tensor_parallel_size", 1),
        ("max_num_batched_tokens", 2048),
        ("temperature", 0.2),
        ("top_p", 0.9),
        ("top_k", 10),
        ("min_p", 0.1),
        ("max_tokens", 256),
        ("cache_dir", str(Path("test-cache"))),
    ],
)
def test_model_dict_accepts_every_public_vllm_model_parameter(key: str, value: object) -> None:
    stage = _stage({"model": "test-model", "tensor_parallel_size": 1, key: value})
    assert getattr(stage._vllm_model, key) == value


def test_model_dict_rejects_nonexistent_vllm_parameter() -> None:
    with pytest.raises(ValueError, match="Unsupported VLLMModel parameters"):
        _stage({"model": "test-model", "disable_dual_chunk_attention": True})


def test_tensor_parallelism_sets_matching_gpu_reservation() -> None:
    stage = _stage({"model": "test-model", "tensor_parallel_size": 2})
    assert stage.resources.gpus == 2


def test_unspecified_tensor_parallelism_defaults_to_one_reserved_gpu() -> None:
    model = VLLMModel(model="test-model")
    stage = _stage(model)
    assert model.tensor_parallel_size == 1
    assert stage.resources.gpus == 1


def test_preinitialized_model_without_tensor_parallelism_is_rejected() -> None:
    model = VLLMModel(model="test-model")
    model._llm = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must set tensor_parallel_size"):
        _stage(model)


def test_retry_keeps_prompts_when_vllm_returns_too_few_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient short response must not silently drop later prompts."""
    model = _model()
    stage = _stage(model, max_retry_rounds=2)
    valid = '{"turns": [{"speaker": "Alice", "utterance": "Hello"}, {"speaker": "Bob", "utterance": "Hi"}]}'
    calls = 0

    def generate(_prompts: list[str]) -> list[str]:
        nonlocal calls
        calls += 1
        return [valid]

    monkeypatch.setattr(model, "generate", generate)

    results = stage.generate_batch_with_retry(["first", "second"], max_retry_rounds=2)

    assert results == [stage.validate_json_output(valid), stage.validate_json_output(valid)]
    assert calls == 2


def test_process_batch_accepts_multi_row_numpy_batch_and_preserves_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    stage = _stage(model)
    tokenizer = _Tokenizer()
    monkeypatch.setattr(model, "setup", lambda: None)
    monkeypatch.setattr(model, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        model,
        "generate",
        lambda _prompts: [
            '{"turns": [{"speaker": "Alice", "utterance": "Hello there"}, '
            '{"speaker": "Bob", "utterance": "Hello Alice"}]}'
        ]
        * 2,
    )

    results = stage.process_batch(np.array([_task("one"), _task("two")], dtype=object))

    assert len(results) == 4
    assert all(result._metadata == {"source_files": ["topics.jsonl"]} for result in results)
    assert all(result._source_id == "source-id" for result in results)
    assert {result.data["topic"] for result in results} == {"one", "two"}


@pytest.mark.parametrize(
    "unsafe_label",
    [str(Path("/") / "tmp" / "escape"), "../escape", "a/b", r"a\b", ".", ".."],
)
def test_generated_turns_reject_path_like_speaker_labels(unsafe_label: str) -> None:
    stage = _stage()
    output = json.dumps(
        {
            "turns": [
                {"speaker": unsafe_label, "utterance": "Hello"},
                {"speaker": "Bob", "utterance": "Hi"},
            ]
        }
    )
    assert stage.validate_json_output(output) is None


def test_preinitialized_model_still_initializes_the_stage_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    model._llm = object()  # type: ignore[assignment]
    stage = _stage(model)
    tokenizer = _Tokenizer()
    monkeypatch.setattr(model, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(model, "generate", lambda _prompts: [])

    assert stage.process_batch([_task()]) == []
    assert stage.tokenizer is tokenizer
