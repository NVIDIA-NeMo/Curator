# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the dependency-light vLLM conversation stage contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.models.vllm_model import VLLMModel
from nemo_curator.stages.audio.llm.vllm_inference import vLLMInference
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats


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
    return vLLMInference(
        prompt="system:Create a conversation about $topic",
        model=model or _model(),
        speakers=["Alice", "Bob"],
        **kwargs,
    )


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
    valid = '{"turns": [{"speaker": "Alice", "overlap": 0, "utterance": "Hello"}, {"speaker": "Bob", "overlap": 0, "utterance": "Hi"}]}'
    calls = 0

    def generate(_prompts: list[str], **_kwargs: object) -> list[str]:
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
        lambda _prompts, **_kwargs: [
            '{"turns": [{"speaker": "Alice", "overlap": 0, "utterance": "Hello there"}, '
            '{"speaker": "Bob", "overlap": 0, "utterance": "Hello Alice"}]}'
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
                {"speaker": "Bob", "overlap": 0, "utterance": "Hi"},
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
    monkeypatch.setattr(model, "generate", lambda _prompts, **_kwargs: [])

    assert stage.process_batch([_task()]) == []
    assert stage.tokenizer is tokenizer


def _valid_conversation() -> dict:
    return {
        "turns": [
            {"speaker": "Alice", "utterance": "Hello there", "overlap": 0.0},
            {"speaker": "Bob", "utterance": "Good morning", "overlap": -0.5},
        ]
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("speaker", "utterance", "overlap")
        for value in (None, True, 1, [], {})
        if not (field == "overlap" and type(value) is int)
    ],
)
def test_schema_rejects_invalid_turn_types(field: str, value: object) -> None:
    conversation = _valid_conversation()
    conversation["turns"][0][field] = value
    assert _stage().validate_json_output(json.dumps(conversation)) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 999, -999])
def test_schema_rejects_nonfinite_and_out_of_bounds_overlap(value: float) -> None:
    conversation = _valid_conversation()
    conversation["turns"][0]["overlap"] = value
    assert _stage().validate_json_output(json.dumps(conversation)) is None


@pytest.mark.parametrize("mutation", ["top_extra", "turn_extra", "one_speaker", "wrong_speaker", "missing_overlap"])
def test_schema_rejects_wrong_membership_and_extra_fields(mutation: str) -> None:
    conversation = _valid_conversation()
    if mutation == "top_extra":
        conversation["extra"] = "bad"
    elif mutation == "turn_extra":
        conversation["turns"][0]["extra"] = "bad"
    elif mutation == "one_speaker":
        conversation["turns"][1]["speaker"] = "Alice"
    elif mutation == "wrong_speaker":
        conversation["turns"][1]["speaker"] = "Charlie"
    else:
        conversation["turns"][0].pop("overlap")
    assert _stage().validate_json_output(json.dumps(conversation)) is None


@pytest.mark.parametrize(
    ("filename", "count"), [("dialog_prompt.yaml", 2), ("triperson_prompt.yaml", 3), ("fourperson_prompt.yaml", 4)]
)
def test_prompt_variants_share_generation_and_validation_schema(filename: str, count: int) -> None:
    prompt = Path(__file__).parents[4] / "tutorials/audio/data-generation/prompts" / filename
    stage = vLLMInference(prompt_file=str(prompt), model=_model())
    stage.tokenizer = _Tokenizer()
    text = stage.get_entry_prompt({"topic": "weather"})
    assert json.dumps(stage.conversation_schema, ensure_ascii=False) in text
    assert "weather" in text
    assert len(stage.speakers) == count
    conversation = {
        "turns": [{"speaker": speaker, "utterance": "Hello there", "overlap": 0} for speaker in stage.speakers]
    }
    assert stage.validate_json_output(json.dumps(conversation)) == conversation
    conversation["turns"][-1]["speaker"] = stage.speakers[0]
    assert stage.validate_json_output(json.dumps(conversation)) is None


def test_identity_includes_source_timing_and_turn_boundaries() -> None:
    stage = _stage()
    turns = _valid_conversation()["turns"]
    source = _task()
    first = stage.generate_conversation_id(turns, source)
    assert first == stage.generate_conversation_id(deepcopy(turns), source)
    changed = deepcopy(turns)
    changed[1]["overlap"] = 0.7
    assert first != stage.generate_conversation_id(changed, source)
    assert first != stage.generate_conversation_id(turns, _task("another topic"))
    duplicate = _task()
    duplicate._set_task_id("source", 1)
    assert first != stage.generate_conversation_id(turns, duplicate)


def test_anonymous_duplicate_inputs_get_distinct_identities() -> None:
    stage = _stage()
    turns = _valid_conversation()["turns"]
    left, right = AudioTask(data={"topic": "same"}), AudioTask(data={"topic": "same"})
    identity = stage.generate_conversation_id(turns, left)
    assert identity != stage.generate_conversation_id(turns, right)
    assert identity == stage.generate_conversation_id(turns, left)


def test_fanout_uses_schema_and_keeps_independent_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    stage = _stage(model)
    stage.tokenizer = _Tokenizer()
    source = _task()
    source._metadata["nested"] = {"items": [1]}
    source._stage_perf = [StagePerfStats("reader", process_time=0.1)]
    schemas = []

    def generate(prompts: list[str], *, json_schema: dict) -> list[str]:
        schemas.append(json_schema)
        return [json.dumps(_valid_conversation()) for _ in prompts]

    monkeypatch.setattr(model, "generate", generate)
    children = stage.process_batch([source])
    assert schemas == [stage.conversation_schema]
    assert len(children) == 2
    assert all(child.data["conversation_num_turns"] == 2 for child in children)
    assert all(child._source_id == source._source_id for child in children)
    assert all(child._stage_perf == source._stage_perf for child in children)
    assert all(not child.task_id for child in children)
    children[0]._metadata["nested"]["items"].append(2)
    assert source._metadata["nested"]["items"] == [1]
    assert children[1]._metadata["nested"]["items"] == [1]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_retry_count(value: object) -> None:
    with pytest.raises(ValueError, match="max_retry_rounds"):
        _stage(max_retry_rounds=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_tensor_parallel_size(value: object) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        _stage({"model": "test-model", "tensor_parallel_size": value})


@pytest.mark.parametrize("speakers", [[], ["one"], ["one", "one"], [1, "two"], ["../one", "two"], "one,two"])
def test_rejects_invalid_speaker_configuration(speakers: object) -> None:
    with pytest.raises(ValueError, match="speakers"):
        vLLMInference(prompt="user:Hello", model=_model(), speakers=speakers)


@pytest.mark.parametrize(("minimum", "maximum"), [(1, -1), (float("nan"), 1), (-1, float("inf")), (False, 1)])
def test_rejects_invalid_overlap_bounds(minimum: float, maximum: float) -> None:
    with pytest.raises(ValueError, match="overlap bounds"):
        vLLMInference(prompt="user:Hello", model=_model(), min_overlap=minimum, max_overlap=maximum)


def test_prompt_field_renders_topic_and_schema() -> None:
    stage = vLLMInference(prompt_field="instruction", model=_model())
    stage.tokenizer = _Tokenizer()
    prompt = stage.get_entry_prompt({"instruction": "user:Discuss $topic", "topic": "weather"})
    assert "Discuss weather" in prompt
    assert json.dumps(stage.conversation_schema) in prompt
    with pytest.raises(ValueError, match="Prompt field"):
        stage.get_entry_prompt({"topic": "weather"})


def test_fenced_json_is_normalized_and_non_speech_text_is_rejected() -> None:
    stage = _stage()
    valid = _valid_conversation()
    valid["turns"][0]["utterance"] = " Hello\u200b  there\t "
    result = stage.validate_json_output("```json\n" + json.dumps(valid) + "\n```")
    assert result["turns"][0]["utterance"] == "Hello there"
    valid["turns"][0]["utterance"] = "...."
    assert stage.validate_json_output(json.dumps(valid)) is None


def test_setup_reuses_initialized_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    calls = []
    monkeypatch.setattr(model, "setup", lambda: calls.append(True))
    monkeypatch.setattr(model, "get_tokenizer", lambda: _Tokenizer())
    stage = _stage(model)
    stage.setup()
    stage.setup()
    assert len(calls) == 1
