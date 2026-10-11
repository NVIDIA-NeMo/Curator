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

# ruff: noqa: INP001

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.models.client import openai_client
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.audio.text_filtering.audio_text_llm import AudioTextLLMStage
from nemo_curator.tasks import AudioTask


def _completion(content: str | None, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
            ],
        },
    )


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Callable:
    def install(handler: Callable) -> list[AsyncOpenAI]:
        clients = []

        def make_client(**kwargs) -> AsyncOpenAI:
            client = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client

        monkeypatch.setattr(openai_client, "AsyncOpenAI", make_client)
        return clients

    return install


def _stage(**kwargs) -> AudioTextLLMStage:
    return AudioTextLLMStage(
        **{"endpoint": "http://dynamo.test/v1", "model_name": "test-model", "prompt": "Add punctuation.", **kwargs}
    )


def test_out_of_order_responses_preserve_audio_payload_and_task_envelope(endpoint: Callable) -> None:
    completed = []

    async def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        if text == "first":
            await asyncio.sleep(0.02)
        completed.append(text)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "test-model"
        return _completion(text + ".")

    clients = endpoint(handle)
    waveform = object()
    segments = [{"start": 0, "end": 1, "text": "first"}]
    task = AudioTask(
        dataset_name="audio-dataset",
        data={"text": "first", "waveform": waveform, "segments": segments, "nested": {"keep": [1, 2]}},
        filepath_key="audio_filepath",
        _metadata={"manifest": "source.jsonl"},
        _stage_perf=["upstream-stat"],
    )
    task._set_task_id("source", 2)
    task._source_id = "partition-2"
    tasks = [task, AudioTask(data={"text": "second"})]

    result = _stage(max_concurrent_requests=2).process_batch(tasks)

    assert completed == ["second", "first"]
    assert all(actual is original for actual, original in zip(result, tasks, strict=True))
    assert [item.data["processed_text"] for item in result] == ["first.", "second."]
    assert task.data["text"] == "first"
    assert task.data["waveform"] is waveform
    assert task.data["segments"] is segments
    assert task.data["nested"] == {"keep": [1, 2]}
    assert (task.dataset_name, task.filepath_key, task.task_id, task._source_id) == (
        "audio-dataset",
        "audio_filepath",
        "source_2",
        "partition-2",
    )
    assert task._metadata == {"manifest": "source.jsonl"}
    assert task._stage_perf == ["upstream-stat"]
    assert clients
    assert all(client.is_closed() for client in clients)


def test_repeated_batches_bound_concurrency_and_close_clients(endpoint: Callable) -> None:
    active = maximum_active = 0

    async def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _completion("done")

    clients = endpoint(handle)
    stage = _stage(max_concurrent_requests=2)
    for _ in range(2):
        tasks = [AudioTask(data={"text": f"row {index}"}) for index in range(5)]
        assert stage.process_batch(tasks) == tasks
    assert maximum_active == 2
    assert len(clients) == 2
    assert all(client.is_closed() for client in clients)


def test_executor_adapter_keeps_per_task_lineage(endpoint: Callable) -> None:
    endpoint(lambda _request: _completion("processed"))
    tasks = [AudioTask(data={"text": "first"}), AudioTask(data={"text": "second"})]
    for index, task in enumerate(tasks):
        task._set_task_id("source", index)
        task._source_id = f"partition-{index}"
        task._metadata["row"] = index

    result = BaseStageAdapter(_stage()).process_batch(tasks)

    assert all(actual is original for actual, original in zip(result, tasks, strict=True))
    assert [task.task_id for task in result] == ["source_0_0", "source_1_0"]
    assert [task._source_id for task in result] == ["partition-0", "partition-1"]
    assert [task._metadata for task in result] == [{"row": 0}, {"row": 1}]
    assert all(len(task._stage_perf) == 1 for task in result)


def test_skip_existing_and_empty_rows_make_no_requests(endpoint: Callable) -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Skipped and empty rows must not reach the endpoint")

    clients = endpoint(unexpected_request)
    tasks = [
        AudioTask(data={"_skipme": "prior filter", "processed_text": "keep"}),
        AudioTask(data={"_skipme": "prior filter"}),
        AudioTask(data={"text": "", "processed_text": "stale"}),
        AudioTask(data={"text": "already done", "processed_text": "cached"}),
    ]
    stage = _stage(skip_if_output_exists=True)
    result = stage.process_batch(tasks)
    assert all(actual is original for actual, original in zip(result, tasks, strict=True))
    assert [task.data["processed_text"] for task in result] == ["keep", "", "stale", "cached"]
    empty_task = AudioTask(data={"text": ""})
    assert stage.process(empty_task).data["processed_text"] == ""
    assert clients == []


@pytest.mark.parametrize("data", [{}, {"text": None}, {"text": 12}, {"text": ["hello"]}])
def test_active_rows_require_a_string_transcript(data: dict, endpoint: Callable) -> None:
    clients = endpoint(lambda _request: _completion("unused"))
    with pytest.raises((TypeError, ValueError, KeyError)):
        _stage().process_batch([AudioTask(data=data)])
    assert clients == []


def test_templated_prompts_resolve_each_rows_language_and_rules(endpoint: Callable) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _completion("accepted")

    endpoint(handle)
    stage = _stage(
        prompt="Fix {language}: {language_rules}\n{text}",
        language_rules={"hi": "Keep Devanagari.", "en": "Keep English."},
        generation_config=GenerationConfig(max_tokens=40, extra_kwargs={"extra_body": {"top_k": 4}}),
    )
    stage.process_batch(
        [
            AudioTask(data={"text": "नमस्ते {text}", "source_lang": "hi"}),
            AudioTask(data={"text": "hello", "source_lang": "en"}),
        ]
    )
    assert [body["messages"] for body in requests] == [
        [{"role": "user", "content": "Fix hi: Keep Devanagari.\nनमस्ते {text}"}],
        [{"role": "user", "content": "Fix en: Keep English.\nhello"}],
    ]
    assert all(body["max_tokens"] == 40 and body["top_k"] == 4 for body in requests)


def test_untemplated_prompt_is_system_message(endpoint: Callable) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _completion("Hello.")

    endpoint(handle)
    task = AudioTask(data={"transcript": "hello {language}"})
    result = _stage(text_key="transcript", output_text_key="punctuated").process(task)
    assert result is task
    assert task.data["punctuated"] == "Hello."
    assert requests[0]["messages"] == [
        {"role": "system", "content": "Add punctuation."},
        {"role": "user", "content": "hello {language}"},
    ]


@pytest.mark.parametrize(
    ("prompt", "data", "rules"),
    [
        ("Fix {language}: {text}", {"text": "hello"}, None),
        ("Fix {language_rules}: {text}", {"text": "hello", "source_lang": "en"}, {"hi": "Hindi only"}),
    ],
)
def test_missing_required_prompt_context_raises(
    prompt: str, data: dict, rules: dict | None, endpoint: Callable
) -> None:
    clients = endpoint(lambda _request: _completion("unused"))
    with pytest.raises((ValueError, KeyError)):
        _stage(prompt=prompt, language_rules=rules).process(AudioTask(data=data))
    assert clients == []


@pytest.mark.parametrize(
    ("content", "finish_reason"), [("", "stop"), ("  ", "stop"), (None, "stop"), ("partial", "length")]
)
def test_invalid_completions_fall_back_with_a_reason(
    content: str | None, finish_reason: str, endpoint: Callable
) -> None:
    endpoint(lambda _request: _completion(content, finish_reason))
    task = AudioTask(data={"text": "original", "additional_notes": {"previous": "keep"}})
    stage = _stage(name="punctuation")
    stage.process(task)
    assert task.data["processed_text"] == "original"
    assert task.data["additional_notes"]["previous"] == "keep"
    assert task.data["additional_notes"][stage.name]


@pytest.mark.parametrize(
    ("mode", "text", "response", "accepted"),
    [
        ("none", "hello", "new words are allowed", True),
        ("itn", "one two three four five six seven eight nine ten", "one two three", True),
        ("itn", "one two three four five six seven eight nine ten", "one two", False),
        ("itn", "one two three four five six seven", "one", True),
        ("itn", "hello world", "hello new world", False),
        ("itn", "hello world", "hello invented", False),
        ("itn", "twenty one", "21", True),
        ("tn", "21", "twenty one", True),
        ("tn", "नमस्ते दुनिया", "namaste duniya", False),
        ("tn", "नमस्ते दुनिया", "नमस्ते दुनिया।", True),
        ("tn", "नमस्ते दुनिया", "Here is the normalized text: नमस्ते दुनिया", False),
    ],
)
def test_validation_preserves_reference_behavior(
    mode: str, text: str, response: str, accepted: bool, endpoint: Callable
) -> None:
    endpoint(lambda _request: _completion(response))
    stage = _stage(validation_mode=mode)
    task = AudioTask(data={"text": text})
    stage.process(task)
    assert task.data["processed_text"] == (response if accepted else text)
    if not accepted:
        assert task.data["additional_notes"][stage.name]


def test_http_failure_does_not_apply_partial_batch_results_and_closes_client(endpoint: Callable) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["messages"][-1]["content"] == "fail":
            return httpx.Response(400, json={"error": {"message": "invalid request", "type": "invalid_request_error"}})
        return _completion("new output")

    clients = endpoint(handle)
    tasks = [
        AudioTask(data={"text": "good", "processed_text": "old output"}),
        AudioTask(data={"text": "fail"}),
        AudioTask(data={"_skipme": "prior filter"}),
        AudioTask(data={"text": ""}),
    ]
    with pytest.raises(APIStatusError):
        _stage(max_retries=0, max_concurrent_requests=1).process_batch(tasks)
    assert tasks[0].data["processed_text"] == "old output"
    assert all("processed_text" not in task.data for task in tasks[1:])
    assert all("additional_notes" not in task.data for task in tasks)
    assert clients
    assert all(client.is_closed() for client in clients)


def test_retryable_http_failure_recovers_and_closes_client(endpoint: Callable) -> None:
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "rate limit", "type": "rate_limit_error"}})
        return _completion("recovered")

    clients = endpoint(handle)
    task = _stage(max_retries=1).process(AudioTask(data={"text": "original"}))
    assert attempts == 2
    assert task.data["processed_text"] == "recovered"
    assert clients
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize(
    "generation_config",
    [
        {"stream": True},
        {"n": 2},
        {"extra_kwargs": {"stream": True}},
        {"extra_kwargs": {"n": 2}},
        {"extra_kwargs": {"extra_body": {"stream": True}}},
        {"extra_kwargs": {"extra_body": {"n": 2}}},
    ],
)
def test_rejects_streaming_or_multiple_completions(generation_config: dict) -> None:
    with pytest.raises(ValueError, match=r"stream|n=1|single|completion"):
        _stage(generation_config=generation_config)


@pytest.mark.parametrize(
    ("options", "error_field"),
    [
        ({"max_retries": True}, "max_retries"),
        ({"max_retries": 1.5}, "max_retries"),
        ({"max_retries": -1}, "max_retries"),
        ({"request_timeout_s": float("nan")}, "request_timeout_s"),
        ({"request_timeout_s": float("inf")}, "request_timeout_s"),
        ({"request_timeout_s": 0}, "request_timeout_s"),
        ({"skip_me_key": ""}, "skip_me_key"),
        ({"notes_key": "text"}, "notes_key"),
        ({"notes_key": "_skipme"}, "notes_key"),
        ({"notes_key": "source_lang"}, "notes_key"),
        ({"notes_key": "processed_text"}, "notes|output_text_key"),
    ],
)
def test_invalid_limits_and_destructive_field_collisions_fail_early(options: dict, error_field: str) -> None:
    with pytest.raises(ValueError, match=error_field):
        _stage(**options)
