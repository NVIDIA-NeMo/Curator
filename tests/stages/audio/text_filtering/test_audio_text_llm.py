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
import gc
import json
import threading
import weakref
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from cloudpickle import dumps, loads
from openai import APIStatusError, APITimeoutError, AsyncOpenAI

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.ray_data.utils import is_actor_stage
from nemo_curator.models.client import openai_client
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
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


def test_agent_contract_supports_remapped_keys_and_optional_language(endpoint: Callable) -> None:
    endpoint(lambda _request: _completion("Hello."))
    stage = _stage(
        prompt="Rewrite {language}: {text}",
        text_key="transcript",
        output_text_key="rewrite",
        language_key="language",
        notes_key="audit",
        skip_me_key="skip_reason",
    )
    assert stage.validate_input(AudioTask(data={"transcript": "hello"}))
    assert not stage.validate_input(AudioTask(data={}))
    contract = assert_agent_ready(
        stage,
        lambda: AudioTask(data={"transcript": "hello"}),
        available_keys={"transcript"},
        expected_cardinality="1:1",
    )
    stage.teardown()

    assert contract.reads.data_keys == ["transcript"]
    assert set(contract.optional_reads.data_keys) == {"language", "audit", "skip_reason", "rewrite"}
    assert set(contract.writes.data_keys) == {"rewrite", "audit"}
    assert contract.key_roles["transcript"] == contract.key_roles["rewrite"] == "text"
    assert contract.accepts_task_type == contract.produces_task_type == "AudioTask"
    assert contract.gates.runtime_secrets == ["api_key"]
    assert contract.gates.lifecycle_side_effects
    assert contract.gates.per_row_independent is False
    assert not contract.gates.requires_gpu


@pytest.mark.parametrize(
    "data", [{}, {"normalized": "hello"}, {"reference": "Hello"}, {"normalized": "hello", "reference": "Hello"}]
)
def test_agent_recovery_contract_accepts_its_empty_input_fallback(endpoint: Callable, data: dict) -> None:
    endpoint(lambda _request: _completion("Hello."))
    stage = _stage(
        mode="recover_entities",
        user_prompt="Original: {ground_truth}\nNormalized: {normalized}",
        text_key="normalized",
        ground_truth_key="reference",
    )
    assert stage.validate_input(AudioTask(data=dict(data)))
    contract = assert_agent_ready(
        stage, lambda: AudioTask(data=dict(data)), available_keys=data, expected_cardinality="1:1"
    )
    stage.teardown()

    assert not contract.reads.data_keys
    assert {"normalized", "reference"} <= set(contract.optional_reads.data_keys)
    assert contract.key_roles["reference"] == "reference_text"


@pytest.mark.parametrize("prediction", ["English", "French", "unparseable"])
@pytest.mark.parametrize("source_language", ["en", None])
def test_agent_language_contract_declares_skip_as_conditional(
    endpoint: Callable, prediction: str, source_language: str | None
) -> None:
    endpoint(lambda _request: _completion(prediction))
    stage = _stage(mode="language_id", output_text_key="detected_language", skip_me_key="skip_reason")
    data = {"text": "hello"}
    if source_language is not None:
        data["source_lang"] = source_language
    assert stage.validate_input(AudioTask(data=dict(data)))
    contract = assert_agent_ready(
        stage,
        lambda: AudioTask(data=dict(data)),
        available_keys=data,
        expected_cardinality="1:1",
    )
    stage.teardown()

    assert "skip_reason" not in contract.writes.data_keys
    assert [write.writes.data_keys for write in contract.conditional_writes] == [["skip_reason"]]
    assert contract.key_roles["detected_language"] == "prediction"


def test_agent_contract_covers_all_fused_operation_keys(endpoint: Callable) -> None:
    endpoint(lambda _request: _completion("Done."))
    stage = _stage(
        prompt="",
        operations=[
            {
                "name": "first",
                "prompt": "Rewrite {language}",
                "text_key": "first_input",
                "output_text_key": "first_output",
            },
            {"name": "second", "prompt": "Rewrite", "text_key": "second_input", "output_text_key": "second_output"},
        ],
    )
    assert stage.validate_input(AudioTask(data={"first_input": "hello", "second_input": "world"}))
    assert not stage.validate_input(AudioTask(data={"first_input": "hello"}))
    contract = assert_agent_ready(
        stage,
        lambda: AudioTask(data={"first_input": "hello", "second_input": "world"}),
        available_keys={"first_input", "second_input"},
        expected_cardinality="1:1",
    )
    stage.teardown()

    assert set(contract.reads.data_keys) == {"first_input", "second_input"}
    assert set(contract.writes.data_keys) == {"first_output", "second_output", "additional_notes"}
    assert all(
        contract.key_roles[key] == "text" for key in ("first_input", "second_input", "first_output", "second_output")
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


def test_repeated_batches_reuse_client_and_event_loop_until_teardown(endpoint: Callable) -> None:
    active = maximum_active = 0
    loops = []

    async def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        loops.append(asyncio.get_running_loop())
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
    assert len(clients) == 1
    assert len(set(loops)) == 1
    assert not clients[0].is_closed()
    assert not loops[0].is_closed()
    stage.teardown()
    stage.teardown()
    assert all(client.is_closed() for client in clients)
    assert loops[0].is_closed()


def test_batch_deadline_drains_requests_and_next_batch_recovers(
    endpoint: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled = []
    fail = True
    loops = []

    async def handle(_request: httpx.Request) -> httpx.Response:
        loops.append(asyncio.get_running_loop())
        if not fail:
            return _completion("recovered")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    clients = endpoint(handle)
    stage = _stage(max_retries=0)
    monkeypatch.setattr(stage, "_batch_timeout", lambda _count, _client: 0.1)
    tasks = [AudioTask(data={"text": "original", "processed_text": "old"}) for _ in range(2)]
    with pytest.raises(TimeoutError):
        stage.process_batch(tasks)
    assert len(cancelled) == 2
    assert all(task.data == {"text": "original", "processed_text": "old"} for task in tasks)
    assert clients[0].is_closed()
    assert loops[0].is_closed()
    fail = False
    stage.process_batch(tasks)
    assert len(clients) == 2
    assert loops[-1] is not loops[0]
    assert all(task.data["processed_text"] == "recovered" for task in tasks)
    stage.teardown()


@pytest.mark.parametrize("used", [False, True])
def test_serialized_stage_has_independent_lazy_runtime_and_remains_a_ray_data_task(
    endpoint: Callable, used: bool
) -> None:
    clients = endpoint(lambda _request: _completion("processed"))
    stage = _stage(num_workers_override=12)
    if used:
        stage.process(AudioTask(data={"text": "original"}))
    restored = loads(dumps(stage))
    assert not is_actor_stage(stage)
    assert not is_actor_stage(restored)
    assert restored.num_workers() == 12
    assert restored._runner is None
    assert restored._client is None
    restored.process(AudioTask(data={"text": "restored"}))
    assert len(clients) == 1 + used
    restored.teardown()
    if used:
        assert not clients[0].is_closed()
    stage.teardown()
    assert all(client.is_closed() for client in clients)


def test_garbage_collection_closes_runtime_without_retaining_stage(endpoint: Callable) -> None:
    clients = endpoint(lambda _request: _completion("processed"))
    stage = _stage()
    stage.process(AudioTask(data={"text": "original"}))
    loop = stage._runner.get_loop()
    reference = weakref.ref(stage)
    del stage
    gc.collect()
    assert reference() is None
    assert clients[0].is_closed()
    assert loop.is_closed()


def test_async_caller_rejection_preserves_runtime_for_sync_reuse(
    endpoint: Callable, recwarn: pytest.WarningsRecorder
) -> None:
    clients = endpoint(lambda _request: _completion("processed"))
    stage = _stage()
    stage.process(AudioTask(data={"text": "original"}))
    loop = stage._runner.get_loop()

    async def invalid_calls() -> None:
        with pytest.raises(RuntimeError, match="synchronous worker thread"):
            stage.process(AudioTask(data={"text": "async"}))
        with pytest.raises(RuntimeError, match="synchronous worker thread"):
            stage.teardown()

    asyncio.run(invalid_calls())
    assert not clients[0].is_closed()
    assert not loop.is_closed()
    assert stage.process(AudioTask(data={"text": "sync"})).data["processed_text"] == "processed"
    assert len(clients) == 1
    stage.teardown()
    assert clients[0].is_closed()
    assert loop.is_closed()
    gc.collect()
    assert not any(issubclass(warning.category, RuntimeWarning) for warning in recwarn)


def test_concurrent_batch_and_teardown_cannot_close_an_active_runtime(endpoint: Callable) -> None:
    started = threading.Event()
    release = threading.Event()
    errors = []

    async def handle(_request: httpx.Request) -> httpx.Response:
        started.set()
        assert await asyncio.to_thread(release.wait, timeout=5)
        return _completion("processed")

    clients = endpoint(handle)
    stage = _stage()
    task = AudioTask(data={"text": "original"})

    def process() -> None:
        try:
            stage.process(task)
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=process, daemon=True)
    thread.start()
    try:
        assert started.wait(timeout=5)
        with pytest.raises(RuntimeError, match="Concurrent batches"):
            stage.process(AudioTask(data={"text": "second"}))
        with pytest.raises(RuntimeError, match="while a batch is running"):
            stage.teardown()
        assert not clients[0].is_closed()
    finally:
        release.set()
        thread.join(timeout=5)
        stage.teardown()
    assert not thread.is_alive()
    assert errors == []
    assert task.data["processed_text"] == "processed"


def test_timeout_preserves_existing_client_retry_policy(endpoint: Callable) -> None:
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        message = ""
        raise httpx.ReadTimeout(message)

    clients = endpoint(handle)
    stage = _stage(max_retries=0)
    task = AudioTask(data={"text": "original"})
    with pytest.raises(APITimeoutError, match="timed out"):
        stage.process(task)
    assert attempts == clients[0].max_retries + 1
    assert task.data == {"text": "original"}
    assert clients[0].is_closed()


def test_cleanup_failure_preserves_original_request_error(endpoint: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    clients = endpoint(lambda _request: httpx.Response(400, json={"error": {"message": "invalid request"}}))
    close_client = AsyncOpenAI.close

    async def failing_close(client: AsyncOpenAI) -> None:
        await close_client(client)
        msg = "cleanup failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(AsyncOpenAI, "close", failing_close)
    stage = _stage(max_retries=0)
    task = AudioTask(data={"text": "original"})
    with pytest.raises(APIStatusError, match="invalid request"):
        stage.process(task)
    assert task.data == {"text": "original"}
    assert clients[0].is_closed()


def test_batch_deadline_accounts_for_both_retry_layers_and_waves(
    endpoint: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint(lambda _request: _completion("done"))
    stage = _stage(request_timeout_s=10, max_retries=3, max_concurrent_requests=2)
    batch_timeout = stage._batch_timeout
    deadlines = []

    def record_deadline(num_requests: int, client: openai_client.AsyncOpenAIClient) -> float:
        deadline = batch_timeout(num_requests, client)
        sdk_retries = client.client.max_retries
        per_request = 10 * (sdk_retries + 1) + 60 * sdk_retries
        assert deadline == 3 * (per_request * 4 + 7 + 3) + 120
        deadlines.append(deadline)
        return deadline

    monkeypatch.setattr(stage, "_batch_timeout", record_deadline)
    stage.process_batch([AudioTask(data={"text": str(index)}) for index in range(5)])
    assert len(deadlines) == 1


def test_repeated_batches_reuse_the_http_connection() -> None:
    peers = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            peers.append(self.client_address)
            body = _completion("processed").content
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    stage = _stage(endpoint=f"http://127.0.0.1:{server.server_port}/v1", max_concurrent_requests=1)
    try:
        for _ in range(2):
            tasks = [AudioTask(data={"text": text}) for text in ("first", "second")]
            assert all(task.data["processed_text"] == "processed" for task in stage.process_batch(tasks))
        assert len(peers) == 4
        assert len(set(peers)) == 1
    finally:
        stage.teardown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


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


@pytest.mark.parametrize("mode", ["text", "recover_entities"])
def test_skip_existing_and_empty_rows_make_no_requests(mode: str, endpoint: Callable) -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Skipped and empty rows must not reach the endpoint")

    clients = endpoint(unexpected_request)
    tasks = [
        AudioTask(data={"_skipme": "prior filter", "processed_text": "keep"}),
        AudioTask(data={"_skipme": "prior filter"}),
        AudioTask(data={"text": "", "processed_text": "stale"}),
        AudioTask(data={"text": "already done", "granary_v1_prediction": "original", "processed_text": "cached"}),
    ]
    stage = _stage(mode=mode, user_prompt="{ground_truth}\n{normalized}", skip_if_output_exists=True)
    result = stage.process_batch(tasks)
    assert all(actual is original for actual, original in zip(result, tasks, strict=True))
    assert [task.data["processed_text"] for task in result] == ["keep", "", "stale", "cached"]
    empty_task = AudioTask(data={"text": ""})
    assert stage.process(empty_task).data["processed_text"] == ""
    assert clients == []


@pytest.mark.parametrize(
    ("output_text_key", "existing_output", "expected"),
    [
        ("text", {}, "Keep this original transcript"),
        ("processed_text", {"processed_text": "saved output"}, "saved output"),
        ("processed_text", {}, ""),
    ],
)
def test_flagged_rows_preserve_existing_output(
    output_text_key: str, existing_output: dict, expected: str, endpoint: Callable
) -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Flagged rows must not reach the endpoint")

    clients = endpoint(unexpected_request)
    original = {"text": "Keep this original transcript", "_skipme": "prior filter", **existing_output}
    task = AudioTask(data={**original, "additional_notes": {"earlier_stage": "keep"}})
    stage = _stage(output_text_key=output_text_key)

    assert stage.process(task) is task
    assert task.data == {
        **original,
        output_text_key: expected,
        "additional_notes": {"earlier_stage": "keep", stage.name: "skipped (flagged)"},
    }
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
        ("Fix {language_rules}: {text}", {"text": "hello"}, {"hi": "Hindi only"}),
        ("Fix {language_rules}: {text}", {"text": "hello", "source_lang": "en"}, {"hi": "Hindi only"}),
    ],
)
def test_missing_required_prompt_context_raises(
    prompt: str, data: dict, rules: dict | None, endpoint: Callable
) -> None:
    clients = endpoint(lambda _request: _completion("unused"))
    stage = _stage(prompt=prompt, language_rules=rules)
    task = AudioTask(data=data)
    assert stage.validate_input(task) is ("source_lang" in data)
    with pytest.raises((ValueError, KeyError)):
        stage.process(task)
    assert clients == []


@pytest.mark.parametrize(
    ("content", "finish_reason"), [("", "stop"), ("  ", "stop"), (None, "stop"), ("partial", "length")]
)
@pytest.mark.parametrize("validation_mode", ["none", "itn"])
def test_completions_follow_reference_validation_policy(
    content: str | None, finish_reason: str, endpoint: Callable, validation_mode: str
) -> None:
    endpoint(lambda _request: _completion(content, finish_reason))
    task = AudioTask(data={"text": "original", "additional_notes": {"previous": "keep"}})
    stage = _stage(name="punctuation", validation_mode=validation_mode)
    stage.process(task)
    assert task.data["processed_text"] == ("original" if validation_mode == "itn" else (content or "").strip())
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


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_retryable_http_failure_recovers_and_reuses_client(endpoint: Callable, status: int) -> None:
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status, json={"error": {"message": "retry later", "type": "server_error"}})
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


def test_missing_plain_language_uses_reference_english_default() -> None:
    stage = _stage(prompt="Fix {language}: {text}")
    assert stage._build_messages("hello", {}) == [{"role": "user", "content": "Fix English: hello"}]
    assert stage._build_messages("hello", {"source_lang": " hi "}) == [{"role": "user", "content": "Fix hi: hello"}]


def test_recovery_renders_two_fields_cleans_output_and_preserves_skip_inputs(endpoint: Callable) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _completion('<think>reasoning</think>\n```text\n"NVIDIA paid $5"\n```')

    endpoint(handle)
    stage = _stage(
        mode="recover_entities",
        text_key="abbreviated_text",
        output_text_key="entity_recovered_text",
        user_prompt='INITIAL:\n"{ground_truth}"\n\nNORMALIZED:\n"{normalized}"',
        generation_config={"max_tokens": 1024},
    )
    data = {"abbreviated_text": " nvidia paid $5 ", "granary_v1_prediction": " NVIDIA paid five dollars "}
    tasks = [
        AudioTask(data=data),
        AudioTask(data={**data, "_skipme": True}),
        AudioTask(data={"abbreviated_text": "keep"}),
    ]
    stage.process_batch(tasks)
    assert requests[0]["messages"] == [
        {"role": "system", "content": "Add punctuation."},
        {"role": "user", "content": 'INITIAL:\n"NVIDIA paid five dollars"\n\nNORMALIZED:\n"nvidia paid $5"'},
    ]
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == 1024
    assert [task.data["entity_recovered_text"] for task in tasks] == [
        "NVIDIA paid $5",
        data["abbreviated_text"],
        "keep",
    ]
    assert [task.data["additional_notes"][stage.name] for task in tasks] == [
        "recovered",
        "skipped (flagged)",
        "skipped (empty)",
    ]
    assert all(tasks[0].data[key] == value for key, value in data.items())


@pytest.mark.parametrize("response", ["<think>only reasoning</think>", '"same"'])
def test_recovery_equal_inputs_still_call_model_and_empty_output_falls_back(response: str, endpoint: Callable) -> None:
    clients = endpoint(lambda _request: _completion(response))
    stage = _stage(mode="recover_entities", user_prompt="{ground_truth}\n{normalized}")
    task = stage.process(AudioTask(data={"text": "same", "granary_v1_prediction": "same"}))
    assert len(clients) == 1
    assert task.data["processed_text"] == "same"
    assert task.data["additional_notes"][stage.name] == (
        "unchanged" if response == '"same"' else "fallback (empty output)"
    )


def test_parallel_prompts_share_concurrency_and_keep_per_operation_configs(endpoint: Callable) -> None:
    active = maximum = 0
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        body = json.loads(request.content)
        requests.append(body)
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _completion(body["messages"][0]["content"])

    clients = endpoint(handle)
    stage = _stage(
        prompt="",
        max_concurrent_requests=2,
        generation_config={"max_tokens": 10},
        operations=[
            {"name": "caption", "prompt": "Caption", "output_text_key": "caption"},
            {"name": "qa", "prompt": "QA", "output_text_key": "qa", "generation_config": {"max_tokens": 20}},
        ],
    )
    tasks = [AudioTask(data={"text": f"row {i}"}) for i in range(3)]
    assert stage.process_batch(tasks) is tasks
    assert len(clients) == 1
    assert not clients[0].is_closed()
    assert maximum == 2
    assert {body["max_tokens"] for body in requests} == {10, 20}
    assert all(task.data["caption"] == "Caption" and task.data["qa"] == "QA" for task in tasks)
    assert all(set(task.data["additional_notes"]) == {"caption", "qa"} for task in tasks)
    stage.teardown()


def test_parallel_http_failure_is_atomic_across_operations(endpoint: Callable) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][0]["content"] == "Fail":
            return httpx.Response(400, json={"error": {"message": "bad", "type": "invalid_request_error"}})
        return _completion("changed")

    clients = endpoint(handle)
    stage = _stage(
        prompt="",
        max_retries=0,
        operations=[
            {"name": "caption", "prompt": "Caption", "output_text_key": "caption"},
            {"name": "qa", "prompt": "Fail", "output_text_key": "qa"},
        ],
    )
    task = AudioTask(data={"text": "original"})
    with pytest.raises(APIStatusError):
        stage.process(task)
    assert task.data == {"text": "original"}
    assert clients[0].is_closed()


@pytest.mark.parametrize(
    "second",
    [
        {"name": "first", "prompt": "same name", "output_text_key": "other"},
        {"name": "second", "prompt": "same output", "output_text_key": "out"},
        {"name": "second", "prompt": "dependent", "text_key": "out", "output_text_key": "other"},
        {"name": "second", "prompt": "different endpoint", "endpoint": "http://other/v1"},
        {"name": "second", "prompt": "gating", "mode": "language_id"},
    ],
)
def test_parallel_groups_reject_conflicting_or_dependent_operations(second: dict) -> None:
    with pytest.raises(ValueError, match="Parallel operations"):
        _stage(prompt="", operations=[{"name": "first", "prompt": "First", "output_text_key": "out"}, second])
