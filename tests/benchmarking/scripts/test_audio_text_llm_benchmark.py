# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import DictConfig
from openai import AsyncOpenAI

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.models.client import openai_client
from nemo_curator.stages.audio.text_filtering import AudioTextLLMStage
from nemo_curator.tasks import AudioTask, FileGroupTask, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))

import audio_text_llm_benchmark as benchmark

_OVERRIDES = [
    "model_name=test-model",
    "text_key=transcript",
    "operation.prompt=Add punctuation.",
    "operation.validation_mode=none",
    "batch_size=2",
    "max_concurrent_requests=1",
]


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """These tests use a local executor and must not start a Ray cluster."""


class _LocalExecutor(BaseExecutor):
    def __init__(self, state: SimpleNamespace):
        super().__init__()
        self.state = state

    def execute(self, stages: list[AudioTextLLMStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        assert len(stages) == 1
        stage = stages[0]
        assert isinstance(stage, AudioTextLLMStage)
        assert initial_tasks
        self.state.stage = stage
        self.state.inputs = initial_tasks
        self.state.events.append("execute")
        adapter = BaseStageAdapter(stage)
        outputs = []
        stage.setup()
        try:
            for start in range(0, len(initial_tasks), stage.batch_size):
                self.state.clock += 2
                outputs.extend(adapter.process_batch(initial_tasks[start : start + stage.batch_size]))
            return outputs[:-1] if self.state.drop_last else outputs
        finally:
            stage.teardown()
            self.state.events.append("executor:done")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(clock=0, events=[], clients=[], requests=[], status=200, drop_last=False)

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state.requests.append(body)
        assert request.url == "http://dynamo.test/v1/chat/completions"
        if state.status != 200:
            return httpx.Response(
                state.status, json={"error": {"message": "invalid request", "type": "invalid_request"}}
            )
        text = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None if text == "fallback" else text.title() + ".",
                        },
                    }
                ],
            },
        )

    def make_client(**kwargs) -> AsyncOpenAI:
        client = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        state.clients.append(client)
        return client

    @contextmanager
    def resource(name: str, startup: int) -> Iterator[SimpleNamespace]:
        state.events.append(f"{name}:start")
        state.clock += startup
        try:
            yield SimpleNamespace(endpoint="http://dynamo.test/v1")
        finally:
            state.clock += 100
            state.events.append(f"{name}:stop")

    original_instantiate = benchmark.hydra.utils.instantiate

    def instantiate(config: DictConfig | dict, *args, **kwargs) -> object:
        if config.get("_target_") == "nemo_curator.core.serve.InferenceServer":
            return resource("server", 13)
        return original_instantiate(config, *args, **kwargs)

    original_read = benchmark.ManifestReaderStage.process

    def read(stage: benchmark.ManifestReaderStage, task: FileGroupTask) -> list[AudioTask]:
        state.events.append("read")
        state.clock += 7
        return original_read(stage, task)

    monkeypatch.setattr(openai_client, "AsyncOpenAI", make_client)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: state.clock))
    monkeypatch.setattr(benchmark, "create_ray_client_from_yaml", lambda _cfg: resource("ray", 11))
    monkeypatch.setattr(benchmark, "create_executor_from_yaml", lambda _cfg: _LocalExecutor(state))
    monkeypatch.setattr(benchmark.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(benchmark.ManifestReaderStage, "process", read)
    return state


def _manifest(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


@pytest.mark.parametrize("managed_server", [False, True])
def test_real_stage_accounting_preserves_rows_and_excludes_startup(
    tmp_path: Path, runtime: SimpleNamespace, managed_server: bool
) -> None:
    rows = [
        {"transcript": "hello", "audio_filepath": "untouched.wav", "nested": {"keep": [1, 2]}},
        {"transcript": "fallback"},
        {"_skipme": True},
        {"transcript": " "},
        {"transcript": "last"},
    ]
    overrides = [*_OVERRIDES, *([] if managed_server else ["endpoint=http://dynamo.test/v1"])]

    result = benchmark.run_audio_text_llm_benchmark(_manifest(tmp_path, rows), overrides)

    assert result["metrics"] == {
        "is_success": True,
        "time_taken_s": 6,
        "inference_server_startup_s": 13 if managed_server else 0,
        "num_input_rows": 5,
        "num_output_rows": 5,
        "num_requests": 3,
        "num_applied": 2,
        "num_fallbacks": 1,
        "num_skipped": 2,
        "throughput_requests_per_sec": 0.5,
        "fallback_ratio": 1 / 3,
    }
    tasks = result["tasks"]
    assert [task.data["pnc_text"] for task in tasks] == ["Hello.", "fallback", "", " ", "Last."]
    assert all(task is original for task, original in zip(tasks, runtime.inputs, strict=True))
    assert tasks[0].data["nested"] == rows[0]["nested"]
    assert tasks[0].data["audio_filepath"] == "untouched.wav"
    assert all(len(task._stage_perf) == 1 for task in tasks)
    assert result["params"]["operation"]["text_key"] == "transcript"
    assert result["params"]["operation"]["validation_mode"] == "none"
    assert len(runtime.clients) == 2
    assert all(client.is_closed() for client in runtime.clients)
    assert [request["messages"][-1]["content"] for request in runtime.requests] == ["hello", "fallback", "last"]
    assert all(request["model"] == "test-model" for request in runtime.requests)
    assert runtime.events == [
        "read",
        "ray:start",
        *(["server:start"] if managed_server else []),
        "execute",
        "executor:done",
        *(["server:stop"] if managed_server else []),
        "ray:stop",
    ]


def test_cli_records_failure_and_closes_owned_resources(
    tmp_path: Path, runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime.status = 400
    output_path = tmp_path / "results"
    argv = [
        "benchmark",
        "--input-manifest",
        _manifest(tmp_path, [{"transcript": "hello"}]),
        "--benchmark-results-path",
        str(output_path),
    ]
    for override in _OVERRIDES:
        argv.extend(["--override", override])
    monkeypatch.setattr(sys, "argv", argv)

    assert benchmark.main() == 1

    metrics = json.loads((output_path / "metrics.json").read_text())
    params = json.loads((output_path / "params.json").read_text())
    assert metrics["is_success"] is False
    assert "invalid request" in metrics["error_message"]
    assert "throughput_requests_per_sec" not in metrics
    assert params["input_manifest"] == str(tmp_path / "input.jsonl")
    assert params["overrides"] == _OVERRIDES
    assert (output_path / "tasks.pkl").is_file()
    assert len(runtime.clients) == 1
    assert all(client.is_closed() for client in runtime.clients)
    assert runtime.events[-3:] == ["executor:done", "server:stop", "ray:stop"]


@pytest.mark.parametrize("rows", [[], [{"_skipme": True}], [{"transcript": " "}]])
def test_empty_workload_cannot_be_reported_as_success(
    tmp_path: Path, runtime: SimpleNamespace, rows: list[dict]
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=r"nonempty input manifest|no inference requests"):
        benchmark.run_audio_text_llm_benchmark(_manifest(tmp_path, rows), _OVERRIDES)

    assert runtime.requests == []
    assert runtime.clients == []
    assert runtime.events == (
        ["read"]
        if not rows
        else ["read", "ray:start", "server:start", "execute", "executor:done", "server:stop", "ray:stop"]
    )


def test_dropped_output_fails_after_real_inference(tmp_path: Path, runtime: SimpleNamespace) -> None:
    runtime.drop_last = True
    with pytest.raises(RuntimeError, match="returned 0 rows for 1 input rows"):
        benchmark.run_audio_text_llm_benchmark(_manifest(tmp_path, [{"transcript": "hello"}]), _OVERRIDES)

    assert len(runtime.requests) == 1
    assert runtime.events[-2:] == ["server:stop", "ray:stop"]


@pytest.mark.parametrize(
    ("task", "error", "message"),
    [
        (object(), TypeError, "AudioTask with a string output"),
        (AudioTask(data={}), TypeError, "AudioTask with a string output"),
        (AudioTask(data={"processed_text": 42}), TypeError, "AudioTask with a string output"),
        (AudioTask(data={"processed_text": "hello"}), RuntimeError, "Missing or invalid processing note"),
        (
            AudioTask(data={"processed_text": "hello", "additional_notes": {"AudioTextLLM": "unrecognized"}}),
            RuntimeError,
            "Missing or invalid processing note",
        ),
    ],
)
def test_invalid_output_cannot_contribute_to_success(task: object, error: type[Exception], message: str) -> None:
    stage = AudioTextLLMStage(endpoint="http://dynamo.test/v1", model_name="test-model", prompt="Add punctuation.")
    with pytest.raises(error, match=message):
        benchmark._collect_output_metrics([task], stage, 1)
