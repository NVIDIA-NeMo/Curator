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

import hashlib
import json
import pickle
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
    "operation.validation_mode=itn",
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
        assert initial_tasks
        self.state.stages = stages
        self.state.inputs = initial_tasks
        self.state.events.append("execute")
        outputs = initial_tasks
        try:
            for stage in stages:
                assert isinstance(stage, AudioTextLLMStage)
                adapter = BaseStageAdapter(stage)
                stage.setup()
                inputs, outputs = outputs, []
                try:
                    for start in range(0, len(inputs), stage.batch_size):
                        self.state.clock += 2
                        outputs.extend(adapter.process_batch(inputs[start : start + stage.batch_size]))
                finally:
                    stage.teardown()
            if self.state.damage == "drop":
                return outputs[:-1]
            if self.state.damage == "duplicate":
                return [outputs[0]] * len(outputs)
            if self.state.damage == "metadata":
                outputs[0].data["nested"]["keep"].append(3)
            if self.state.damage == "attempts":
                outputs[0]._stage_perf[0].custom_metrics["num_request_attempts"] += 1
            if self.state.damage == "metrics":
                outputs[0]._stage_perf[0].custom_metrics.clear()
            if self.state.serialize:
                outputs = [pickle.loads(pickle.dumps(task)) for task in outputs]  # noqa: S301
            return list(reversed(outputs)) if self.state.reverse else outputs
        finally:
            self.state.events.append("executor:done")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:  # noqa: C901
    state = SimpleNamespace(
        clock=0,
        events=[],
        clients=[],
        requests=[],
        status=200,
        damage=None,
        reverse=False,
        full=False,
        serialize=False,
        retry_once=False,
        usage=True,
    )

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state.requests.append(body)
        assert request.url == "http://dynamo.test/v1/chat/completions"
        if state.retry_once and len(state.requests) == 1:
            return httpx.Response(429, json={"error": {"message": "retry later", "type": "rate_limit_error"}})
        if state.status != 200:
            return httpx.Response(
                state.status, json={"error": {"message": "invalid request", "type": "invalid_request"}}
            )
        text = body["messages"][-1]["content"]
        content = None if text == "fallback" else text.title() + "."
        if state.full:
            operation = body["messages"][0]["content"]
            if operation == "RECOVER":
                content = "यह गलत" if "यह गलत" in text else '<think>checking</think>\n"राम आया"'
            elif operation == "PNC":
                content = text + "।"
            elif operation == "LID":
                content = (
                    "Primary: English\nLanguages: English" if "गलत" in text else "Primary: Hindi\nLanguages: Hindi"
                )
            else:
                expected = "राम आया।" if operation in {"TN", "DISFLUENCY"} else "राम घर आया।"
                assert text == expected
                content = {
                    "TN": "राम घर आया।",
                    "CAPTION": "यह विवरण है।",
                    "CODE": "राम घर आया।",
                    "QA": "(कौन आया?, राम)",
                    "ITN": "राम आया।",
                    "DISFLUENCY": "राम आया",
                }[operation]
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13} if state.usage else None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": content,
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
    path.write_text(
        "".join(json.dumps({"audio_item_id": f"row-{index}", **row}) + "\n" for index, row in enumerate(rows))
    )
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

    manifest = _manifest(tmp_path, rows)
    result = benchmark.run_audio_text_llm_benchmark(manifest, overrides, config_name="single_operation")

    assert {key: value for key, value in result["metrics"].items() if key != "stage_metrics"} == {
        "is_success": True,
        "time_taken_s": 6,
        "inference_server_startup_s": 13 if managed_server else 0,
        "num_input_rows": 5,
        "num_output_rows": 5,
        "input_output_coverage_ratio": 1.0,
        "input_manifest_bytes": Path(manifest).stat().st_size,
        "input_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "num_operations": 1,
        "num_requests": 3,
        "num_request_attempts": 3,
        "num_retries": 0,
        "num_usage_reports": 3,
        "prompt_tokens": 30,
        "completion_tokens": 9,
        "token_usage_complete": True,
        "num_applied": 2,
        "num_fallbacks": 1,
        "num_skipped": 2,
        "throughput_requests_per_sec": 0.5,
        "throughput_request_attempts_per_sec": 0.5,
        "throughput_completion_tokens_per_sec": 1.5,
        "throughput_rows_per_sec": 5 / 6,
        "fallback_ratio": 1 / 3,
        "operations": {"PnCRestoration": {"num_requests": 3, "num_applied": 2, "num_fallbacks": 1, "num_skipped": 2}},
    }
    tasks = result["tasks"]
    assert [task.data["pnc_text"] for task in tasks] == ["Hello.", "fallback", "", " ", "Last."]
    assert all(task is original for task, original in zip(tasks, runtime.inputs, strict=True))
    assert tasks[0].data["nested"] == rows[0]["nested"]
    assert tasks[0].data["audio_filepath"] == "untouched.wav"
    assert all(len(task._stage_perf) == 1 for task in tasks)
    assert result["params"]["stages"][0]["text_key"] == "transcript"
    assert result["params"]["stages"][0]["validation_mode"] == "itn"
    stage_metrics = result["metrics"]["stage_metrics"]["PnCRestoration"]
    assert stage_metrics["num_batches"] == 3
    assert stage_metrics["num_requests"] == stage_metrics["num_request_attempts"] == 3
    assert stage_metrics["process_time_s"] >= 0
    assert len(runtime.clients) == 1
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


def test_retry_accounting_survives_independent_row_serialization(tmp_path: Path, runtime: SimpleNamespace) -> None:
    runtime.retry_once = True
    runtime.serialize = True
    rows = [{"transcript": "first"}, {"transcript": "second"}, {"transcript": "third"}]
    result = benchmark.run_audio_text_llm_benchmark(
        _manifest(tmp_path, rows), [*_OVERRIDES, "max_retries=1"], config_name="single_operation"
    )
    metrics = result["metrics"]
    assert metrics["num_requests"] == 3
    assert metrics["num_request_attempts"] == len(runtime.requests) == 4
    assert metrics["num_retries"] == 1
    assert metrics["prompt_tokens"] == 30
    assert metrics["completion_tokens"] == 9
    assert metrics["stage_metrics"]["PnCRestoration"]["num_batches"] == 2
    perfs = [task._stage_perf[0] for task in result["tasks"]]
    assert perfs[0] is not perfs[1]
    assert perfs[0].batch_id == perfs[1].batch_id
    assert perfs[0].batch_id != perfs[2].batch_id


def test_missing_server_usage_does_not_claim_token_throughput(tmp_path: Path, runtime: SimpleNamespace) -> None:
    runtime.usage = False
    result = benchmark.run_audio_text_llm_benchmark(
        _manifest(tmp_path, [{"transcript": "hello"}]), _OVERRIDES, config_name="single_operation"
    )
    assert result["metrics"]["token_usage_complete"] is False
    assert result["metrics"]["num_usage_reports"] == 0
    assert result["metrics"]["throughput_completion_tokens_per_sec"] is None


@pytest.mark.parametrize("damage", ["attempts", "metrics"])
def test_missing_or_inconsistent_request_metrics_cannot_report_success(
    tmp_path: Path, runtime: SimpleNamespace, damage: str
) -> None:
    runtime.damage = damage
    with pytest.raises(RuntimeError, match=r"accounting disagrees|Missing inference metrics"):
        benchmark.run_audio_text_llm_benchmark(
            _manifest(tmp_path, [{"transcript": "hello"}]), _OVERRIDES, config_name="single_operation"
        )


@pytest.mark.parametrize("all_rejected", [False, True])
def test_full_graph_processes_recovery_dependencies_and_language_skips(
    tmp_path: Path, runtime: SimpleNamespace, all_rejected: bool
) -> None:
    runtime.full = True
    runtime.reverse = True
    rows = [
        {
            "audio_item_id": "wrong",
            "granary_v1_prediction": "यह गलत",
            "abbreviated_text": "यह गलत",
            "source_lang": "hi",
        },
        {
            "audio_item_id": "flagged",
            "granary_v1_prediction": "मूल",
            "abbreviated_text": "मूल",
            "source_lang": "hi",
            "_skipme": "upstream",
        },
    ]
    if not all_rejected:
        rows.extend(
            [
                {
                    "audio_item_id": "equal",
                    "granary_v1_prediction": "राम आया",
                    "abbreviated_text": "राम आया",
                    "source_lang": "hi",
                },
                {
                    "audio_item_id": "repaired",
                    "granary_v1_prediction": "राम आया",
                    "abbreviated_text": "रम आया",
                    "source_lang": "hi",
                    "additional_notes": {"upstream": "keep"},
                },
            ]
        )
    overrides = [
        "endpoint=http://dynamo.test/v1",
        "model_name=test-model",
        "batch_size=2",
        "max_concurrent_requests=2",
    ]
    prompts = {
        "recover_entities": "RECOVER",
        "pnc": "PNC",
        "language_id": "LID",
        "tn": "TN",
        "captioning": "CAPTION",
        "code_switching": "CODE",
        "speech_qa": "QA",
        "itn": "ITN",
        "disfluency": "DISFLUENCY",
    }
    overrides.extend(f"{key}.prompt={prompt}" for key, prompt in prompts.items())
    manifest = _manifest(tmp_path, rows)

    result = benchmark.run_audio_text_llm_benchmark(
        manifest,
        overrides,
        expected_num_rows=len(rows),
        expected_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    )

    outputs = {task.data["audio_item_id"]: task.data for task in result["tasks"]}
    metrics = result["metrics"]
    assert len(runtime.stages) == 7
    assert metrics["num_operations"] == 9
    assert metrics["num_requests"] == len(runtime.requests) == (3 if all_rejected else 21)
    assert metrics["num_applied"] == metrics["num_requests"]
    assert metrics["num_fallbacks"] == 0
    assert metrics["num_skipped"] == 15
    assert outputs["wrong"]["_skipme"] == "Wrong language:LLMLanguageVerification"
    assert outputs["flagged"]["_skipme"] == "upstream"
    assert outputs["flagged"]["entity_recovered_text"] == "मूल"
    assert outputs["wrong"]["tn_raw"] == outputs["flagged"]["pnc_text"] == ""
    assert metrics["operations"]["TextNormalization"]["num_requests"] == (0 if all_rejected else 2)
    if not all_rejected:
        assert outputs["equal"]["additional_notes"]["RecoverEntities"] == "unchanged"
        assert outputs["repaired"]["additional_notes"]["RecoverEntities"] == "recovered"
        assert outputs["repaired"]["additional_notes"]["upstream"] == "keep"
        assert outputs["repaired"]["abbreviated_text"] == "रम आया"
        assert outputs["repaired"]["granary_v1_prediction"] == "राम आया"
        assert outputs["repaired"]["itn_clean"] == "राम आया"
        assert outputs["repaired"]["captioning_text"] == "यह विवरण है।"
    assert all(client.is_closed() for client in runtime.clients)
    assert "server:start" not in runtime.events


@pytest.mark.parametrize("invalid", ["duplicate", "count", "hash"])
def test_cohort_preflight_rejects_mismatch_before_startup(
    tmp_path: Path, runtime: SimpleNamespace, invalid: str
) -> None:
    rows = [{"transcript": "hello"}, {"transcript": "last"}]
    kwargs = {}
    if invalid == "duplicate":
        rows[0]["audio_item_id"] = rows[1]["audio_item_id"] = "duplicate"
    elif invalid == "count":
        kwargs["expected_num_rows"] = 3
    else:
        kwargs["expected_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"unique nonempty string|Expected 3 input rows|SHA-256"):
        benchmark.run_audio_text_llm_benchmark(
            _manifest(tmp_path, rows), _OVERRIDES, config_name="single_operation", **kwargs
        )
    assert runtime.events == ["read"]
    assert runtime.requests == []


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
        "--config-name=single_operation",
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
        benchmark.run_audio_text_llm_benchmark(_manifest(tmp_path, rows), _OVERRIDES, config_name="single_operation")

    assert runtime.requests == []
    assert runtime.clients == []
    assert runtime.events == (
        ["read"]
        if not rows
        else ["read", "ray:start", "server:start", "execute", "executor:done", "server:stop", "ray:stop"]
    )


@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        ("drop", RuntimeError, "returned 1 rows for 2 input rows"),
        ("duplicate", ValueError, "unique nonempty string"),
        ("metadata", RuntimeError, "changed input field"),
    ],
)
def test_corrupt_output_fails_after_real_inference(
    tmp_path: Path, runtime: SimpleNamespace, damage: str, error: type[Exception], message: str
) -> None:
    runtime.damage = damage
    rows = [{"transcript": "hello", "nested": {"keep": [1, 2]}}, {"transcript": "last"}]
    with pytest.raises(error, match=message):
        benchmark.run_audio_text_llm_benchmark(_manifest(tmp_path, rows), _OVERRIDES, config_name="single_operation")

    assert len(runtime.requests) == 2
    assert runtime.events[-2:] == ["server:stop", "ray:stop"]


@pytest.mark.parametrize(
    ("task", "error", "message"),
    [
        (object(), TypeError, "AudioTask with string output"),
        (AudioTask(data={}), TypeError, "AudioTask with string output"),
        (AudioTask(data={"processed_text": 42}), TypeError, "AudioTask with string output"),
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
        benchmark._collect_output_metrics([task], [stage], 1)
