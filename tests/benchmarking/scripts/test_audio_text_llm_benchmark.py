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
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.models.client import openai_client
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.audio.text_filtering import AudioTextLLMStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))

import audio_text_llm_benchmark as benchmark


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """These tests use a local executor and must not start a Ray cluster."""


class _LocalExecutor(BaseExecutor):
    def __init__(self, state: SimpleNamespace):
        super().__init__()
        self.state = state

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        assert initial_tasks is None
        self.state.stages = stages
        self.state.inputs = initial_tasks
        self.state.events.append("execute")
        outputs = [EmptyTask()]
        try:
            for stage in stages:
                adapter = BaseStageAdapter(stage)
                stage.setup()
                inputs, outputs = outputs, []
                try:
                    for start in range(0, len(inputs), stage.batch_size):
                        self.state.clock += 2 if isinstance(stage, AudioTextLLMStage) else 1
                        outputs.extend(adapter.process_batch(inputs[start : start + stage.batch_size]))
                    if isinstance(stage, benchmark.ManifestReaderStage):
                        self.state.inputs = outputs
                finally:
                    stage.teardown()
            if self.state.damage == "drop":
                return outputs[:-1]
            if self.state.damage == "duplicate":
                return [outputs[0]] * len(outputs)
            if self.state.damage == "metadata":
                outputs[0].data["nested"]["keep"].append(3)
            if self.state.serialize:
                outputs = [pickle.loads(pickle.dumps(task)) for task in outputs]  # noqa: S301
            return list(reversed(outputs)) if self.state.reverse else outputs
        finally:
            self.state.events.append("executor:done")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:  # noqa: C901, PLR0915
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
    config = OmegaConf.load(benchmark._CONFIG_DIR / "text_pipeline.yaml")
    config.model_identifier = "test-model"
    config.batch_size = 2
    config.max_concurrent_requests = 1
    config.max_retries = 1
    for stage, prompt in zip(
        config.stages[1:-1], ["RECOVER", "PNC", "LID", "TN", "", "ITN", "DISFLUENCY"], strict=True
    ):
        if prompt:
            stage.prompt = prompt
    config.stages[1].user_prompt = "{normalized}"
    for operation, prompt in zip(config.stages[5].operations, ["CAPTION", "CODE", "QA"], strict=True):
        operation.prompt = prompt
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    OmegaConf.save(config, config_dir / "text_pipeline.yaml")
    monkeypatch.setattr(benchmark, "_CONFIG_DIR", config_dir)

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
        operation = body["messages"][0]["content"]
        content = text
        if operation == "PNC":
            content = None if text == "fallback" else text.title() + "."
        elif operation == "LID":
            content = "Primary: English\nLanguages: English"
        if state.full:
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
            state.server_config = OmegaConf.to_container(config, resolve=True)
            return resource("server", 13)
        return original_instantiate(config, *args, **kwargs)

    original_read = benchmark.ManifestReaderStage.process

    def read(stage: benchmark.ManifestReaderStage, task: FileGroupTask) -> list[AudioTask]:
        state.events.append("read")
        state.clock += 7
        return original_read(stage, task)

    def ray_client(config: DictConfig) -> AbstractContextManager[SimpleNamespace]:
        state.ray_num_gpus = config.ray_client.num_gpus
        return resource("ray", 11)

    def executor(config: DictConfig) -> _LocalExecutor:
        state.backend = config.backend
        return _LocalExecutor(state)

    original_write = ManifestWriterStage.process

    def write(stage: ManifestWriterStage, task: AudioTask) -> AudioTask:
        result = original_write(stage, task)
        if state.damage == "written_output":
            path = Path(stage.output_path)
            path.write_text(path.read_text().replace('"itn_clean": "Hello."', '"itn_clean": "different"'))
        return result

    monkeypatch.setattr(openai_client, "AsyncOpenAI", make_client)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: state.clock))
    monkeypatch.setattr(benchmark, "create_ray_client_from_yaml", ray_client)
    monkeypatch.setattr(benchmark, "create_executor_from_yaml", executor)
    monkeypatch.setattr(benchmark.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(benchmark.ManifestReaderStage, "process", read)
    monkeypatch.setattr(ManifestWriterStage, "process", write)
    return state


def _run(manifest: str, **kwargs) -> dict:
    return benchmark.run_audio_text_llm_benchmark(
        manifest,
        output_dir=str(Path(manifest).parent / "output"),
        cache_root=str(Path(manifest).parent / "cache"),
        **kwargs,
    )


def _manifest(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(
            json.dumps({"audio_item_id": f"row-{index}", "source_lang": "en", **row}) + "\n"
            for index, row in enumerate(rows)
        )
    )
    return str(path)


@pytest.mark.parametrize("managed_server", [False, True])
def test_real_stage_accounting_preserves_rows_and_excludes_startup(
    tmp_path: Path, runtime: SimpleNamespace, managed_server: bool
) -> None:
    rows = [
        {"text": "hello", "abbreviated_text": "hello", "audio_filepath": "untouched.wav", "nested": {"keep": [1, 2]}},
        {"text": "fallback", "abbreviated_text": "fallback"},
        {"_skipme": True},
        {"text": " ", "abbreviated_text": " "},
        {"text": "last", "abbreviated_text": "last"},
    ]
    manifest = _manifest(tmp_path, rows)
    result = _run(manifest, endpoint=None if managed_server else "http://dynamo.test/v1")

    assert runtime.ray_num_gpus == (None if managed_server else 0)
    assert result["metrics"] == {
        "is_success": True,
        "time_taken_s": 56,
        "inference_server_startup_s": 13 if managed_server else 0,
        "num_input_rows": 5,
        "num_output_rows": 5,
        "input_output_coverage_ratio": 1.0,
        "input_manifest_bytes": Path(manifest).stat().st_size,
        "input_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "num_operations": 9,
        "num_requests": 27,
        "num_applied": 26,
        "num_fallbacks": 1,
        "num_skipped": 18,
        "throughput_requests_per_sec": 27 / 56,
        "throughput_rows_per_sec": 5 / 56,
        "fallback_ratio": 1 / 27,
        "operations": {
            operation.name: {
                "num_requests": 3,
                "num_applied": 2 if operation.name == "PnCRestoration" else 3,
                "num_fallbacks": 1 if operation.name == "PnCRestoration" else 0,
                "num_skipped": 2,
            }
            for stage in runtime.stages
            if isinstance(stage, AudioTextLLMStage)
            for operation in stage.operation_stages()
        },
    }
    tasks = result["tasks"]
    assert [task.data["pnc_text"] for task in tasks] == ["Hello.", "fallback", "", " ", "Last."]
    assert all(task.data is original.data for task, original in zip(tasks, runtime.inputs, strict=True))
    assert tasks[0].data["nested"] == rows[0]["nested"]
    assert tasks[0].data["audio_filepath"] == "untouched.wav"
    assert all(len(task._stage_perf) == 10 for task in tasks)
    tutorial = OmegaConf.load(benchmark._CONFIG_DIR / "text_pipeline.yaml")
    tutorial.manifest_path = manifest
    tutorial.output_path = result["params"]["output_manifest"]
    tutorial.endpoint = "http://dynamo.test/v1"
    tutorial.cache_root = str(tmp_path / "cache")
    assert result["params"]["stages"] == OmegaConf.to_container(tutorial.stages, resolve=True)
    assert result["params"]["inference_server"] == (
        OmegaConf.to_container(tutorial.inference_server, resolve=True) if managed_server else None
    )
    written = [json.loads(line) for line in Path(result["params"]["output_manifest"]).read_text().splitlines()]
    assert written == [task.data for task in tasks]
    assert result["params"]["config_path"] == str(benchmark._CONFIG_DIR / "text_pipeline.yaml")
    assert result["params"]["timing_scope"] == "reader_processing_writer"
    assert len(runtime.clients) == 7
    assert all(client.is_closed() for client in runtime.clients)
    assert len(runtime.requests) == 27
    assert all(request["model"] == "test-model" for request in runtime.requests)
    assert runtime.events == [
        "read",
        "ray:start",
        *(["server:start"] if managed_server else []),
        "execute",
        "read",
        "executor:done",
        "read",
        *(["server:stop"] if managed_server else []),
        "ray:stop",
    ]


@pytest.mark.parametrize("usage", [False, True])
def test_logical_counts_survive_retries_and_independent_row_serialization(
    tmp_path: Path, runtime: SimpleNamespace, usage: bool
) -> None:
    runtime.retry_once = True
    runtime.serialize = True
    runtime.usage = usage
    rows = [{"text": text, "abbreviated_text": text} for text in ("first", "second", "third")]
    result = _run(_manifest(tmp_path, rows))
    metrics = result["metrics"]
    assert metrics["num_requests"] == metrics["num_applied"] == 27
    assert metrics["num_output_rows"] == 3
    assert len(runtime.requests) == 28
    assert metrics["throughput_requests_per_sec"] == 27 / 40
    assert "num_request_attempts" not in metrics
    assert "stage_metrics" not in metrics
    assert "throughput_completion_tokens_per_sec" not in metrics
    assert len(runtime.clients) == 7
    assert all(client.is_closed() for client in runtime.clients)


@pytest.mark.parametrize("all_rejected", [False, True])
def test_full_graph_processes_recovery_dependencies_and_language_skips(
    tmp_path: Path, runtime: SimpleNamespace, all_rejected: bool
) -> None:
    runtime.full = True
    runtime.reverse = True
    rows = [
        {
            "audio_item_id": "wrong",
            "text": "यह गलत",
            "abbreviated_text": "यह गलत",
            "source_lang": "hi",
        },
        {
            "audio_item_id": "flagged",
            "text": "मूल",
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
                    "text": "राम आया",
                    "abbreviated_text": "राम आया",
                    "source_lang": "hi",
                },
                {
                    "audio_item_id": "repaired",
                    "text": "राम आया",
                    "abbreviated_text": "रम आया",
                    "source_lang": "hi",
                    "additional_notes": {"upstream": "keep"},
                },
            ]
        )
    manifest = _manifest(tmp_path, rows)

    result = _run(
        manifest,
        endpoint="http://dynamo.test/v1",
        expected_num_rows=len(rows),
        expected_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    )

    outputs = {task.data["audio_item_id"]: task.data for task in result["tasks"]}
    metrics = result["metrics"]
    assert len(runtime.stages) == 10
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
        assert outputs["repaired"]["text"] == "राम आया"
        assert outputs["repaired"]["itn_clean"] == "राम आया"
        assert outputs["repaired"]["captioning_text"] == "यह विवरण है।"
    assert all(client.is_closed() for client in runtime.clients)
    assert "server:start" not in runtime.events


@pytest.mark.parametrize("invalid", ["duplicate", "count", "hash"])
def test_cohort_preflight_rejects_mismatch_before_startup(
    tmp_path: Path, runtime: SimpleNamespace, invalid: str
) -> None:
    rows = [{"text": text, "abbreviated_text": text} for text in ("hello", "last")]
    kwargs = {}
    if invalid == "duplicate":
        rows[0]["audio_item_id"] = rows[1]["audio_item_id"] = "duplicate"
    elif invalid == "count":
        kwargs["expected_num_rows"] = 3
    else:
        kwargs["expected_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"unique nonempty string|Expected 3 input rows|SHA-256"):
        _run(_manifest(tmp_path, rows), **kwargs)
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
        _manifest(tmp_path, [{"text": "hello", "abbreviated_text": "hello"}]),
        "--benchmark-results-path",
        str(output_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--cache-root",
        str(tmp_path / "cache"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert benchmark.main() == 1

    metrics = json.loads((output_path / "metrics.json").read_text())
    params = json.loads((output_path / "params.json").read_text())
    assert metrics["is_success"] is False
    assert "invalid request" in metrics["error_message"]
    assert "throughput_requests_per_sec" not in metrics
    assert params["input_manifest"] == str(tmp_path / "input.jsonl")
    assert params["output_dir"] == str(tmp_path / "output")
    assert "overrides" not in params
    assert (output_path / "tasks.pkl").is_file()
    assert len(runtime.clients) == 1
    assert all(client.is_closed() for client in runtime.clients)
    assert runtime.events[-3:] == ["executor:done", "server:stop", "ray:stop"]


def test_cli_existing_endpoint_writes_complete_output_without_saving_tasks(
    tmp_path: Path, runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, [{"text": "hello", "abbreviated_text": "hello"}])
    results_path = tmp_path / "results"
    results_path.mkdir()
    (results_path / "tasks.pkl").write_bytes(b"old tasks")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--input-manifest",
            manifest,
            "--output-dir",
            str(tmp_path / "output"),
            "--benchmark-results-path",
            str(results_path),
            "--executor",
            "xenna",
            "--endpoint",
            "http://dynamo.test/v1",
            "--model-name",
            "existing-server-model",
            "--no-save-tasks",
        ],
    )

    assert benchmark.main() == 0

    params = json.loads((results_path / "params.json").read_text())
    metrics = json.loads((results_path / "metrics.json").read_text())
    assert metrics["is_success"] is True
    assert params["backend"] == runtime.backend == "xenna"
    assert params["inference_server"] is None
    assert json.loads(Path(params["output_manifest"]).read_text())["itn_clean"] == "Hello."
    assert all(request["model"] == "existing-server-model" for request in runtime.requests)
    assert not (results_path / "tasks.pkl").exists()
    assert "server:start" not in runtime.events


def test_local_model_path_preserves_served_model_name(tmp_path: Path, runtime: SimpleNamespace) -> None:
    result = _run(
        _manifest(tmp_path, [{"text": "hello", "abbreviated_text": "hello"}]),
        model_path="/models/gemma-snapshot",
    )

    model = runtime.server_config["models"][0]
    assert model["model_identifier"] == "/models/gemma-snapshot"
    assert model["model_name"] == result["params"]["model_name"] == "test-model"
    assert all(request["model"] == "test-model" for request in runtime.requests)


def test_existing_output_is_not_overwritten(tmp_path: Path, runtime: SimpleNamespace) -> None:
    manifest = _manifest(tmp_path, [{"text": "hello", "abbreviated_text": "hello"}])
    output = tmp_path / "output" / "text_pipeline.jsonl"
    output.parent.mkdir()
    output.write_text("previous output\n")

    with pytest.raises(ValueError, match="output already exists"):
        _run(manifest)

    assert output.read_text() == "previous output\n"
    assert runtime.events == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "cache_root is required"),
        ({"endpoint": "http://dynamo.test/v1", "model_path": "/models/gemma"}, "cannot reconfigure"),
    ],
)
def test_invalid_serving_arguments_fail_before_startup(
    tmp_path: Path, runtime: SimpleNamespace, kwargs: dict, message: str
) -> None:
    manifest = _manifest(tmp_path, [{"text": "hello", "abbreviated_text": "hello"}])
    with pytest.raises(ValueError, match=message):
        benchmark.run_audio_text_llm_benchmark(manifest, str(tmp_path / "output"), **kwargs)
    assert runtime.events == []


@pytest.mark.parametrize("rows", [[], [{"_skipme": True}], [{"text": " ", "abbreviated_text": " "}]])
def test_empty_workload_cannot_be_reported_as_success(
    tmp_path: Path, runtime: SimpleNamespace, rows: list[dict]
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=r"nonempty input manifest|no inference requests"):
        _run(_manifest(tmp_path, rows))

    assert runtime.requests == []
    assert runtime.clients == []
    assert runtime.events == (
        ["read"]
        if not rows
        else ["read", "ray:start", "server:start", "execute", "read", "executor:done", "server:stop", "ray:stop"]
    )


@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        ("drop", RuntimeError, "returned 1 rows for 2 input rows"),
        ("duplicate", ValueError, "unique nonempty string"),
        ("metadata", RuntimeError, "changed input field"),
        ("written_output", RuntimeError, "Written audio text output does not match"),
    ],
)
def test_corrupt_output_fails_after_real_inference(
    tmp_path: Path, runtime: SimpleNamespace, damage: str, error: type[Exception], message: str
) -> None:
    runtime.damage = damage
    rows = [
        {"text": "hello", "abbreviated_text": "hello", "nested": {"keep": [1, 2]}},
        {"text": "last", "abbreviated_text": "last"},
    ]
    with pytest.raises(error, match=message):
        _run(_manifest(tmp_path, rows))

    assert len(runtime.requests) == 18
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
