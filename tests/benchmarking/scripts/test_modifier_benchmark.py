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

import json
import pickle
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

import modifier_benchmark as benchmark
from utils import write_benchmark_results

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.core.client import RayClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, EmptyTask, Task


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """These tests execute real stages locally without a distributed scheduler."""


class SerialExecutor:
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        tasks = initial_tasks or [EmptyTask()]
        for stage in stages:
            stage.setup()
            adapter = BaseStageAdapter(stage)
            tasks = [result for task in tasks for result in adapter.process_batch([task])]
            tasks = pickle.loads(pickle.dumps(tasks))  # noqa: S301
        return tasks


@pytest.mark.parametrize("arrow", [False, True])
def test_measure_text_counts_utf8_without_changing_payload(arrow: bool) -> None:
    data = {"text": ["abc", "é", "🙂", ""], "id": [1, 2, 3, 4]}
    payload = pa.table(data) if arrow else pd.DataFrame(data)
    before = payload.copy() if not arrow else payload
    task = DocumentBatch(dataset_name="test", data=payload, _metadata={"source_files": ["input"]})

    assert benchmark.MeasureText("input").process(task) is task
    assert task.data is payload
    assert task.data.equals(before)
    assert task._metadata == {
        "source_files": ["input"],
        "text_volume": {"num_input_documents": 4, "input_text_bytes": 9},
    }
    benchmark.MeasureText("output").process(task)
    assert task._metadata["text_volume"] == {
        "num_input_documents": 4,
        "input_text_bytes": 9,
        "num_output_documents": 4,
        "output_text_bytes": 9,
    }


@pytest.mark.parametrize("text", [None, 123, b"bytes", "\ud800"])
def test_measure_text_rejects_invalid_text(text: object) -> None:
    task = DocumentBatch(dataset_name="test", data=pd.DataFrame({"text": [text]}))
    with pytest.raises((TypeError, UnicodeEncodeError)):
        benchmark.MeasureText("input").process(task)
    assert "text_volume" not in task._metadata


def test_measure_text_requires_text_column() -> None:
    task = DocumentBatch(dataset_name="test", data=pa.table({"other": ["value"]}))
    with pytest.raises(KeyError):
        benchmark.MeasureText("input").process(task)


@pytest.mark.parametrize("input_filetype", ["jsonl", "parquet"])
@pytest.mark.parametrize("size_cap", [None, 1.0])
def test_benchmark_measures_real_pipeline_and_writes_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_filetype: str, size_cap: float | None
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    batches = [["**café**", '"Hello"', "lorem ipsum"], ["", "A\n\n\nB https://example.org", "cafÃ©"]]
    for index, texts in enumerate(batches):
        path = input_path / f"{index}.{input_filetype}"
        if input_filetype == "jsonl":
            path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
        else:
            pq.write_table(pa.table({"text": texts}), path)

    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    ticks = iter([100.0, 104.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    result_path = tmp_path / "results"
    results = benchmark.run_modify_benchmark(
        input_path, tmp_path / "output", "xenna", result_path, size_cap, input_filetype
    )
    metrics = results["metrics"]
    assert metrics["is_success"]
    assert metrics["num_input_documents"] == metrics["num_output_documents"] == 6
    assert metrics["num_documents_processed"] == 6
    assert metrics["num_output_tasks"] == 2
    assert metrics["time_taken_s"] == 4.0
    assert metrics["throughput_docs_per_sec"] == 1.5
    expected = ["café", "Hello", "", "", "A\n\nB ", "café"]
    written = [text for task in results["tasks"] for text in pq.read_table(task.data[0])["text"].to_pylist()]
    assert sorted(written) == sorted(expected)
    assert metrics["input_text_bytes"] == sum(len(text.encode("utf-8")) for batch in batches for text in batch)
    assert metrics["output_text_bytes"] == sum(len(text.encode("utf-8")) for text in expected)

    write_benchmark_results(results, result_path, save_tasks=False)
    saved_metrics = json.loads((result_path / "metrics.json").read_text())
    assert all(saved_metrics[key] == value for key, value in metrics.items())
    assert not (result_path / "tasks.pkl").exists()
    params = json.loads((result_path / "params.json").read_text())
    assert params == results["params"]
    assert params["input_filetype"] == input_filetype
    assert params["dataset_size_gb"] == size_cap
    recipe = params["modifier_recipe"]
    assert recipe[0]["settings"]["remove_if_at_top_or_bottom"] is True
    assert recipe[4]["settings"]["normalization"] is None


def test_size_cap_measures_only_selected_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    selected = input_path / "small.jsonl"
    selected.write_text('{"text": "café"}\n')
    (input_path / "large.jsonl").write_text(json.dumps({"text": None, "padding": "x" * 1000}) + "\n")
    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    results = benchmark.run_modify_benchmark(
        input_path,
        tmp_path / "output",
        "xenna",
        tmp_path / "results",
        dataset_size_gb=selected.stat().st_size / (1024**3),
    )
    metrics = results["metrics"]
    assert metrics["is_success"]
    assert metrics["num_input_documents"] == metrics["num_output_documents"] == 1
    assert metrics["input_text_bytes"] == metrics["output_text_bytes"] == 5


@pytest.mark.parametrize("failure", ["invalid_text", "writer", "count_changed", "missing_counter", "empty"])
def test_failed_benchmark_does_not_report_complete_volumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text": null}\n' if failure == "invalid_text" else '{"text": "hello"}\n')
    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    if failure == "writer":

        def fail_write(*_: object) -> None:
            msg = "Cannot write output"
            raise OSError(msg)

        monkeypatch.setattr(benchmark.ParquetWriter, "write_data", fail_write)
    elif failure in {"count_changed", "missing_counter", "empty"}:
        original_run = benchmark.Pipeline.run

        def corrupt_results(pipeline: benchmark.Pipeline, executor: SerialExecutor) -> list[Task]:
            tasks = original_run(pipeline, executor)
            if failure == "empty":
                return []
            counts = tasks[0]._metadata["text_volume"]
            if failure == "count_changed":
                counts["num_output_documents"] = 0
            else:
                del counts["input_text_bytes"]
            return tasks

        monkeypatch.setattr(benchmark.Pipeline, "run", corrupt_results)

    results = benchmark.run_modify_benchmark(input_path, tmp_path / "output", "xenna", tmp_path / "results")
    metrics = results["metrics"]
    assert not metrics["is_success"]
    assert results["tasks"] == []
    assert metrics["num_documents_processed"] == metrics["throughput_docs_per_sec"] == 0
    for key in ("num_input_documents", "num_output_documents", "input_text_bytes", "output_text_bytes"):
        assert metrics[key] is None


@pytest.mark.parametrize("executor", ["xenna", "ray_data"])
def test_real_backend_preserves_measurements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, executor: str) -> None:
    source = tmp_path / "input"
    source.mkdir()
    batches = [["**café**", '"Hello"', "lorem ipsum"], ["", "A\n\n\nB https://example.org", "cafÃ©"]]
    for index, texts in enumerate(batches):
        (source / f"{index}.jsonl").write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="n519-ray-") as ray_dir:
        client = RayClient(
            num_cpus=16, num_gpus=0, object_store_memory=2 * 1024**3, include_dashboard=False, ray_temp_dir=ray_dir
        )
        client.start()
        try:
            results = benchmark.run_modify_benchmark(source, tmp_path / "output", executor, tmp_path / "results")
            metrics = results["metrics"]
            assert metrics["is_success"]
            assert metrics["num_input_documents"] == metrics["num_output_documents"] == 6
            assert metrics["input_text_bytes"] == 59
            assert metrics["output_text_bytes"] == 20
            assert sum(pq.read_table(task.data[0]).num_rows for task in results["tasks"]) == 6
        finally:
            client.stop()
