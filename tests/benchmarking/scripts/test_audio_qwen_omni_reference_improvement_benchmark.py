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
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import create_executor_from_yaml, create_pipeline_from_yaml
from nemo_curator.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

import audio_qwen_omni_reference_improvement_benchmark as benchmark


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def output_row() -> dict:
    return {
        "audio_filepath": "/audio/clip.flac",
        "text": "HELLO WORLD",
        "primary_model_prediction": "Hello world.",
        "best_prediction": "Hello world.",
        "abbreviated_text": "Hello world.",
        "best_prediction_source": "primary",
        "duration": 2.0,
    }


def test_output_metrics_count_fallbacks_and_skips(tmp_path: Path, output_row: dict) -> None:
    rows = [
        output_row,
        {
            **output_row,
            "audio_filepath": "/audio/reference.flac",
            "best_prediction_source": "reference",
            "best_prediction": "HELLO WORLD",
        },
        {
            **output_row,
            "audio_filepath": "/audio/unsupported.flac",
            "best_prediction_source": "ground_truth",
            "best_prediction": "HELLO WORLD",
        },
        {**output_row, "audio_filepath": "/audio/skip.flac", "_skipme": "Hallucination detected"},
    ]
    input_rows = {row["audio_filepath"]: {"text": row["text"]} for row in rows}
    path = _write_rows(tmp_path / "output.jsonl", rows)

    assert benchmark._collect_output_metrics(path, input_rows) == {
        "num_tasks_processed": 4,
        "output_coverage_ratio": 1.0,
        "reference_preservation_ratio": 1.0,
        "num_reference_fallbacks": 1,
        "num_ground_truth_fallbacks": 1,
        "num_skipped_tasks": 1,
        "total_audio_duration_hours": 8 / 3600,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("text", "Different reference", "Reference text changed"),
        ("text", " ", "Empty or invalid reference"),
        ("primary_model_prediction", "", "Empty or invalid primary_model_prediction"),
        ("best_prediction", None, "Empty or invalid best_prediction"),
        ("abbreviated_text", "\t", "Empty or invalid abbreviated_text"),
        ("duration", 0, "Invalid duration"),
        ("duration", float("inf"), "Invalid duration"),
        ("duration", True, "Invalid duration"),
        ("best_prediction_source", "fallback", "Unexpected prediction source"),
        ("best_prediction_source", "reference", "Reference fallback does not match"),
        ("_skipme", False, "Invalid skip reason"),
    ],
)
def test_rejects_corrupt_output(tmp_path: Path, output_row: dict, field: str, value: object, message: str) -> None:
    input_rows = {output_row["audio_filepath"]: {"text": output_row["text"]}}
    path = _write_rows(tmp_path / "output.jsonl", [{**output_row, field: value}])
    with pytest.raises(ValueError, match=message):
        benchmark._collect_output_metrics(path, input_rows)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "extra"])
def test_rejects_inexact_output_coverage(tmp_path: Path, output_row: dict, fault: str) -> None:
    input_rows = {output_row["audio_filepath"]: {"text": output_row["text"]}}
    rows = [output_row]
    if fault == "duplicate":
        rows.append(output_row)
    elif fault == "missing":
        input_rows["/audio/missing.flac"] = {"text": "Missing"}
    else:
        rows.append({**output_row, "audio_filepath": "/audio/extra.flac"})
    path = _write_rows(tmp_path / "output.jsonl", rows)
    with pytest.raises(ValueError, match=r"Duplicate|missing=1|extra=1"):
        benchmark._collect_output_metrics(path, input_rows)


@pytest.mark.parametrize("contents", ["", "{}\n", "[]\n", "invalid json\n"])
def test_rejects_empty_or_malformed_manifest(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match=r"Manifest contains no rows|Invalid manifest row"):
        benchmark._read_manifest(path)


def test_rejects_duplicate_input_paths(tmp_path: Path, output_row: dict) -> None:
    path = _write_rows(tmp_path / "input.jsonl", [output_row, output_row])
    with pytest.raises(ValueError, match="Duplicate audio_filepath"):
        benchmark._read_manifest(path)


def test_main_reports_pipeline_failure(tmp_path: Path, output_row: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_rows(tmp_path / "input.jsonl", [output_row])
    original_manifest = manifest.read_bytes()
    results = tmp_path / "run"
    pipeline = Mock()
    pipeline.run.side_effect = RuntimeError("worker failed")
    monkeypatch.setattr(benchmark, "create_pipeline_from_yaml", Mock(return_value=pipeline))
    monkeypatch.setattr(benchmark, "create_executor_from_yaml", Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--input-manifest", str(manifest), "--benchmark-results-path", str(results)],
    )

    assert benchmark.main() == 1
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["is_success"] is False
    assert metrics["error_message"] == "worker failed"
    assert manifest.read_bytes() == original_manifest


def test_main_merges_runner_parameters(tmp_path: Path, output_row: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_rows(tmp_path / "input.jsonl", [output_row])
    results = tmp_path / "run"
    results.mkdir()
    (results / "scratch").mkdir()
    runner_params = {"ray_num_gpus": 8, "entry_timeout_s": 7200}
    (results / "params.json").write_text(json.dumps(runner_params))

    def create_pipeline(cfg: DictConfig) -> Mock:
        pipeline = Mock()
        pipeline.run.side_effect = lambda **_kwargs: _write_rows(Path(cfg.output_path), [output_row])
        return pipeline

    monkeypatch.setattr(benchmark, "create_pipeline_from_yaml", create_pipeline)
    monkeypatch.setattr(benchmark, "create_executor_from_yaml", Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--input-manifest", str(manifest), "--benchmark-results-path", str(results)],
    )

    assert benchmark.main() == 0
    params = json.loads((results / "params.json").read_text())
    assert {key: params[key] for key in runner_params} == runner_params
    assert params["input_manifest"] == str(manifest)
    assert json.loads((results / "metrics.json").read_text())["is_success"] is True


@pytest.mark.parametrize("backend", [None, "xenna", "ray_data"])
def test_benchmark_preserves_tutorial_settings(
    tmp_path: Path, output_row: dict, monkeypatch: pytest.MonkeyPatch, backend: str | None
) -> None:
    manifest = _write_rows(tmp_path / "input.jsonl", [output_row])
    results = tmp_path / "run"
    model = tmp_path / "model"
    model.mkdir()
    expected = OmegaConf.load(benchmark._PIPELINE_CONFIG)
    expected.manifest_path = str(manifest)
    expected.output_path = str(results / "results/qwen_omni_reference_improvement.jsonl")
    expected.workspace_dir = str(results / "scratch")
    expected.model_id = str(model)
    if backend is not None:
        expected.backend = backend
    expected_pipeline = create_pipeline_from_yaml(expected)
    expected_executor = create_executor_from_yaml(expected)

    def run(pipeline: Pipeline, executor: object) -> None:
        assert pipeline.stages == expected_pipeline.stages
        assert type(executor) is type(expected_executor)
        if expected.backend == "xenna":
            assert executor.config == expected_executor.config
        _write_rows(Path(expected.output_path), [output_row])

    monkeypatch.setattr(Pipeline, "run", run)
    result = benchmark.run_audio_qwen_omni_reference_improvement_benchmark(
        str(results), str(manifest), model_id=str(model), executor=backend
    )

    assert result["params"]["pipeline_config"] == OmegaConf.to_container(expected, resolve=True)
    assert result["metrics"]["output_coverage_ratio"] == 1.0


@pytest.mark.parametrize("existing", ["results", "scratch", "scratch/audio.wav", "metrics.json", "tasks.pkl"])
@pytest.mark.parametrize("direct_call", [False, True])
def test_preserves_preexisting_outputs(
    tmp_path: Path, existing: str, monkeypatch: pytest.MonkeyPatch, direct_call: bool
) -> None:
    existing_path = tmp_path / existing
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_bytes(b"existing output")
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--input-manifest", "/unused.jsonl", "--benchmark-results-path", str(tmp_path)],
    )
    if direct_call:
        with pytest.raises(FileExistsError):
            benchmark.run_audio_qwen_omni_reference_improvement_benchmark(str(tmp_path), "/unused.jsonl")
    else:
        assert benchmark.main() == 1
    assert existing_path.read_bytes() == b"existing output"
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [existing_path]
