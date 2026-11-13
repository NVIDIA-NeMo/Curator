# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Check interpreter identity and managed benchmark cleanup without serving."""

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


def test_requested_venv_symlink_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("run_dynamo_uncapped")
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    execute = Mock()
    monkeypatch.setattr(module.os, "execve", execute)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("NEMO_CURATOR_DYNAMO_CUDA_TAG", "cu129")
    module.ensure_cuda13_python(SimpleNamespace(python=interpreter, curator_root=tmp_path, cuda_tag="cu130"))
    assert execute.call_args.args[0] == str(interpreter)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(interpreter.parent)


@pytest.mark.parametrize("failure", ["startup", "pipeline"])
def test_managed_failure_always_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("run_dynamo_uncapped")
    sampler = SimpleNamespace(start=Mock(), stop=Mock())
    monkeypatch.setattr(module, "GPUSampler", lambda *_args: sampler)
    backend = SimpleNamespace(stop=Mock())
    server = SimpleNamespace(_started=False, _backend_impl=backend, stop=Mock())

    def start() -> None:
        if failure == "startup":
            msg = "startup failed"
            raise RuntimeError(msg)
        server._started = True

    server.start = start
    pipeline = Mock(side_effect=ValueError("pipeline construction failed"))
    with pytest.raises((RuntimeError, ValueError), match="failed"):
        module.execute_run(
            server, pipeline, None, run_dir=tmp_path, topology={}, compute_metrics=Mock(), count_pdfs=Mock()
        )
    server.stop.assert_called_once()
    if failure == "startup":
        backend.stop.assert_called_once()
    sampler.stop.assert_called_once()
    assert (tmp_path / "run-summary.json").is_file()


@pytest.mark.parametrize("outcome", ["empty", "all_error", "mixed"])
def test_managed_success_requires_valid_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str):
    monkeypatch.syspath_prepend(str(TUTORIAL.parents[2] / "benchmarking/scripts"))
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("run_dynamo_uncapped")
    benchmark = importlib.import_module("nemotron_parse_pdf_benchmark")
    sampler = SimpleNamespace(start=Mock(), stop=Mock())
    monkeypatch.setattr(module, "GPUSampler", lambda *_args: sampler)
    monkeypatch.setattr(module.time, "perf_counter", Mock(side_effect=[0.0, 1.0, 2.0, 5.0]))
    server = SimpleNamespace(_started=True, _backend_impl=None, start=Mock(), stop=Mock())
    valid_pages = 2 if outcome == "mixed" else 0
    outputs = []
    if outcome != "empty":
        outputs.append(
            FileGroupTask(
                dataset_name="pdf",
                data=[str(tmp_path / "output.parquet")],
                _metadata={"source_files": ["sample.pdf"]},
                _stage_perf=[
                    StagePerfStats(
                        stage_name="nemotron_parse_inference",
                        custom_metrics={"num_valid_pages": valid_pages, "num_error_pages": 3 - valid_pages},
                    )
                ],
            )
        )
    pipeline = SimpleNamespace(run=Mock(return_value=outputs))

    def execute() -> None:
        module.execute_run(
            server,
            lambda: pipeline,
            None,
            run_dir=tmp_path,
            topology={},
            compute_metrics=benchmark._compute_pdf_parse_metrics,
            count_pdfs=benchmark._count_unique_pdfs,
        )

    if valid_pages:
        execute()
    else:
        with pytest.raises(RuntimeError, match="no valid pages"):
            execute()
    summary = json.loads((tmp_path / "run-summary.json").read_text())
    assert summary["exit_code"] == (0 if valid_pages else 1)
    assert summary["num_pages_processed"] == valid_pages
    assert summary["num_pdfs_processed"] == (0 if outcome == "empty" else 1)
    assert summary["pipeline_elapsed_seconds"] == 3.0
    assert summary["inference_server_startup_seconds_excluded"] == 1.0
    if valid_pages:
        assert summary["error"] is None
    else:
        assert "no valid pages" in summary["error"]
    sampler.stop.assert_called_once()
    server.stop.assert_called_once()


def test_managed_runner_builds_parse35_with_validated_fractional_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    module = importlib.import_module("run_dynamo_uncapped")
    ray_data = importlib.import_module("nemo_curator.backends.ray_data")
    monkeypatch.setattr(ray_data.RayDataExecutor, "__init__", lambda _self: None)
    execute = Mock()
    monkeypatch.setattr(module, "execute_run", execute)
    monkeypatch.setattr(module, "ensure_cuda13_python", Mock())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dynamo_uncapped.py",
            "--eager-mode",
            "off",
            "--curator-root",
            str(TUTORIAL.parents[2]),
            "--manifest",
            str(manifest),
            "--pdf-dir",
            str(tmp_path),
            "--model-path",
            str(model),
            "--output-root",
            str(tmp_path / "results"),
        ],
    )
    module.main()
    server = execute.call_args.args[0]
    assert server.models[0].gpu_fraction == 0.5
    assert server.models[0].num_replicas == 16
    assert str(model) in server.models[0].runtime_env["uv"]["packages"]
    assert server.backend.router.router_kwargs["dyn_chat_processor"] == "dynamo"
    assert Path(server.models[0].dynamo_kwargs["custom_jinja_template"]).is_file()
    assert "chat_template" not in server.backend.router.router_kwargs
