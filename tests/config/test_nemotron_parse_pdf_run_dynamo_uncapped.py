# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Check interpreter identity and managed benchmark cleanup without serving."""

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
