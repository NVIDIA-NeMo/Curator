# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate benchmark serving selection without allocating GPUs or starting Ray."""

import importlib
import shlex
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).parents[2]


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    monkeypatch.syspath_prepend(str(ROOT / "benchmarking/scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "tutorials/interleaved/nemotron_parse_pdf"))
    module = importlib.import_module("nemotron_parse_pdf_benchmark")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        ["--manifest", "input.jsonl", "--pdf-dir", ".", "--output-dir", str(tmp_path)]
    )
    vars(args).update(
        executor="ray_data",
        backend="vllm",
        inference_server_type=None,
        num_replicas=1,
        inference_server_client_workers_per_replica=4,
        model_id=None,
        engine_kwargs=None,
        inference_server_request_timeout_s=300,
        inference_server_health_timeout_s=900,
        benchmark_results_path=tmp_path / "results",
    )
    pipeline = Mock()
    pipeline.run.return_value = []
    build = Mock(return_value=pipeline)
    server = Mock()
    server.models = [Mock(engine_kwargs={})]
    server.endpoint = "http://managed/v1"
    factory = Mock(return_value=server)
    probe = Mock(return_value=1)
    monkeypatch.setattr(module, "setup_executor", Mock())
    monkeypatch.setattr(module, "create_nemotron_parse_pdf_pipeline", build)
    monkeypatch.setattr(module, "create_nemotron_parse_inference_server", factory)
    monkeypatch.setattr(module, "_available_gpu_count", probe)
    return module, args, pipeline, build, server, factory, probe


def test_default_benchmark_routes_parse35_to_managed_dynamo(benchmark: tuple, tmp_path: Path):
    module, args, pipeline, build, server, factory, _ = benchmark
    args.nemotron_parse_path = str(tmp_path / "parse35")
    result = module.run_nemotron_parse_pdf_benchmark(args)
    assert factory.call_args.kwargs["backend"] == "dynamo"
    assert factory.call_args.kwargs["nemotron_parse_path"] == args.nemotron_parse_path
    assert factory.call_args.kwargs["model_path"] == "nvidia/Nemotron-3.5-Parse"
    assert build.call_args.args[0].parse_mode == "single"
    assert build.call_args.kwargs["inference_server_endpoint"] == server.endpoint
    assert result["params"]["parse_mode"] == "single"
    pipeline.run.assert_called_once()
    server.start.assert_called_once()
    server.stop.assert_called_once()


def test_parse35_benchmark_requires_serving_configuration_before_gpu_probe(benchmark: tuple):
    module, args, _, build, _, factory, probe = benchmark
    result = module.run_nemotron_parse_pdf_benchmark(args)
    assert not result["metrics"]["is_success"]
    build.assert_not_called()
    factory.assert_not_called()
    probe.assert_not_called()


def test_external_endpoint_does_not_launch_or_stop_a_server(benchmark: tuple):
    module, args, pipeline, build, server, factory, probe = benchmark
    args.inference_server_endpoint = "http://native/v1"
    args.inference_server_model_name = "served-name"
    result = module.run_nemotron_parse_pdf_benchmark(args)
    assert build.call_args.kwargs["inference_server_endpoint"] == "http://native/v1"
    assert build.call_args.kwargs["inference_server_model_name"] == "served-name"
    assert build.call_args.kwargs["inference_server_client_num_workers"] == 4
    assert result["params"]["inference_server_endpoint"] == "http://native/v1"
    pipeline.run.assert_called_once()
    factory.assert_not_called()
    probe.assert_not_called()
    server.stop.assert_not_called()


def test_explicit_legacy_benchmark_retains_inprocess_path(benchmark: tuple):
    module, args, pipeline, build, _, factory, probe = benchmark
    args.parse_mode = None
    args.model_path = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    module.run_nemotron_parse_pdf_benchmark(args)
    assert build.call_args.kwargs == {"inprocess_backend": "vllm"}
    pipeline.run.assert_called_once()
    factory.assert_not_called()
    probe.assert_called_once()


def test_pinned_legacy_regressions_explicitly_select_their_original_mode(benchmark: tuple):
    module, *_ = benchmark
    config = yaml.safe_load((ROOT / "benchmarking/benchmarks.yaml").read_text())
    entries = [e for e in config["entries"] if e["script"] == "nemotron_parse_pdf_benchmark.py"]
    assert len(entries) == 4
    for entry in entries:
        args, _ = module.create_nemotron_parse_pdf_argparser().parse_known_args(shlex.split(entry["args"]))
        assert args.parse_mode is None
        assert args.model_path == "{dataset:nemotron_parse_pdf_model,files}"
