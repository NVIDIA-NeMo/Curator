# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Exercise external endpoint tutorial wiring without starting serving or Ray."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


def load_module(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    spec = importlib.util.spec_from_file_location(name, TUTORIAL / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_endpoint_does_not_start_managed_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    main = load_module(monkeypatch, "main")
    run = Mock()
    monkeypatch.setattr(main, "_run_pipeline", run)
    monkeypatch.setattr(main, "get_available_cpu_gpu_resources", Mock(side_effect=AssertionError("no Ray probe")))
    monkeypatch.setattr(main, "create_nemotron_parse_inference_server", Mock(side_effect=AssertionError("no server")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--manifest",
            "input.jsonl",
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--parse-mode",
            "triple",
            "--triple-regions",
            "detector",
            "--inference-server-endpoint",
            "http://served/v1",
            "--inference-server-model-name",
            "served-name",
            "--model-path",
            "nvidia/Nemotron-3.5-Parse",
            "--client-num-workers",
            "8",
            "--executor",
            "xenna",
        ],
    )
    main.main()
    args, endpoint, model, workers = run.call_args.args
    assert (endpoint, model, workers) == ("http://served/v1", "served-name", 8)
    assert args.parse_mode == "triple"
    assert args.executor == "xenna"


def test_tutorial_defaults_preserve_legacy_and_warn_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "pipeline_utils")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        ["--manifest", "input.jsonl", "--pdf-dir", ".", "--output-dir", str(tmp_path)]
    )
    assert args.parse_mode is None
    assert args.model_path == "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    pipeline = module.create_nemotron_parse_pdf_pipeline(args, inference_server_endpoint="http://served/v1")
    assert pipeline.stages[-1].on_materialize_error == "warn"


def test_tutorial_requires_endpoint_for_selected_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    main = load_module(monkeypatch, "main")
    monkeypatch.setattr(main, "get_available_cpu_gpu_resources", Mock(side_effect=AssertionError("no Ray probe")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--manifest",
            "input.jsonl",
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--parse-mode",
            "single",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main.main()
