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
import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


def load_module(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    spec = importlib.util.spec_from_file_location(name, TUTORIAL / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_external_endpoint_does_not_start_managed_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    main = load_module(monkeypatch, "main")
    manifest = tmp_path / "input.jsonl"
    manifest.touch()
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
            str(manifest),
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


@pytest.mark.parametrize("manifest_is_directory", [False, True])
def test_invalid_manifest_fails_before_managed_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, manifest_is_directory: bool
):
    main = load_module(monkeypatch, "main")
    manifest = tmp_path / "input.jsonl"
    if manifest_is_directory:
        manifest.mkdir()
    probe = Mock(return_value=(4, 1))
    factory = MagicMock()
    monkeypatch.setattr(main, "get_available_cpu_gpu_resources", probe)
    monkeypatch.setattr(main, "create_nemotron_parse_inference_server", factory)
    monkeypatch.setattr(main, "_run_pipeline", Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--manifest",
            str(manifest),
            "--pdf-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--nemotron-parse-path",
            str(tmp_path / "model"),
        ],
    )
    with pytest.raises(OSError, match=r"input\.jsonl"):
        main.main()
    probe.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("parse_mode", [None, "single", "double", "triple"])
def test_zero_valid_page_check_applies_only_to_served_parse35(monkeypatch: pytest.MonkeyPatch, parse_mode: str | None):
    main = load_module(monkeypatch, "main")
    pipeline = SimpleNamespace(run=Mock(return_value=[]))
    monkeypatch.setattr(main, "create_nemotron_parse_pdf_pipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(main, "RayDataExecutor", Mock())
    args = SimpleNamespace(executor="ray-data", parse_mode=parse_mode)
    if parse_mode is None:
        main._run_pipeline(args, "http://served/v1", "model", 4)
    else:
        with pytest.raises(RuntimeError, match="no valid pages"):
            main._run_pipeline(args, "http://served/v1", "model", 4)


def test_tutorial_defaults_to_served_parse35_and_warn_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "pipeline_utils")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        ["--manifest", "input.jsonl", "--pdf-dir", ".", "--output-dir", str(tmp_path)]
    )
    assert args.parse_mode == "single"
    assert args.model_path == "nvidia/Nemotron-3.5-Parse"
    pipeline = module.create_nemotron_parse_pdf_pipeline(args, inference_server_endpoint="http://served/v1")
    assert pipeline.stages[-1].on_materialize_error == "warn"
    assert pipeline.stages[0]._inference.parse_mode == "single"


def test_tutorial_explicit_legacy_path_remains_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "pipeline_utils")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        [
            "--manifest",
            "input.jsonl",
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--legacy-parse",
            "--model-path",
            "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        ]
    )
    pipeline = module.create_nemotron_parse_pdf_pipeline(args, inference_server_endpoint="http://served/v1")
    assert args.parse_mode is None
    assert pipeline.stages[0]._inference.parse_mode is None
    assert pipeline.stages[0].model_path == "nvidia/NVIDIA-Nemotron-Parse-v1.2"


def test_tutorial_rejects_conflicting_parse_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "pipeline_utils")
    with pytest.raises(SystemExit, match="2"):
        module.create_nemotron_parse_pdf_argparser().parse_args(
            [
                "--manifest",
                "input.jsonl",
                "--pdf-dir",
                ".",
                "--output-dir",
                str(tmp_path),
                "--legacy-parse",
                "--parse-mode",
                "single",
            ]
        )


def test_tutorial_requires_endpoint_or_path_for_selected_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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


def test_managed_parse_mode_starts_model_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    main = load_module(monkeypatch, "main")
    manifest = tmp_path / "input.jsonl"
    manifest.touch()
    model_path = str(tmp_path / "parse35")
    server = MagicMock()
    server.endpoint = "http://managed/v1"
    factory = Mock(return_value=server)
    run = Mock()
    monkeypatch.setattr(main, "create_nemotron_parse_inference_server", factory)
    monkeypatch.setattr(main, "get_available_cpu_gpu_resources", Mock(return_value=(14, 2)))
    monkeypatch.setattr(main, "_run_pipeline", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--manifest",
            str(manifest),
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--parse-mode",
            "single",
            "--model-path",
            model_path,
            "--nemotron-parse-path",
            model_path,
            "--max-num-seqs",
            "128",
        ],
    )
    main.main()

    factory.assert_called_once_with(
        model_path=model_path,
        model_name=model_path,
        backend="dynamo",
        num_replicas=2,
        engine_kwargs={"max_num_seqs": 128},
        nemotron_parse_path=model_path,
    )
    _, endpoint, model, workers = run.call_args.args
    assert (endpoint, model, workers) == ("http://managed/v1", model_path, 8)


@pytest.mark.parametrize(
    ("path", "section_marker"),
    [
        ("tutorials/interleaved/nemotron_parse_pdf/README.md", "## Served Nemotron Parse 3.5"),
        ("fern/versions/main/pages/curate-text/load-data/nemotron-parse-pdf.mdx", "## Served Nemotron Parse 3.5"),
        ("nemo_curator/stages/interleaved/pdf/nemotron_parse/inference.py", "_PARSE_INSTALL_HINT = ("),
    ],
)
def test_served_install_declares_pdf_renderer(path: str, section_marker: str):
    root = Path(__file__).parents[2]
    section = (root / path).read_text().split(section_marker, 1)[1]
    command = re.search(r"nemo-curator\[([^]]+)\]", section)
    assert command is not None
    optional = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    dependencies = [requirement for extra in command[1].split(",") for requirement in optional[extra]]
    assert any(re.match(r"^pypdfium2(?:$|[<>=!~;])", requirement) for requirement in dependencies)


def test_tutorial_encoded_image_reuse_is_independent_of_transport_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = load_module(monkeypatch, "pipeline_utils")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        [
            "--manifest",
            "input.jsonl",
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--parse-mode",
            "single",
            "--reuse-encoded-images",
        ]
    )
    pipeline = module.create_nemotron_parse_pdf_pipeline(args, inference_server_endpoint="http://served/v1")
    reader = pipeline.stages[0]
    assert reader._preprocessor.reuse_encoded_images
    assert reader._inference.reuse_encoded_images


def test_tutorial_accepts_namespace_without_encoded_image_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "pipeline_utils")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        ["--manifest", "input.jsonl", "--pdf-dir", ".", "--output-dir", str(tmp_path)]
    )
    del args.reuse_encoded_images
    pipeline = module.create_nemotron_parse_pdf_pipeline(args, inference_server_endpoint="http://served/v1")
    reader = pipeline.stages[0]
    assert not reader._preprocessor.reuse_encoded_images
    assert not reader._inference.reuse_encoded_images


@pytest.mark.parametrize("parse_mode", [None, "double", "triple"])
def test_tutorial_pipeline_rejects_encoded_image_reuse_outside_single_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parse_mode: str | None
):
    module = load_module(monkeypatch, "pipeline_utils")
    options = ["--parse-mode", parse_mode] if parse_mode is not None else ["--legacy-parse"]
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        [
            "--manifest",
            "input.jsonl",
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--reuse-encoded-images",
            *options,
        ]
    )
    with pytest.raises(ValueError, match=r"Encoded image reuse requires served Parse 3\.5 single mode"):
        module.create_nemotron_parse_pdf_pipeline(args, inference_server_endpoint="http://served/v1")


def test_tutorial_pipeline_rejects_encoded_image_reuse_without_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = load_module(monkeypatch, "pipeline_utils")
    args = module.create_nemotron_parse_pdf_argparser().parse_args(
        [
            "--manifest",
            "input.jsonl",
            "--pdf-dir",
            ".",
            "--output-dir",
            str(tmp_path),
            "--parse-mode",
            "single",
            "--reuse-encoded-images",
        ]
    )
    with pytest.raises(ValueError, match=r"Encoded image reuse requires served Parse 3\.5 single mode"):
        module.create_nemotron_parse_pdf_pipeline(args)


def test_legacy_inprocess_entry_point_keeps_its_model_and_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "inprocess")
    client = Mock()
    pipeline = Mock()
    pipeline.run.return_value = []
    build = Mock(return_value=pipeline)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(module, "RayClient", Mock(return_value=client))
    monkeypatch.setattr(module, "XennaExecutor", Mock())
    monkeypatch.setattr(module, "create_nemotron_parse_pdf_pipeline", build)
    monkeypatch.setattr(
        sys, "argv", ["inprocess.py", "--manifest", "input.jsonl", "--pdf-dir", ".", "--output-dir", str(tmp_path)]
    )
    module.main()
    args = build.call_args.args[0]
    assert args.parse_mode is None
    assert args.model_path == "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    client.start.assert_called_once()
    client.stop.assert_called_once()


def test_inprocess_rejects_served_mode_before_ray_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = load_module(monkeypatch, "inprocess")
    client = Mock()
    monkeypatch.setattr(module, "RayClient", client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inprocess.py",
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
        module.main()
    client.assert_not_called()
