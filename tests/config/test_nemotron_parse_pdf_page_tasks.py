# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise page-task entry points through the real pipeline/writer builder."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo_curator.pipeline import Pipeline
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG, write_manifest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("entry_point", ["main", "benchmark"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_page_task_entry_points_require_fresh_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry_point: str, existing_output: bool
):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarking/scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "tutorials/interleaved/nemotron_parse_pdf"))
    if entry_point == "main":
        spec = importlib.util.spec_from_file_location(
            "page_task_main", ROOT / "tutorials/interleaved/nemotron_parse_pdf/main.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "RayDataExecutor", Mock())
    else:
        module = importlib.import_module("nemotron_parse_pdf_benchmark")
        monkeypatch.setattr(module, "setup_executor", Mock())
        monkeypatch.setattr(module, "write_benchmark_results", Mock())
        monkeypatch.setattr(module, "_available_gpu_count", Mock(side_effect=AssertionError("no GPU probe")))
    run = Mock(return_value=[])
    monkeypatch.setattr(Pipeline, "run", run)
    output = tmp_path / "output"
    if existing_output:
        output.mkdir()
    manifest = write_manifest(tmp_path, [{"file_name": "sample.pdf", "num_pages": 2}])
    config = tmp_path / "extraction.json"
    config.write_text(json.dumps(TEST_EXTRACTION_CONFIG))
    argv = [
        entry_point,
        "--manifest",
        str(manifest),
        "--pdf-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--parse-mode",
        "single",
        "--inference-server-endpoint",
        "http://served/v1",
        "--pages-per-task",
        "64",
        "--page-budget-extraction-config",
        str(config),
    ]
    if entry_point == "benchmark":
        argv.extend(
            [
                "--benchmark-results-path",
                str(tmp_path / "results"),
                "--num-replicas",
                "1",
                "--inference-server-num-gpus",
                "1",
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    if entry_point == "main" and existing_output:
        with pytest.raises(FileExistsError, match="already exists"):
            module.main()
    else:
        module.main()
    assert run.call_count == int(not existing_output)
    assert output.is_dir()
