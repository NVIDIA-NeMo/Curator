# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
import fasttext_filter_benchmark as benchmark

from tests.benchmarking.scripts.test_modifier_benchmark import SerialExecutor
from tests.benchmarking.scripts.test_score_filter_benchmark import filter_config


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """This test checks accounting around a real small filter, not FastText models."""


def test_combined_pipeline_records_recipe_and_complete_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"text": "é"}\n')
    cfg = tmp_path / "recipe.yaml"
    OmegaConf.save(OmegaConf.create(filter_config(source, tmp_path / "output")), cfg)
    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    result = benchmark.run_fasttext_filter_benchmark(
        source, tmp_path / "output", "xenna", tmp_path / "results", cfg, tmp_path / "language", tmp_path / "quality"
    )
    assert result["metrics"]["is_success"]
    assert result["metrics"]["num_input_documents"] == 1
    assert result["metrics"]["input_text_bytes"] == 2
    assert result["metrics"]["num_output_documents"] == result["metrics"]["output_text_bytes"] == 0
    assert result["params"]["measurement_semantics"] == "combined language and quality filtering"
    assert result["params"]["resolved_pipeline"]["fasttext_langid_model_path"] == str(tmp_path / "language")
