# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
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
    """These fixtures execute stages locally without a Ray scheduler."""


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


@pytest.mark.parametrize("filter_mode", ["language", "quality"])
def test_standalone_mode_runs_real_model_without_loading_other_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filter_mode: str
) -> None:
    import fasttext

    source = tmp_path / "input.jsonl"
    texts = ["hello café", "hello world"]
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    training = tmp_path / "training.txt"
    label = "en" if filter_mode == "language" else "hq"
    training.write_text(f"__label__{label} hello world\n__label__{label} hello café\n")
    model_path = tmp_path / "model.bin"
    model = fasttext.train_supervised(str(training), dim=4, epoch=2, thread=1, minCount=1, bucket=100, verbose=0)
    model.save_model(str(model_path))
    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    cfg = Path(__file__).resolve().parents[3] / "nemo_curator/config/text/fasttext_filter_pipeline.yaml"
    missing_model = tmp_path / "unused-does-not-exist.bin"
    result = benchmark.run_fasttext_filter_benchmark(
        source,
        tmp_path / "output",
        "xenna",
        tmp_path / "results",
        cfg,
        model_path if filter_mode == "language" else missing_model,
        model_path if filter_mode == "quality" else missing_model,
        filter_mode=filter_mode,
    )
    assert result["metrics"]["is_success"]
    assert result["metrics"]["num_input_documents"] == result["metrics"]["num_output_documents"] == 2
    assert (
        result["metrics"]["input_text_bytes"]
        == result["metrics"]["output_text_bytes"]
        == sum(len(text.encode("utf-8")) for text in texts)
    )
    stages = result["params"]["resolved_pipeline"]["stages"]
    assert len(stages) == 3
    settings = stages[1]["filter_obj"]
    if filter_mode == "language":
        assert settings["min_langid_score"] == 0.3
    else:
        assert settings["label"] == "__label__hq"
        assert settings["alpha"] == 3
        assert settings["seed"] == 42
    assert result["params"]["filter_mode"] == filter_mode
    assert "standalone" in result["params"]["measurement_semantics"]


def test_standalone_mode_rejects_unrecognized_recipe(tmp_path: Path) -> None:
    cfg = OmegaConf.create(filter_config(tmp_path / "input", tmp_path / "output"))
    with pytest.raises(ValueError, match="unrecognized stage"):
        benchmark.select_filter_mode(cfg, "language")
