# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
import score_filter_benchmark as benchmark

from tests.benchmarking.scripts.test_modifier_benchmark import SerialExecutor


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Real readers, filters and writers execute serially in these tests."""


def filter_config(input_path: Path, output_path: Path) -> dict:
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "text_field": "text",
        "fasttext_langid_model_path": "unused-by-this-instrumentation-fixture",
        "fasttext_quality_model_path": "unused-by-this-instrumentation-fixture",
        "stages": [
            {"_target_": "nemo_curator.stages.text.io.reader.JsonlReader", "file_paths": "${input_path}"},
            {
                "_target_": "nemo_curator.stages.text.filters.score_filter.ScoreFilter",
                "filter_obj": {
                    "_target_": "nemo_curator.stages.text.filters.heuristic.string.WordCountFilter",
                    "min_words": 2,
                },
                "text_field": "${text_field}",
            },
            {"_target_": "nemo_curator.stages.text.io.writer.JsonlWriter", "path": "${output_path}"},
        ],
    }


@pytest.mark.parametrize("all_rejected", [False, True])
def test_input_denominator_includes_fully_rejected_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_rejected: bool
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    batches = [["é", "x"], ["hello" if all_rejected else "hello world"]]
    for index, texts in enumerate(batches):
        (source / f"{index}.jsonl").write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    cfg = tmp_path / "recipe.yaml"
    OmegaConf.save(OmegaConf.create(filter_config(source, tmp_path / "output")), cfg)
    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    result = benchmark.run_score_filter_benchmark(source, tmp_path / "output", "xenna", tmp_path / "results", cfg)
    metrics = result["metrics"]
    assert metrics["is_success"]
    assert metrics["num_input_documents"] == metrics["num_documents_processed"] == 3
    assert metrics["num_output_documents"] == metrics["num_kept_documents"] == (0 if all_rejected else 1)
    assert metrics["input_text_bytes"] == sum(len(text.encode("utf-8")) for batch in batches for text in batch)
    assert metrics["output_text_bytes"] == (0 if all_rejected else 11)
    assert metrics["num_rejected_documents"] == (3 if all_rejected else 2)
    assert result["params"]["resolved_pipeline"]["stages"][1]["filter_obj"]["min_words"] == 2
