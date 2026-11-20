# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
import exact_removal_sample_benchmark as benchmark


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Fixture preparation and output verification do not require a GPU."""


def test_paired_fixture_has_known_exact_duplicates_and_utf8_bound(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in ["é", "é", "keep", "too large"]))
    corpus = tmp_path / "corpus"
    fixture, expected = benchmark.prepare_fixture(source, corpus, max_text_bytes=6)
    texts = [json.loads(line)["text"] for p in corpus.glob("*.jsonl") for line in p.read_text().splitlines()]
    assert texts == ["é", "é", "keep", "keep"]
    assert fixture["num_input_documents"] == 4
    assert fixture["expected_removed_documents"] == 2
    assert fixture["input_text_bytes"] == 12
    assert fixture["expected_output_text_bytes"] == 6
    assert expected == {hashlib.sha256(t.encode()).hexdigest() for t in ["é", "keep"]}
    for f in fixture["corpus_files"]:
        assert f["sha256"] == hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        benchmark.prepare_fixture(source, corpus, max_text_bytes=6)


@pytest.mark.parametrize("texts", [["é", "keep"], ["é", "é"], ["é"], ["é", "keep", "keep"]])
def test_output_verifies_exactly_one_of_each_text(tmp_path: Path, texts: list[str]) -> None:
    pq.write_table(pa.table({"text": texts}), tmp_path / "part.parquet")
    expected = {hashlib.sha256(t.encode()).hexdigest() for t in ["é", "keep"]}
    if texts == ["é", "keep"]:
        benchmark.validate_removed_output(tmp_path, expected)
    else:
        with pytest.raises(ValueError, match="exactly one"):
            benchmark.validate_removed_output(tmp_path, expected)


def test_failed_identification_never_starts_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.jsonl").write_text('{"text":"keep"}\n')
    monkeypatch.setattr(
        benchmark,
        "run_exact_duplicate_identification_benchmark",
        lambda *_, **__: {"metrics": {"is_success": False}, "tasks": []},
    )

    def unexpected_removal(*_, **__) -> None:
        pytest.fail("Removal started without validated exact IDs/registry")

    monkeypatch.setattr(benchmark, "run_removal_benchmark", unexpected_removal)
    with pytest.raises(ValueError, match="Exact identification"):
        benchmark.run_sample(source, tmp_path / "work", 100)
