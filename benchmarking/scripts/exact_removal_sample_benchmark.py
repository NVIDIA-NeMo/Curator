# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain a bounded paired-text fixture, identify exact IDs, then measure removal."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from dedup_removal_benchmark import run_removal_benchmark
from exact_dedup_identification_benchmark import run_exact_duplicate_identification_benchmark
from utils import write_benchmark_results


def prepare_fixture(source: Path, destination: Path, max_text_bytes: int) -> tuple[dict, set[str]]:
    if not 0 < max_text_bytes <= 256 * 1024**2:
        msg = "Fixture base-text cap must be positive and at most 256 MiB"
        raise ValueError(msg)
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    sources = []
    total_bytes = 0
    full = False
    for path in sorted(source.glob("*.jsonl")):
        selected = 0
        with path.open() as stream:
            for line in stream:
                text = json.loads(line)["text"]
                if not isinstance(text, str):
                    msg = "Fixture source text must be a string"
                    raise TypeError(msg)
                if text in seen:
                    continue
                encoded = text.encode("utf-8")
                if total_bytes + len(encoded) > max_text_bytes:
                    full = True
                    break
                part = destination / f"part.{len(seen) // 4096:05d}.jsonl"
                with part.open("ab") as output:
                    output.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8") * 2)
                seen.add(text)
                selected += 1
                total_bytes += len(encoded)
        if selected:
            sources.append({"path": str(path), "selected_unique_documents": selected})
        if full:
            break
    if not seen or total_bytes == 0:
        msg = "Fixture must contain nonzero useful text"
        raise ValueError(msg)
    files = [
        {"path": str(p), "size_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(destination.glob("*.jsonl"))
    ]
    hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in seen}
    return {
        "fixture_semantics": "Convenience prefix of distinct prepared-CC texts, each written twice; controlled 50% exact duplicates, not natural CC prevalence. Source metadata is not copied.",
        "source_files": sources,
        "corpus_files": files,
        "max_source_text_bytes": max_text_bytes,
        "num_input_documents": 2 * len(seen),
        "expected_removed_documents": len(seen),
        "expected_output_documents": len(seen),
        "input_text_bytes": 2 * total_bytes,
        "expected_output_text_bytes": total_bytes,
    }, hashes


def validate_removed_output(path: Path, expected: set[str]) -> None:
    actual = set()
    rows = 0
    for file in path.glob("*.parquet"):
        for batch in pq.ParquetFile(file).iter_batches(columns=["text"]):
            for text in batch.column(0).to_pylist():
                rows += 1
                actual.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    if rows != len(expected) or actual != expected:
        msg = "Removal output must contain exactly one copy of every fixture text"
        raise ValueError(msg)


def run_sample(source: Path, work: Path, max_text_bytes: int) -> dict:
    corpus, identification, output = work / "corpus", work / "identification", work / "removed"
    start = time.perf_counter()
    fixture, expected = prepare_fixture(source, corpus, max_text_bytes)
    fixture_time = time.perf_counter() - start
    manifest = work / "fixture-manifest.json"
    manifest.write_text(json.dumps(fixture, indent=2))
    identified = run_exact_duplicate_identification_benchmark(
        str(corpus),
        str(identification),
        input_filetype="jsonl",
        input_blocksize="1.5GiB",
    )
    write_benchmark_results(identified, work / "identification-metrics", save_tasks=False)
    im = identified["metrics"]
    if (
        not im["is_success"]
        or im["num_input_documents"] != fixture["num_input_documents"]
        or im["num_duplicates"] != len(expected)
    ):
        msg = "Exact identification must match the fixture's known input and duplicate counts"
        raise ValueError(msg)
    registry = identified["params"]["input_denominator_source"]
    ids = identification / "ExactDuplicateIds"
    fixture.update(
        exact_ids_path=str(ids),
        id_generator_path=registry,
        identification_time_s=im["time_taken_s"],
        fixture_preparation_time_s=fixture_time,
    )
    manifest.write_text(json.dumps(fixture, indent=2))
    for file in fixture["corpus_files"]:
        if hashlib.sha256(Path(file["path"]).read_bytes()).hexdigest() != file["sha256"]:
            msg = "Fixture corpus changed after exact identification"
            raise ValueError(msg)
    result = run_removal_benchmark(
        str(corpus),
        str(ids),
        str(output),
        "xenna",
        input_filetype="jsonl",
        output_filetype="parquet",
        duplicate_id_field="_curator_dedup_id",
        id_generator_path=registry,
        blocksize="1.5GiB",
    )
    metrics = result["metrics"]
    if any(metrics[k] != fixture[k] for k in ["num_input_documents", "input_text_bytes"]) or any(
        metrics[k] != fixture[expected_key]
        for k, expected_key in [
            ("num_output_documents", "expected_output_documents"),
            ("num_removed_documents", "expected_removed_documents"),
            ("num_duplicates_removed", "expected_removed_documents"),
            ("output_text_bytes", "expected_output_text_bytes"),
        ]
    ):
        msg = "Exact removal count/byte invariants do not match the identified fixture"
        raise ValueError(msg)
    validate_removed_output(output, expected)
    result["params"].update(
        fixture_semantics=fixture["fixture_semantics"],
        fixture_manifest=str(manifest),
        fixture=fixture,
        measurement_semantics="Exact-ID removal only; fixture generation, identification and output validation excluded from primary elapsed time. Artifacts retained; preparation and removal run sequentially in this script, not by YAML ordering.",
    )
    metrics.update(
        fixture_preparation_time_s=fixture_time,
        fixture_identification_time_s=im["time_taken_s"],
        output_correctness_verified=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--work-path", type=Path, required=True)
    parser.add_argument("--max-source-text-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--save-tasks", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = {
        "params": {"source_path": str(args.source_path), "work_path": str(args.work_path)},
        "metrics": {"is_success": False, "time_taken_s": None},
        "tasks": [],
    }
    try:
        result = run_sample(args.source_path, args.work_path, args.max_source_text_bytes)
    finally:
        write_benchmark_results(result, args.benchmark_results_path, save_tasks=args.save_tasks)
    return 0 if result["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
