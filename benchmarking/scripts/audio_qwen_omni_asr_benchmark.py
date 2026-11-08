# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Benchmark the Qwen Omni ASR tutorial on a staged manifest."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
from utils import write_benchmark_results

from nemo_curator.config.run import create_executor_from_yaml, create_pipeline_from_yaml

_PIPELINE_CONFIG = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/qwen_omni_asr/pipeline.yaml"


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audio_path = row["audio_filepath"]
                transcript = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                msg = f"Invalid manifest row {path}:{line_number}"
                raise ValueError(msg) from error
            if not isinstance(audio_path, str) or not audio_path.strip():
                msg = f"Invalid audio_filepath in {path}:{line_number}"
                raise ValueError(msg)
            if not isinstance(transcript, str) or not transcript.strip():
                msg = f"Empty or invalid input transcript in {path}:{line_number}"
                raise ValueError(msg)
            if audio_path in rows:
                msg = f"Duplicate audio_filepath in {path}: {audio_path}"
                raise ValueError(msg)
            rows[audio_path] = row
    if not rows:
        msg = f"Manifest contains no rows: {path}"
        raise ValueError(msg)
    return rows


def _collect_output_metrics(output_path: Path, input_rows: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    output_rows = _read_manifest(output_path)
    expected_paths, actual_paths = set(input_rows), set(output_rows)
    if expected_paths != actual_paths:
        msg = (
            "Output audio paths do not match input: "
            f"missing={len(expected_paths - actual_paths)}, extra={len(actual_paths - expected_paths)}"
        )
        raise ValueError(msg)

    total_duration_s = 0.0
    input_text_fallbacks = 0
    skipped = 0
    for audio_path, row in output_rows.items():
        if row["text"] != input_rows[audio_path]["text"]:
            msg = f"Input transcript changed for {audio_path}"
            raise ValueError(msg)
        for field in ("primary_model_prediction", "best_prediction", "abbreviated_text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                msg = f"Empty or invalid {field} for {audio_path}"
                raise ValueError(msg)
        duration = row.get("duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            msg = f"Invalid duration for {audio_path}: {duration}"
            raise ValueError(msg)
        source = row.get("best_prediction_source")
        if source not in ("primary", "reference", "ground_truth"):
            msg = f"Unexpected prediction source for {audio_path}: {source}"
            raise ValueError(msg)
        if source in ("reference", "ground_truth") and row["best_prediction"] != row["text"].strip():
            msg = f"Fallback does not match input transcript for {audio_path}"
            raise ValueError(msg)
        skip_reason = row.get("_skipme", "")
        if not isinstance(skip_reason, str):
            msg = f"Invalid skip reason for {audio_path}: {skip_reason}"
            raise ValueError(msg)  # noqa: TRY004
        input_text_fallbacks += source in ("reference", "ground_truth")
        skipped += bool(skip_reason)
        total_duration_s += duration

    return {
        "num_tasks_processed": len(output_rows),
        "output_coverage_ratio": len(output_rows) / len(input_rows),
        "input_text_preservation_ratio": 1.0,
        "num_input_text_fallbacks": input_text_fallbacks,
        "num_skipped_tasks": skipped,
        "total_audio_duration_hours": total_duration_s / 3600,
    }


def _check_fresh_outputs(benchmark_results_path: Path) -> None:
    # The runner precreates an empty scratch directory and params.json;
    # write_benchmark_results merges its metadata with our parameters.
    for name in ("results", "scratch", "metrics.json", "tasks.pkl"):
        path = benchmark_results_path / name
        if name == "scratch" and path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            continue
        if path.exists() or path.is_symlink():
            msg = f"Benchmark output already exists: {path}; choose a fresh results directory"
            raise FileExistsError(msg)


def run_audio_qwen_omni_asr_benchmark(
    benchmark_results_path: str,
    input_manifest: str,
    model_id: str | None = None,
    executor: str | None = None,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Run the shared tutorial graph and validate every output against its input."""
    results_path = Path(benchmark_results_path).resolve()
    _check_fresh_outputs(results_path)
    input_rows = _read_manifest(Path(input_manifest))
    cfg = OmegaConf.load(_PIPELINE_CONFIG)
    cfg.manifest_path = str(Path(input_manifest).resolve())
    cfg.output_path = str(results_path / "results" / "qwen_omni_asr.jsonl")
    cfg.workspace_dir = str(results_path / "scratch")
    if executor is not None:
        cfg.backend = executor
    if execution_mode is not None:
        cfg.execution_mode = execution_mode
    if model_id is not None:
        if not Path(model_id).is_dir():
            msg = f"Pre-staged model directory does not exist: {model_id}"
            raise FileNotFoundError(msg)
        cfg.model_id = str(Path(model_id).resolve())
    pipeline = create_pipeline_from_yaml(cfg)
    executor_obj = create_executor_from_yaml(cfg)

    run_start = time.perf_counter()
    pipeline.run(executor=executor_obj)
    run_time_s = time.perf_counter() - run_start
    output_metrics = _collect_output_metrics(Path(cfg.output_path), input_rows)
    return {
        "params": {
            "executor": cfg.backend,
            "execution_mode": cfg.execution_mode if cfg.backend == "xenna" else None,
            "input_manifest": cfg.manifest_path,
            "benchmark_results_path": str(results_path),
            "pipeline_config_path": str(_PIPELINE_CONFIG),
            "pipeline_config": OmegaConf.to_container(cfg, resolve=True),
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_s,
            **output_metrics,
            "throughput_tasks_per_sec": output_metrics["num_tasks_processed"] / run_time_s if run_time_s else 0.0,
            "throughput_audio_hours_per_hour": (
                output_metrics["total_audio_duration_hours"] * 3600 / run_time_s if run_time_s else 0.0
            ),
        },
        "tasks": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--model-id", default=None, help="Pre-staged Qwen model directory; otherwise use tutorial model ID."
    )
    parser.add_argument("--executor", choices=["xenna", "ray_data"], default=None)
    parser.add_argument("--execution-mode", choices=["streaming", "batch"], default=None)
    parser.add_argument("--save-tasks", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    save_tasks = vars(args).pop("save_tasks")

    try:
        _check_fresh_outputs(Path(args.benchmark_results_path))
    except FileExistsError as error:
        logger.error(str(error))
        return 1

    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results.update(run_audio_qwen_omni_asr_benchmark(**vars(args)))
    except Exception as error:
        logger.exception("Qwen Omni ASR benchmark failed")
        results["metrics"]["error_message"] = str(error)
    finally:
        write_benchmark_results(results, args.benchmark_results_path, save_tasks=save_tasks)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
