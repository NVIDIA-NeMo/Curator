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

"""Benchmark the two-model Common Crawl extraction LLM judge example.

Prepare the 32,000-record input dataset as 64 shards from the Curator repository root with::

    python tutorials/eval/llm_judge/cc_extract_example/prepare_cc_extraction_dataset.py \
        --start-snapshot 2026-30 --end-snapshot 2026-30 \
        --download-dir {datasets_path}/llm_judge/cc_warcs \
        --output-path {datasets_path}/llm_judge/cc_extractions \
        --url-limit 64 --record-limit 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from utils import write_benchmark_results

from nemo_curator.eval.llm_judge import LLMJudgeWorkflow

_CURATOR_ROOT = Path(__file__).resolve().parents[2]


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


def _count_jsonl_rows(path: Path) -> int:
    files = _jsonl_files(path)
    if not files:
        msg = f"No JSONL files found under {path}"
        raise FileNotFoundError(msg)
    row_count = 0
    for file_path in files:
        with file_path.open(encoding="utf-8") as input_file:
            row_count += sum(1 for line in input_file if line.strip())
    return row_count


def _prepare_judge_config(  # noqa: C901
    source_path: Path,
    output_path: Path,
    *,
    qwen_model_path: Path,
    gemma_model_path: Path,
) -> Path:
    """Write a run-local config with absolute prompt and model paths."""
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        msg = f"Judge config must contain a mapping: {source_path}"
        raise TypeError(msg)

    model_paths = {"qwen": qwen_model_path, "gemma": gemma_model_path}
    models = config.get("models")
    if not isinstance(models, list):
        msg = "Judge config must define a models list"
        raise TypeError(msg)
    configured_aliases = {str(model.get("alias")) for model in models if isinstance(model, dict)}
    if configured_aliases != set(model_paths):
        msg = f"Expected qwen and gemma model aliases, found {sorted(configured_aliases)}"
        raise ValueError(msg)
    for model in models:
        model_path = model_paths[str(model["alias"])].resolve()
        if not model_path.is_dir():
            msg = f"Model path does not exist or is not a directory: {model_path}"
            raise FileNotFoundError(msg)
        model["model"] = str(model_path)

    stages = config.get("execution", {}).get("stages", [])
    for stage in stages:
        for judge in stage.get("judges", []):
            for key in ("prompt_path", "system_prompt_path"):
                if value := judge.get(key):
                    prompt_path = Path(value)
                    if not prompt_path.is_absolute():
                        judge[key] = str((source_path.parent / prompt_path).resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def run_llm_judge_benchmark(  # noqa: PLR0913
    *,
    judge_config: str,
    qwen_model_path: str,
    gemma_model_path: str,
    language: str,
    fasttext_langid_model_path: str,
    input_path: str,
    output_path: str,
    benchmark_results_path: str,
) -> dict[str, Any]:
    """Run the tutorial LLM judge workflow and collect benchmark metrics."""
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    benchmark_results_path_obj = Path(benchmark_results_path)
    judge_config_path = Path(judge_config)
    if not judge_config_path.is_absolute():
        judge_config_path = _CURATOR_ROOT / judge_config_path
    prepared_config = _prepare_judge_config(
        judge_config_path.resolve(),
        benchmark_results_path_obj / "resolved_judge_config.yaml",
        qwen_model_path=Path(qwen_model_path),
        gemma_model_path=Path(gemma_model_path),
    )
    input_row_count = _count_jsonl_rows(input_path_obj)

    logger.info(f"Judge config: {prepared_config}")
    logger.info(f"Input path: {input_path_obj}")
    logger.info(f"Input rows: {input_row_count}")
    logger.info(f"Output path: {output_path_obj}")
    logger.info(f"FastText language ID model: {fasttext_langid_model_path}")

    workflow = LLMJudgeWorkflow(
        judge_config=prepared_config,
        input_path=str(input_path_obj),
        output_path=str(output_path_obj),
        input_format="jsonl",
        output_format="jsonl",
        language=language,
        fasttext_langid_model_path=fasttext_langid_model_path,
    )

    run_start_time = time.perf_counter()
    workflow_result = workflow.run()
    run_time_taken = time.perf_counter() - run_start_time
    output_row_count = _count_jsonl_rows(output_path_obj)
    row_count_match = input_row_count == output_row_count
    throughput_rows_per_sec = output_row_count / run_time_taken if run_time_taken > 0 else 0.0

    logger.success(f"LLM judge benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Input: {input_row_count} rows")
    logger.success(f"Output: {output_row_count} rows")
    logger.success(f"Throughput: {throughput_rows_per_sec:.2f} rows/sec")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "input_row_count": input_row_count,
            "output_row_count": output_row_count,
            "input_output_row_count_match": row_count_match,
            "throughput_rows_per_sec": throughput_rows_per_sec,
        },
        "tasks": workflow_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--qwen-model-path", required=True)
    parser.add_argument("--gemma-model-path", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--fasttext-langid-model-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    success_code = 1
    try:
        result_dict.update(
            run_llm_judge_benchmark(
                judge_config=args.judge_config,
                qwen_model_path=args.qwen_model_path,
                gemma_model_path=args.gemma_model_path,
                language=args.language,
                fasttext_langid_model_path=args.fasttext_langid_model_path,
                input_path=args.input_path,
                output_path=args.output_path,
                benchmark_results_path=args.benchmark_results_path,
            )
        )
        success_code = 0
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
