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

"""OCR (NemotronOCR-v2 + Nemotron-Nano-Omni scoring QA) pipeline benchmark.

Reuses the pipeline and argparser from
``tutorials/synthetic/omni/ocr_pipeline.py``.

This is a *functional / async-throughput smoke* benchmark for the full pipeline,
including the ``OCRScoringQAStage`` which issues concurrent (async) calls to the
NVIDIA Inference API. Because the scoring stage is bound by an external service,
its wall-clock is not a clean perf-regression signal — the gating metrics are
local correctness (images processed, conversations generated, API success rate),
while throughput numbers are reported for tracking only.

Requires ``NVINFERENCE_API_KEY`` to be set for the scoring stage.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "synthetic" / "omni"))

from ocr_pipeline import build_pipeline, create_ocr_argparser  # noqa: E402


def _summarize_stage_perf(output_tasks: list) -> dict[str, dict[str, float]]:
    """Roll up per-stage processing times across all output tasks."""
    stage_perf: dict[str, list[float]] = {}
    for task in output_tasks:
        for perf in getattr(task, "_stage_perf", []):
            stage_perf.setdefault(perf.stage_name, []).append(perf.process_time)
    return {
        name: {
            "count": len(times),
            "total_s": sum(times),
            "mean_s": sum(times) / len(times) if times else 0,
            "min_s": min(times) if times else 0,
            "max_s": max(times) if times else 0,
        }
        for name, times in stage_perf.items()
    }


def run_ocr_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the OCR pipeline and collect metrics."""
    executor = setup_executor(args.executor)

    logger.info(f"HF dataset: {args.hf_dataset} (split={args.hf_split}, limit={args.hf_limit})")
    logger.info(f"Image cache: {args.hf_image_dir}  Output: {args.output_path}")
    logger.info(f"Scoring QA: run={args.run_scoring_qa} model={args.scoring_qa_model_id}")

    pipeline = build_pipeline(args)

    run_start = time.perf_counter()
    success = False
    output_tasks: list = []
    num_images = num_with_ocr = num_eligible = num_responded = num_conversations = 0

    try:
        logger.info(f"Pipeline description:\n{pipeline.describe()}")
        output_tasks = pipeline.run(executor)
        run_time = time.perf_counter() - run_start

        for task in output_tasks:
            data = getattr(task, "data", None)
            if data is None:
                continue
            num_images += 1
            ocr_dense = getattr(data, "ocr_dense", None)
            if ocr_dense:
                num_with_ocr += 1
                num_eligible += 1  # tasks with bboxes are the ones sent to the scoring model
                if getattr(data, "ocr_scoring_response_raw", None):
                    num_responded += 1
            if getattr(data, "conversation", None) is not None:
                num_conversations += 1

        logger.success(f"Benchmark completed in {run_time:.2f}s — {num_images} images, {num_conversations} convs")
        success = True

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        run_time = time.perf_counter() - run_start

    scoring_success_rate = (num_responded / num_eligible) if num_eligible else 0.0

    return {
        "params": {
            "executor": args.executor,
            "hf_dataset": args.hf_dataset,
            "hf_split": args.hf_split,
            "hf_limit": args.hf_limit,
            "run_scoring_qa": args.run_scoring_qa,
            "scoring_qa_model_id": args.scoring_qa_model_id,
            "scoring_qa_dense_dump_prob": args.scoring_qa_dense_dump_prob,
            "output_path": str(args.output_path),
            "benchmark_results_path": str(args.benchmark_results_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time,
            "num_images_processed": num_images,
            "num_images_with_ocr": num_with_ocr,
            "num_conversations_generated": num_conversations,
            "scoring_api_success_rate": scoring_success_rate,
            "throughput_images_per_sec": num_images / run_time if run_time > 0 else 0,
            "stage_performance": _summarize_stage_perf(output_tasks),
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = create_ocr_argparser()
    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        required=True,
        help="Path to write benchmark results (params.json, metrics.json, tasks.pkl)",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for pipeline execution",
    )
    args = parser.parse_args()

    logger.info("=== OCR Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results = run_ocr_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
