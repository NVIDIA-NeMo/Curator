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

"""Math pipeline benchmark for nightly benchmarking.

Runs a composable math curation pipeline with optional stages:
  - Always: ParquetReader -> MathExtractStage -> JsonlWriter
  - --run-classifier: adds FineMathClassifier after extraction
  - --run-llm-cleanup: adds TokenSplitter -> LLMCleanup -> ChunkMerge after extraction

Different YAML entries activate different stage combinations.

Example usage:
  # Preprocess only (extraction)
  python math_pipeline_benchmark.py --benchmark-results-path=/tmp/results \\
      --input-path=/datasets/finemath4plus/enriched --output-path=/tmp/output

  # Preprocess + classifier
  python math_pipeline_benchmark.py --benchmark-results-path=/tmp/results \\
      --input-path=/datasets/finemath4plus/enriched --output-path=/tmp/output \\
      --run-classifier

  # Preprocess + LLM cleanup
  python math_pipeline_benchmark.py --benchmark-results-path=/tmp/results \\
      --input-path=/datasets/finemath4plus/enriched --output-path=/tmp/output \\
      --run-llm-cleanup --model=microsoft/phi-4 --prompt=HTML_TO_TEXT_PROMPT \\
      --chunk-data --chunk-length=4096
"""

import argparse
import time
from pathlib import Path

import ray.data
from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.math.classifiers.finemath import FineMathClassifier
from nemo_curator.stages.math.download.extract import MathContentExtractor, MathExtractStage
from nemo_curator.stages.math.modifiers.chunking import TokenSplitterStage
from nemo_curator.stages.math.modifiers.llm_cleanup import LLMCleanupStage
from nemo_curator.stages.math.modifiers.merge_chunks import ChunkMergeStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils import prompts
from nemo_curator.utils.file_utils import get_all_file_paths_under

MIN_HIGH_QUALITY_SCORE = 3


def _fill_null_text(text: str | None) -> str:
    import pandas as pd

    if pd.isna(text) or text is None:
        return ""
    return str(text)


def create_math_pipeline(args: argparse.Namespace, input_files: list[str]) -> Pipeline:
    """Build the math benchmark pipeline with stages selected by CLI flags."""
    pipeline = Pipeline(
        name="math_pipeline_benchmark",
        description="Math curation benchmark: extract + optional classifier/LLM cleanup",
    )

    pipeline.add_stage(ParquetReader(file_paths=input_files, blocksize=args.input_blocksize))

    pipeline.add_stage(
        MathExtractStage(
            extractor=MathContentExtractor(
                binary_column="binary_content",
                url_column="url",
                mime_type_column="content_mime_type",
            ),
            add_filename_column=False,
        )
    )

    if args.run_classifier:
        pipeline.add_stage(
            FineMathClassifier(text_field="text").with_(
                {"finemath_classifier_model": {"resources": Resources(cpus=1.0, gpus=1.0)}}
            )
        )

    if args.run_llm_cleanup:
        pipeline.add_stage(Modify(modifier_fn=_fill_null_text, input_fields="text", output_fields="text"))

        if args.chunk_data and args.chunk_length:
            pipeline.add_stage(
                TokenSplitterStage(
                    model_name=args.model,
                    text_field="text",
                    max_length_tokens=args.chunk_length,
                )
            )

        try:
            system_prompt = getattr(prompts, args.prompt)
        except AttributeError:
            logger.warning(f"Prompt '{args.prompt}' not found in prompts module, using as literal string.")
            system_prompt = args.prompt

        pipeline.add_stage(
            LLMCleanupStage(
                model=args.model,
                system_prompt=system_prompt,
                text_field="text",
                output_field="cleaned_text",
                max_model_len=args.max_model_len,
            ).with_(resources=Resources(cpus=1.0, gpus=1.0))
        )

        if args.chunk_data and args.chunk_length:
            pipeline.add_stage(
                ChunkMergeStage(
                    text_field="cleaned_text",
                    raw_text_field="text",
                    chunk_id_field="chunk_id",
                    groupby_columns=["url"],
                )
            )

    pipeline.add_stage(JsonlWriter(path=str(args.output_path)))

    return pipeline


def compute_extraction_metrics(output_dir: str) -> dict:
    """Compute post-hoc extraction metrics from output JSONL files."""
    metrics = {}
    try:
        jsonl_files = get_all_file_paths_under(output_dir, keep_extensions=[".jsonl"])
        if not jsonl_files:
            return metrics

        ds = ray.data.read_json(jsonl_files).select_columns(["type", "text"])

        def _count_types(batch: dict) -> dict:
            types = batch.get("type", [])
            texts = batch.get("text", [])
            return {
                "html": [sum(1 for t in types if t == "html")],
                "text": [sum(1 for t in types if t == "text")],
                "notebook": [sum(1 for t in types if t == "notebook")],
                "html_empty": [
                    sum(1 for t, tx in zip(types, texts, strict=False) if t == "html" and not str(tx or "").strip())
                ],
            }

        counts = ds.map_batches(_count_types, batch_format="numpy")
        totals = dict.fromkeys(("html", "text", "notebook", "html_empty"), 0)
        for row in counts.iter_rows():
            for k in totals:
                totals[k] += int(row[k])

        metrics["type_html_count"] = totals["html"]
        metrics["type_text_count"] = totals["text"]
        metrics["type_notebook_count"] = totals["notebook"]
        metrics["html_empty_text_count"] = totals["html_empty"]

    except Exception as e:
        logger.warning(f"Could not compute extraction metrics: {e}")

    return metrics


def compute_classifier_metrics(output_dir: str) -> dict:
    """Compute post-hoc classifier metrics from output JSONL files."""
    metrics = {}
    try:
        jsonl_files = get_all_file_paths_under(output_dir, keep_extensions=[".jsonl"])
        if not jsonl_files:
            return metrics

        ds = ray.data.read_json(jsonl_files).select_columns(["finemath_int_scores"])

        score_counts = [0] * 6
        score_sum = 0
        total = 0
        for batch in ds.iter_batches(batch_format="numpy"):
            for s in batch["finemath_int_scores"]:
                score_int = int(s)
                score_counts[min(score_int, 5)] += 1
                score_sum += score_int
                total += 1

        if total > 0:
            metrics["mean_finemath_score"] = score_sum / total

            for i in range(6):
                metrics[f"score_distribution_{i}"] = score_counts[i]

            metrics["docs_score_ge_3"] = sum(score_counts[MIN_HIGH_QUALITY_SCORE:])

    except Exception as e:
        logger.warning(f"Could not compute classifier metrics: {e}")

    return metrics


def compute_llm_cleanup_metrics(output_dir: str) -> dict:
    """Compute post-hoc LLM cleanup metrics from output JSONL files."""
    metrics = {}
    try:
        jsonl_files = get_all_file_paths_under(output_dir, keep_extensions=[".jsonl"])
        if not jsonl_files:
            return metrics

        ds = ray.data.read_json(jsonl_files).select_columns(["cleaned_text"])

        def _text_stats(batch: dict) -> dict:
            lengths = [len(str(t or "")) for t in batch.get("cleaned_text", [])]
            return {
                "count": [len(lengths)],
                "total_length": [sum(lengths)],
                "no_content": [sum(1 for ln in lengths if ln == 0)],
            }

        agg = ds.map_batches(_text_stats, batch_format="numpy")
        totals = {"count": 0, "total_length": 0, "no_content": 0}
        for row in agg.iter_rows():
            for k in totals:
                totals[k] += int(row[k])

        metrics["num_output_documents_post_merge"] = totals["count"]
        if totals["count"] > 0:
            metrics["avg_output_text_length"] = totals["total_length"] / totals["count"]
        metrics["no_useful_content_count"] = totals["no_content"]

    except Exception as e:
        logger.warning(f"Could not compute LLM cleanup metrics: {e}")

    return metrics


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the math pipeline benchmark and collect metrics."""
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load input files with optional size limiting
    if args.dataset_size_gb:
        input_files = load_dataset_files(input_path, dataset_size_gb=args.dataset_size_gb)
    else:
        input_files = load_dataset_files(input_path, dataset_ratio=1.0)

    num_input_files = len(input_files)
    logger.info(f"Input files: {num_input_files}")
    logger.info(f"Output path: {output_path}")
    stages = "extract"
    if args.run_classifier:
        stages += " + classifier"
    if args.run_llm_cleanup:
        stages += " + llm_cleanup"
    logger.info(f"Stages: {stages}")

    pipeline = create_math_pipeline(args, input_files)
    executor = setup_executor(args.executor)

    logger.info(f"Pipeline description:\n{pipeline.describe()}")
    logger.info("Starting math pipeline execution...")

    start = time.perf_counter()

    try:
        results = pipeline.run(executor, initial_tasks=None)
        success = True
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results = []
        success = False

    elapsed = time.perf_counter() - start

    num_input_documents = TaskPerfUtils.get_aggregated_stage_stat(results, "extract_", "num_items_processed")
    num_output_documents = TaskPerfUtils.get_aggregated_stage_stat(results, "jsonl_writer", "num_items_processed")

    metrics = {
        "is_success": success,
        "time_taken_s": elapsed,
        "num_output_tasks": len(results) if results else 0,
        "num_input_documents": int(num_input_documents),
        "num_output_documents": int(num_output_documents),
        "num_input_files": num_input_files,
        "throughput_docs_per_sec": num_input_documents / elapsed if elapsed > 0 else 0,
    }

    if num_input_documents > 0:
        metrics["extraction_success_rate"] = num_output_documents / num_input_documents

    task_metrics = TaskPerfUtils.aggregate_task_metrics(results, prefix="task")
    metrics.update(task_metrics)

    # Domain-specific post-hoc metrics from output files
    output_dir = str(output_path)
    metrics.update(compute_extraction_metrics(output_dir))

    if args.run_classifier:
        metrics.update(compute_classifier_metrics(output_dir))

    if args.run_llm_cleanup:
        metrics.update(compute_llm_cleanup_metrics(output_dir))

    logger.success(f"Benchmark completed in {elapsed:.2f}s")
    logger.success(f"Throughput: {metrics['throughput_docs_per_sec']:.1f} docs/sec")

    return {
        "params": {"args": vars(args), "num_input_files": num_input_files},
        "metrics": metrics,
        "tasks": results or [],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Math pipeline benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")

    # Input/output
    p.add_argument("--input-path", type=str, required=True, help="Path to enriched parquet dataset")
    p.add_argument("--output-path", type=str, required=True, help="Output directory for JSONL results")
    p.add_argument("--dataset-size-gb", type=float, default=None, help="Limit input dataset size (GB); None = use all")
    p.add_argument(
        "--input-blocksize",
        type=str,
        default=None,
        help="Merge input files into tasks of this size (e.g. '50MB'); None = 1 file per task",
    )

    # Stage selection
    stage_group = p.add_argument_group("Stage Selection")
    stage_group.add_argument("--run-classifier", action="store_true", help="Add FineMath classifier after extraction")
    stage_group.add_argument("--run-llm-cleanup", action="store_true", help="Add LLM cleanup after extraction")

    # LLM cleanup options
    llm_group = p.add_argument_group("LLM Cleanup Options (requires --run-llm-cleanup)")
    llm_group.add_argument("--model", type=str, default="microsoft/phi-4", help="LLM model identifier")
    llm_group.add_argument(
        "--prompt", type=str, default="HTML_TO_TEXT_PROMPT", help="Prompt name from prompts module or literal string"
    )
    llm_group.add_argument("--chunk-data", action="store_true", help="Enable token-based chunking before LLM")
    llm_group.add_argument("--chunk-length", type=int, default=4096, help="Max tokens per chunk")
    llm_group.add_argument("--max-model-len", type=int, default=None, help="Max model context length for vLLM")

    # Executor
    p.add_argument("--executor", type=str, default="xenna", choices=["xenna", "ray_data"])

    args = p.parse_args()

    if args.run_llm_cleanup and not args.model:
        p.error("--model is required when using --run-llm-cleanup")

    logger.info("=== Math Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results = {
        "params": {"args": vars(args), "num_input_files": 0},
        "metrics": {
            "is_success": False,
            "time_taken_s": 0,
            "num_output_tasks": 0,
            "num_input_documents": 0,
            "num_output_documents": 0,
            "throughput_docs_per_sec": 0,
        },
        "tasks": [],
    }
    try:
        results = run_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
