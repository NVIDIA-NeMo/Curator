# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Common Crawl E2E pipeline benchmark for nightly benchmarking.

Runs a full end-to-end text processing pipeline:
  CommonCrawlDownloadExtractStage → AddId → HeuristicFilters → FastTextLangId
  → QualityClassifier (optional) → FineWebEduClassifier (optional) → Writer

This benchmark covers stages not tested by the existing common_crawl_benchmark.py,
specifically: AddId, comprehensive heuristic filters, FastText language identification,
and GPU-based quality/education classifiers.

Based on patterns from tutorials/text/download-and-extract/download_extract_tutorial.ipynb
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Literal

import ray
from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.download.common_crawl.stage import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.classifiers import FineWebEduClassifier, QualityClassifier
from nemo_curator.stages.text.filters import (
    FastTextLangId,
    PunctuationFilter,
    RepeatedLinesFilter,
    RepeatingTopNGramsFilter,
    UrlsFilter,
    WordCountFilter,
)
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.stages.text.modules.add_id import AddId
from nemo_curator.stages.text.modules.score_filter import ScoreFilter


def create_e2e_pipeline(  # noqa: PLR0913
    download_dir: Path,
    output_dir: Path,
    snapshot: str,
    output_format: Literal["parquet", "jsonl"],
    html_extraction_algorithm: str,
    url_limit: int | None,
    record_limit: int | None,
    fasttext_model_path: str,
    min_langid_score: float,
    min_words: int,
    max_words: int,
    max_url_ratio: float,
    max_repeated_lines_ratio: float,
    max_repeating_ngram_ratio: float,
    max_punctuation_ratio: float,
    enable_quality_classifier: bool,
    quality_filter_by: list[str] | None,
    enable_edu_classifier: bool,
    edu_filter_by: list[str] | None,
    classifier_batch_size: int,
) -> Pipeline:
    """Create the E2E pipeline with CC download, AddId, heuristic filters, language ID, and classifiers.

    Args:
        download_dir: Directory to store downloaded WARC files.
        output_dir: Directory to write output files.
        snapshot: Single CC snapshot to process (e.g., "2024-30" in YYYY-WeekNumber format).
        output_format: Output format ("parquet" or "jsonl").
        html_extraction_algorithm: HTML extraction algorithm (justext, resiliparse, trafilatura).
        url_limit: Maximum number of WARC files to download.
        record_limit: Maximum records per WARC file.
        fasttext_model_path: Path to FastText language ID model (lid.176.bin).
        min_langid_score: Minimum language ID confidence score.
        min_words: Minimum word count for documents.
        max_words: Maximum word count for documents.
        max_url_ratio: Maximum URL-to-text ratio.
        max_repeated_lines_ratio: Maximum ratio of repeated lines.
        max_repeating_ngram_ratio: Maximum ratio of repeating top n-grams.
        max_punctuation_ratio: Maximum ratio of sentences without punctuation.
        enable_quality_classifier: Whether to enable the QualityClassifier stage.
        quality_filter_by: Labels to filter by for quality classification (e.g., ["High", "Medium"]).
        enable_edu_classifier: Whether to enable the FineWebEduClassifier stage.
        edu_filter_by: Labels to filter by for edu classification (e.g., ["high_quality"]).
        classifier_batch_size: Batch size for model inference in classifiers.

    Returns:
        Pipeline: Configured E2E pipeline.
    """
    pipeline = Pipeline(
        name="cc_e2e_pipeline",
        description="E2E Common Crawl pipeline with AddId, heuristic filters, language ID, and classifiers",
    )

    # Stage 1: Common Crawl Download + Extract
    pipeline.add_stage(
        CommonCrawlDownloadExtractStage(
            start_snapshot=snapshot,
            end_snapshot=snapshot,
            download_dir=str(download_dir),
            crawl_type="main",
            html_extraction=html_extraction_algorithm,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=True,
        )
    )

    # Stage 2: Add unique document IDs
    pipeline.add_stage(
        AddId(
            id_field="doc_id",
            id_prefix="cc",
            overwrite=False,
        )
    )

    # Stage 3: Heuristic filters (comprehensive set)
    heuristic_filters = [
        WordCountFilter(min_words=min_words, max_words=max_words),
        UrlsFilter(max_url_to_text_ratio=max_url_ratio),
        RepeatedLinesFilter(max_repeated_line_fraction=max_repeated_lines_ratio),
        RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=max_repeating_ngram_ratio),
        PunctuationFilter(max_num_sentences_without_endmark_ratio=max_punctuation_ratio),
    ]

    pipeline.add_stage(
        ScoreFilter(
            filter_obj=heuristic_filters,
            text_field="text",
            score_field=[
                "word_count_score",
                "url_ratio_score",
                "repeated_lines_score",
                "ngram_ratio_score",
                "punctuation_score",
            ],
        )
    )

    # Stage 4: FastText Language ID filter
    pipeline.add_stage(
        ScoreFilter(
            filter_obj=FastTextLangId(model_path=fasttext_model_path, min_langid_score=min_langid_score),
            text_field="text",
            score_field="langid_score",
        )
    )

    # Stage 5: Quality Classifier (GPU-based)
    if enable_quality_classifier:
        pipeline.add_stage(
            QualityClassifier(
                text_field="text",
                label_field="quality_pred",
                score_field="quality_score",
                filter_by=quality_filter_by,
                model_inference_batch_size=classifier_batch_size,
            )
        )

    # Stage 6: FineWeb Edu Classifier (GPU-based)
    if enable_edu_classifier:
        pipeline.add_stage(
            FineWebEduClassifier(
                text_field="text",
                filter_by=edu_filter_by,
                model_inference_batch_size=classifier_batch_size,
            )
        )

    # Stage 7: Write output
    if output_format == "jsonl":
        writer = JsonlWriter(path=str(output_dir), write_kwargs={"force_ascii": False})
    else:
        writer = ParquetWriter(path=str(output_dir))
    pipeline.add_stage(writer)

    return pipeline


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the E2E pipeline benchmark and collect metrics.

    Args:
        args: Parsed command line arguments.

    Returns:
        dict: Benchmark results containing params, metrics, and tasks.
    """
    download_dir = Path(args.download_path).resolve()
    download_dir.mkdir(exist_ok=True, parents=True)

    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    pipeline = create_e2e_pipeline(
        download_dir=download_dir,
        output_dir=output_dir,
        snapshot=args.snapshot,
        output_format=args.output_format,
        html_extraction_algorithm=args.html_extraction,
        url_limit=args.url_limit,
        record_limit=args.record_limit,
        fasttext_model_path=args.fasttext_model_path,
        min_langid_score=args.min_langid_score,
        min_words=args.min_words,
        max_words=args.max_words,
        max_url_ratio=args.max_url_ratio,
        max_repeated_lines_ratio=args.max_repeated_lines_ratio,
        max_repeating_ngram_ratio=args.max_repeating_ngram_ratio,
        max_punctuation_ratio=args.max_punctuation_ratio,
        enable_quality_classifier=args.enable_quality_classifier,
        quality_filter_by=args.quality_filter_by,
        enable_edu_classifier=args.enable_edu_classifier,
        edu_filter_by=args.edu_filter_by,
        classifier_batch_size=args.classifier_batch_size,
    )

    # Select executor
    if args.executor == "xenna":
        from nemo_curator.backends.xenna.executor import XennaExecutor

        executor = XennaExecutor()
    elif args.executor == "ray_data":
        from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor
        executor = RayDataExecutor()
    else:
        msg = f"Invalid executor type: {args.executor}"
        raise ValueError(msg)

    logger.info("Starting CC E2E pipeline execution...")
    logger.info(f"Snapshot: {args.snapshot}")
    logger.info(f"URL limit: {args.url_limit}")
    logger.info(f"FastText model: {args.fasttext_model_path}")
    logger.info(f"Quality classifier enabled: {args.enable_quality_classifier}")
    logger.info(f"Edu classifier enabled: {args.enable_edu_classifier}")

    start = time.perf_counter()

    try:
        results = pipeline.run(executor, initial_tasks=None)
        success = True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed: {e}")
        results = []
        success = False

    elapsed = time.perf_counter() - start

    total_documents_output = sum(task.num_items for task in results) if results else 0

    return {
        "params": {
            "download_path": str(download_dir),
            "output_path": str(output_dir),
            "snapshot": args.snapshot,
            "output_format": args.output_format,
            "html_extraction": args.html_extraction,
            "url_limit": args.url_limit,
            "record_limit": args.record_limit,
            "fasttext_model_path": args.fasttext_model_path,
            "min_langid_score": args.min_langid_score,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "max_url_ratio": args.max_url_ratio,
            "max_repeated_lines_ratio": args.max_repeated_lines_ratio,
            "max_repeating_ngram_ratio": args.max_repeating_ngram_ratio,
            "max_punctuation_ratio": args.max_punctuation_ratio,
            "enable_quality_classifier": args.enable_quality_classifier,
            "quality_filter_by": args.quality_filter_by,
            "enable_edu_classifier": args.enable_edu_classifier,
            "edu_filter_by": args.edu_filter_by,
            "classifier_batch_size": args.classifier_batch_size,
            "executor": args.executor,
            "ray_temp_dir": args.ray_temp_dir,
            "num_gpus": args.num_gpus,
            "num_cpus": args.num_cpus,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": elapsed,
            "num_output_tasks": len(results) if results else 0,
            "total_documents_output": total_documents_output,
            "throughput_docs_per_sec": total_documents_output / elapsed if elapsed > 0 else 0,
        },
        "tasks": results or [],
    }


def write_results(benchmark_results_path: str, results: dict) -> None:
    """Write benchmark results to the output directory.

    Args:
        benchmark_results_path: Directory to write results.
        results: Benchmark results dictionary.
    """
    out = Path(benchmark_results_path)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "params.json", "w") as f:
        json.dump(results["params"], f, indent=2)
    with open(out / "metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)
    with open(out / "tasks.pkl", "wb") as f:
        pickle.dump(results["tasks"], f)


def main() -> int:
    """Main entry point for the benchmark script."""
    p = argparse.ArgumentParser(description="Common Crawl E2E pipeline benchmark")

    # Contract arg for nightly driver
    p.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")

    # Pipeline configuration - Common Crawl
    # Snapshot format: YYYY-WeekNumber (e.g., "2024-30" for week 30 of 2024)
    # See: https://data.commoncrawl.org/ for valid snapshots
    p.add_argument("--download_path", type=str, default="./cc_e2e_downloads")
    p.add_argument("--output_path", type=str, default="./cc_e2e_output")
    p.add_argument("--snapshot", type=str, default="2024-30", help="CC snapshot in YYYY-WeekNumber format")
    p.add_argument("--output_format", type=str, default="jsonl", choices=["parquet", "jsonl"])
    p.add_argument(
        "--html_extraction", type=str, default="justext", choices=["justext", "resiliparse", "trafilatura"]
    )
    p.add_argument("--url_limit", type=int, default=2, help="Max WARC files to download")
    p.add_argument("--record_limit", type=int, default=1000, help="Max records per WARC file")

    # FastText Language ID configuration
    p.add_argument("--fasttext_model_path", type=str, required=True, help="Path to FastText lid.176.bin model")
    p.add_argument("--min_langid_score", type=float, default=0.3, help="Minimum language ID confidence")

    # Heuristic filter thresholds
    p.add_argument("--min_words", type=int, default=50, help="Minimum word count")
    p.add_argument("--max_words", type=int, default=100000, help="Maximum word count")
    p.add_argument("--max_url_ratio", type=float, default=0.2, help="Maximum URL-to-text ratio")
    p.add_argument("--max_repeated_lines_ratio", type=float, default=0.7, help="Maximum repeated lines ratio")
    p.add_argument("--max_repeating_ngram_ratio", type=float, default=0.2, help="Maximum repeating n-gram ratio")
    p.add_argument("--max_punctuation_ratio", type=float, default=0.85, help="Maximum sentences without punctuation")

    # Quality/Edu classifier configuration
    p.add_argument(
        "--enable_quality_classifier", action="store_true", help="Enable QualityClassifier stage (GPU-based)"
    )
    p.add_argument(
        "--quality_filter_by",
        type=str,
        nargs="*",
        default=None,
        help="Labels to keep for quality classifier (e.g., High Medium)",
    )
    p.add_argument(
        "--enable_edu_classifier", action="store_true", help="Enable FineWebEduClassifier stage (GPU-based)"
    )
    p.add_argument(
        "--edu_filter_by",
        type=str,
        nargs="*",
        default=None,
        help="Labels to keep for edu classifier (e.g., high_quality)",
    )
    p.add_argument(
        "--classifier_batch_size", type=int, default=256, help="Batch size for classifier model inference"
    )

    # Executor selection
    p.add_argument("--executor", type=str, default="ray_data", choices=["xenna", "ray_data"])

    # Ray cluster configuration
    p.add_argument("--ray_temp_dir", type=str, default="/raid/vjawa/ray_tmp", help="Ray temporary directory")
    p.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs for Ray cluster")
    p.add_argument("--num_cpus", type=int, default=None, help="Number of CPUs for Ray cluster")

    # HuggingFace model cache
    p.add_argument("--hf_home", type=str, default="/raid/vjawa/hf_cache", help="HuggingFace cache directory")

    args = p.parse_args()

    logger.info("=== CC E2E Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results = run_benchmark(args)
    write_results(args.benchmark_results_path, results)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
