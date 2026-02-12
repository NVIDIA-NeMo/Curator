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

"""ArXiv E2E pipeline benchmark for nightly benchmarking.

Runs a full end-to-end text processing pipeline:
  ArxivExtract → AddId → HeuristicFilters → QualityClassifier → FineWebEduClassifier → Writer

Supports two modes:
  1. Download mode: Downloads ArXiv papers from S3 (s3://arxiv/src/)
  2. Local tar files mode (default): Processes tar files from local directory (--tar_input_path)

Example usage:
  # Local tar files mode - process pre-downloaded tar files (default)
  python arxiv_e2e_pipeline_benchmark.py --benchmark-results-path=/tmp/results \\
      --tar-input-path=/datasets/prospector-lm/arxiv_downloads

  # Download mode - download 2 tar files from S3
  python arxiv_e2e_pipeline_benchmark.py --benchmark-results-path=/tmp/results \\
      --download-from-s3 --url-limit=2
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.classifiers import FineWebEduClassifier, QualityClassifier
from nemo_curator.stages.text.download.arxiv import ArxivDownloadExtractStage
from nemo_curator.stages.text.download.arxiv.extract import ArxivExtractor
from nemo_curator.stages.text.download.arxiv.iterator import ArxivIterator
from nemo_curator.stages.text.download.base import URLGenerator
from nemo_curator.stages.text.download.base.iterator import DocumentIterateExtractStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
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
from nemo_curator.tasks import DocumentBatch, _EmptyTask
from nemo_curator.tasks.utils import TaskPerfUtils

# Default filter parameters
DEFAULT_MIN_WORDS = 100
DEFAULT_MAX_WORDS = 500000
DEFAULT_MAX_URL_RATIO = 0.1
DEFAULT_MAX_REPEATED_LINES_RATIO = 0.5
DEFAULT_MAX_REPEATING_NGRAM_RATIO = 0.3
DEFAULT_MAX_PUNCTUATION_RATIO = 0.9
DEFAULT_CLASSIFIER_BATCH_SIZE = 256

# FastText Language ID defaults
DEFAULT_MIN_LANGID_SCORE = 0.3


@dataclass
class LocalTarUrlGenerator(URLGenerator):
    """Generates URLs (paths) from local tar files in a directory."""

    tar_dir: str
    limit: int | None = None

    def generate_urls(self) -> list[str]:
        """List all tar files in the directory."""
        tar_path = Path(self.tar_dir)
        tar_files = sorted(tar_path.glob("*.tar"))
        urls = [str(f) for f in tar_files]
        if self.limit:
            urls = urls[: self.limit]
        logger.info(f"Found {len(urls)} tar files in {self.tar_dir}")
        return urls


@dataclass
class LocalArxivExtractStage(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage for processing local ArXiv tar files.

    This stage:
    1. Lists local tar files from a directory
    2. Iterates through tar files extracting LaTeX content
    3. Extracts and cleans text from LaTeX files
    """

    tar_dir: str
    url_limit: int | None = None
    record_limit: int | None = None
    add_filename_column: bool | str = True
    log_frequency: int = 1000

    def __post_init__(self) -> None:
        """Initialize the constituent stages."""
        # URL generation stage (lists local tar files)
        url_stage = URLGenerationStage(
            url_generator=LocalTarUrlGenerator(tar_dir=self.tar_dir, limit=self.url_limit),
            limit=self.url_limit,
        )

        # Iterate-extract stage (extracts records from tar files and cleans LaTeX to text)
        iterate_extract_stage = DocumentIterateExtractStage(
            iterator=ArxivIterator(log_frequency=self.log_frequency),
            extractor=ArxivExtractor(),
            record_limit=self.record_limit,
            add_filename_column=self.add_filename_column,
        )

        self.stages = [url_stage, iterate_extract_stage]
        self.name = "local_arxiv_extract"
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent stages."""
        return self.stages


def create_e2e_pipeline(  # noqa: PLR0913
    # Input source options
    tar_input_path: str | None,
    download_from_s3: bool,
    download_dir: Path,
    url_limit: int | None,
    record_limit: int | None,
    log_frequency: int,
    fasttext_langid_model_path: str | None,
    # Output options
    output_dir: Path,
    output_format: Literal["parquet", "jsonl"],
    # Filter parameters
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    max_url_ratio: float = DEFAULT_MAX_URL_RATIO,
    max_repeated_lines_ratio: float = DEFAULT_MAX_REPEATED_LINES_RATIO,
    max_repeating_ngram_ratio: float = DEFAULT_MAX_REPEATING_NGRAM_RATIO,
    max_punctuation_ratio: float = DEFAULT_MAX_PUNCTUATION_RATIO,
    # FastText Language ID parameters
    min_langid_score: float = DEFAULT_MIN_LANGID_SCORE,
    # Classifier parameters
    classifier_batch_size: int = DEFAULT_CLASSIFIER_BATCH_SIZE,
) -> Pipeline:
    """Create the E2E pipeline with configurable input source and processing stages.

    Args:
        tar_input_path: Path to directory containing ArXiv tar files.
        download_from_s3: If True, download from S3 instead of using local tar files.
        download_dir: Directory to store downloaded ArXiv tar files (when downloading).
        url_limit: Maximum number of ArXiv tar files to process.
        record_limit: Maximum records (papers) per tar file.
        log_frequency: How often to log extraction progress.
        output_dir: Directory to write output files.
        output_format: Output format ("parquet" or "jsonl").
        min_words: Minimum word count for documents.
        max_words: Maximum word count for documents.
        max_url_ratio: Maximum URL-to-text ratio.
        max_repeated_lines_ratio: Maximum ratio of repeated lines.
        max_repeating_ngram_ratio: Maximum ratio of repeating top n-grams.
        max_punctuation_ratio: Maximum ratio of sentences without punctuation.
        fasttext_langid_model_path: Path to FastText language ID model (lid.176.bin).
        min_langid_score: Minimum language ID confidence score.
        classifier_batch_size: Batch size for model inference in classifiers.

    Returns:
        Pipeline: Configured E2E pipeline.
    """
    pipeline = Pipeline(
        name="arxiv_e2e_pipeline",
        description="E2E ArXiv pipeline with AddId, heuristic filters, and classifiers",
    )

    # ========== INPUT STAGE ==========
    if download_from_s3:
        # Download Mode: Download from S3
        logger.info("Using ArXiv S3 download mode")
        pipeline.add_stage(
            ArxivDownloadExtractStage(
                download_dir=str(download_dir),
                url_limit=url_limit,
                record_limit=record_limit,
                add_filename_column=True,
                log_frequency=log_frequency,
                verbose=True,
            )
        )
    else:
        # Local Tar Files Mode: Process local tar files
        logger.info(f"Using local tar files from: {tar_input_path}")
        pipeline.add_stage(
            LocalArxivExtractStage(
                tar_dir=tar_input_path,
                url_limit=url_limit,
                record_limit=record_limit,
                add_filename_column=True,
                log_frequency=log_frequency,
            )
        )

    # Add unique document IDs
    pipeline.add_stage(
        AddId(
            id_field="doc_id",
            id_prefix="arxiv",
            overwrite=False,
        )
    )

    # ========== FILTER STAGES ==========
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

    # ========== LANGUAGE ID FILTER ==========
    pipeline.add_stage(
        ScoreFilter(
            filter_obj=FastTextLangId(model_path=fasttext_langid_model_path, min_langid_score=min_langid_score),
            text_field="text",
            score_field="langid_score",
        )
    )

    # ========== CLASSIFIER STAGES ==========
    pipeline.add_stage(
        QualityClassifier(
            text_field="text",
            label_field="quality_pred",
            score_field="quality_score",
            filter_by=None,
            model_inference_batch_size=classifier_batch_size,
        )
    )

    pipeline.add_stage(
        FineWebEduClassifier(
            text_field="text",
            filter_by=None,
            model_inference_batch_size=classifier_batch_size,
        )
    )

    # ========== OUTPUT STAGE ==========
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
        tar_input_path=args.tar_input_path,
        download_from_s3=args.download_from_s3,
        download_dir=download_dir,
        url_limit=args.url_limit,
        record_limit=args.record_limit,
        log_frequency=args.log_frequency,
        output_dir=output_dir,
        output_format=args.output_format,
        min_words=args.min_words,
        max_words=args.max_words,
        max_url_ratio=args.max_url_ratio,
        max_repeated_lines_ratio=args.max_repeated_lines_ratio,
        max_repeating_ngram_ratio=args.max_repeating_ngram_ratio,
        max_punctuation_ratio=args.max_punctuation_ratio,
        fasttext_langid_model_path=args.fasttext_langid_model_path,
        min_langid_score=args.min_langid_score,
        classifier_batch_size=args.classifier_batch_size,
    )

    executor = setup_executor(args.executor)

    # Log configuration
    logger.info("Starting ArXiv E2E pipeline execution...")
    if args.download_from_s3:
        logger.info(f"Input mode: S3 download (url_limit={args.url_limit}, record_limit={args.record_limit})")
    else:
        logger.info(f"Input mode: Local tar files from {args.tar_input_path}")

    start = time.perf_counter()
    results = pipeline.run(executor, initial_tasks=None)
    elapsed = time.perf_counter() - start

    # Calculate metrics from stage performance data
    num_tar_files = len(results) if results else 0
    num_input_documents = TaskPerfUtils.get_aggregated_stage_stat(results, "extract_", "num_items_processed")
    writer_stage_name = f"{args.output_format}_writer"
    num_output_documents = TaskPerfUtils.get_aggregated_stage_stat(results, writer_stage_name, "num_items_processed")
    throughput_tar_files_per_sec = num_tar_files / elapsed if elapsed > 0 else 0
    throughput_docs_per_sec = num_input_documents / elapsed if elapsed > 0 else 0

    logger.success(f"Benchmark completed in {elapsed:.2f}s")
    logger.success(f"Tar files processed: {num_tar_files}")
    logger.success(f"Input documents (rows extracted): {num_input_documents}")
    if num_input_documents > 0:
        logger.success(
            f"Output documents (rows after filtering): {num_output_documents} (kept {num_output_documents / num_input_documents * 100:.1f}%)"
        )
    else:
        logger.success("Output documents: 0")
    logger.success(
        f"Throughput: {throughput_tar_files_per_sec:.2f} tar files/sec, {throughput_docs_per_sec:.1f} docs/sec"
    )

    return {
        "params": {
            "tar_input_path": args.tar_input_path,
            "download_from_s3": args.download_from_s3,
            "download_path": str(download_dir),
            "output_path": str(output_dir),
            "output_format": args.output_format,
            "url_limit": args.url_limit,
            "record_limit": args.record_limit,
            "log_frequency": args.log_frequency,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "max_url_ratio": args.max_url_ratio,
            "max_repeated_lines_ratio": args.max_repeated_lines_ratio,
            "max_repeating_ngram_ratio": args.max_repeating_ngram_ratio,
            "max_punctuation_ratio": args.max_punctuation_ratio,
            "fasttext_langid_model_path": args.fasttext_langid_model_path,
            "min_langid_score": args.min_langid_score,
            "classifier_batch_size": args.classifier_batch_size,
            "executor": args.executor,
            "args": vars(args),
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed,
            "num_output_tasks": len(results) if results else 0,
            "num_tar_files": num_tar_files,
            "num_input_documents": num_input_documents,
            "num_output_documents": num_output_documents,
            "throughput_tar_files_per_sec": throughput_tar_files_per_sec,
            "throughput_docs_per_sec": throughput_docs_per_sec,
        },
        "tasks": results or [],
    }


def main() -> int:
    """Main entry point for the benchmark script."""
    p = argparse.ArgumentParser(
        description="ArXiv E2E pipeline benchmark", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Contract arg for nightly driver
    p.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")

    # ========== INPUT SOURCE OPTIONS ==========
    input_group = p.add_argument_group("Input Source", "Choose between local tar files or S3 download")
    input_group.add_argument(
        "--tar-input-path",
        type=str,
        help="Path to directory containing ArXiv tar files (required unless --download-from-s3 is set)",
    )
    input_group.add_argument(
        "--download-from-s3",
        action="store_true",
        help="Download tar files from S3 instead of using local files",
    )

    # ========== DOWNLOAD/PROCESSING OPTIONS ==========
    download_group = p.add_argument_group("Download/Processing Options")
    download_group.add_argument("--download-path", type=str, default="./arxiv_e2e_downloads")
    download_group.add_argument(
        "--url-limit", type=int, default=None, help="Max ArXiv tar files to process (None = all)"
    )
    download_group.add_argument(
        "--record-limit", type=int, default=None, help="Max papers per tar file (None = no limit)"
    )
    download_group.add_argument("--log-frequency", type=int, default=1000, help="Log progress every N papers")

    # ========== OUTPUT OPTIONS ==========
    output_group = p.add_argument_group("Output Options")
    output_group.add_argument("--output-path", type=str, default="./arxiv_e2e_output")
    output_group.add_argument("--output-format", type=str, default="jsonl", choices=["parquet", "jsonl"])

    # ========== FILTER OPTIONS ==========
    filter_group = p.add_argument_group("Filter Options")
    filter_group.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS, help="Minimum word count")
    filter_group.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="Maximum word count")
    filter_group.add_argument("--max-url-ratio", type=float, default=DEFAULT_MAX_URL_RATIO)
    filter_group.add_argument("--max-repeated-lines-ratio", type=float, default=DEFAULT_MAX_REPEATED_LINES_RATIO)
    filter_group.add_argument("--max-repeating-ngram-ratio", type=float, default=DEFAULT_MAX_REPEATING_NGRAM_RATIO)
    filter_group.add_argument("--max-punctuation-ratio", type=float, default=DEFAULT_MAX_PUNCTUATION_RATIO)

    # ========== LANGUAGE ID OPTIONS ==========
    langid_group = p.add_argument_group("Language ID Options")
    langid_group.add_argument(
        "--fasttext-langid-model-path",
        type=str,
        help="Path to FastText language ID model (lid.176.bin)",
    )
    langid_group.add_argument(
        "--min-langid-score",
        type=float,
        default=DEFAULT_MIN_LANGID_SCORE,
        help=f"Minimum language ID confidence score (default: {DEFAULT_MIN_LANGID_SCORE})",
    )

    # ========== CLASSIFIER OPTIONS ==========
    classifier_group = p.add_argument_group("Classifier Options")
    classifier_group.add_argument("--classifier-batch-size", type=int, default=DEFAULT_CLASSIFIER_BATCH_SIZE)

    # ========== EXECUTOR OPTIONS ==========
    p.add_argument("--executor", type=str, default="ray_data", choices=["xenna", "ray_data", "ray_actors"])

    args = p.parse_args()

    # Validate: tar_input_path is required when not downloading from S3
    if not args.download_from_s3 and not args.tar_input_path:
        p.error("--tar-input-path is required when not using --download-from-s3")

    logger.info("=== ArXiv E2E Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    # Initialize results with failure state - will be overwritten on success
    results = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
            "time_taken_s": 0,
            "num_output_tasks": 0,
            "num_tar_files": 0,
            "num_input_documents": 0,
            "num_output_documents": 0,
            "throughput_tar_files_per_sec": 0,
            "throughput_docs_per_sec": 0,
        },
        "tasks": [],
    }
    success_code = 0
    try:
        results = run_benchmark(args)
        success_code = 0 if results["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
