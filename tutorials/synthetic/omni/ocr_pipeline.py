"""OCR dense pipeline entry point.

Stages:
  1. ocr_nemotron   NemotronOCR-v2 (multilingual) word-level OCR → populates ocr_dense
  2. scoring_qa     Combined Gemini bbox scoring + multi-turn QA generation (optional)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# gRPC + Ray fork safety: prevents SIGSEGV in grpc_core::GetEnv() when Ray
# spawns new worker processes by forking a parent that has live threads.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.synthetic.omni.io import (
    JsonlTarImageReaderStage,
    ResultWriterStage,
    SkipProcessedStage,
    merge_output_shards,
)
from nemo_curator.stages.synthetic.omni.ocr_nemotron_v2 import OCRNemotronV2Stage
from nemo_curator.stages.synthetic.omni.ocr_scoring_qa import OCRScoringQAStage
from nemo_curator.tasks.ocr import OCRData


def _normalize_input_paths(
    input_path: Path | str | list[Path] | list[str],
) -> str | list[str]:
    if isinstance(input_path, (list, tuple)):
        return [str(p) for p in input_path]
    return str(input_path)


def create_ocr_pipeline(
    input_path: Path | str | list[Path] | list[str],
    output_path: Path,
    tar_base_path: Path | None = None,
    image_parent: Path | None = None,
    verbose: bool = False,
    resume: bool = False,
    valid_only: bool = False,
    num_workers: int | None = None,
    nemotron_model_dir: Path | None = None,
    run_scoring_qa: bool = False,
    scoring_qa_model_id: str = "gcp/google/gemini-3-flash-preview",
    scoring_qa_min_bbox_match: int = 7,
    scoring_qa_max_text_errors: int = 0,
    scoring_qa_fail_on_missing_text: bool = False,
    scoring_qa_dense_dump_prob: float = 0.10,
) -> Pipeline:
    """Create the OCR pipeline using NemotronOCR-v2 (multilingual) on every image.

    Reads images from JSONL+tar shards, runs NemotronOCR-v2 word-level OCR on
    every image, and optionally runs a combined Gemini bbox scoring + QA generation
    stage to produce training conversations.

    Args:
        input_path: Path to a JSONL file, a directory of JSONL files, or a list
            of JSONL file paths.
        output_path: Path to the output JSONL file.
        tar_base_path: Base directory for tar shards referenced by the JSONL.
            Defaults to the parent of ``input_path`` when a single path is given.
        image_parent: If set, image paths in the output are made relative to this
            directory (useful for portability across machines).
        verbose: Enable verbose reader logging.
        resume: If True, skip images already present in ``output_path``.
        valid_only: If True, exclude invalid records from the output.
        num_workers: Override the number of Xenna workers for all model stages.
            If None, the Xenna autoscaler decides.
        nemotron_model_dir: Path to the NemotronOCR-v2 model directory.
            If None, downloads from HuggingFace.
        run_scoring_qa: If True, run combined Gemini bbox scoring + QA generation.
        scoring_qa_model_id: NVIDIA Inference API model ID for scoring QA stage.
        scoring_qa_min_bbox_match: Minimum bbox_match score (0–10) for a valid bbox.
        scoring_qa_max_text_errors: Maximum text_errors count for a valid bbox.
        scoring_qa_fail_on_missing_text: If True, mark the image invalid when
            Gemini reports missing text regions.
        scoring_qa_dense_dump_prob: Probability of generating a single-turn dense
            dump instead of multi-turn QA for complete-OCR images.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    pipeline = Pipeline(name="ocr-nemotron")

    file_paths = _normalize_input_paths(input_path)
    pipeline.add_stage(
        FilePartitioningStage(
            file_paths=file_paths,
            file_extensions=[".jsonl"],
            files_per_partition=1,
        )
    )

    _tar_base = tar_base_path
    if _tar_base is None and not isinstance(input_path, (list, tuple)):
        _tar_base = Path(input_path).parent
    pipeline.add_stage(
        JsonlTarImageReaderStage(
            tar_base_path=str(_tar_base or "."),
            verbose=verbose,
            task_type=OCRData,
        )
    )

    if resume:
        pipeline.add_stage(
            SkipProcessedStage(
                output_path=output_path,
                image_parent=image_parent,
            )
        )

    pipeline.add_stage(
        OCRNemotronV2Stage(
            model_dir=nemotron_model_dir,
            num_workers=num_workers,
        )
    )

    if run_scoring_qa:
        pipeline.add_stage(OCRScoringQAStage(
            model_id=scoring_qa_model_id,
            min_bbox_match=scoring_qa_min_bbox_match,
            max_text_errors=scoring_qa_max_text_errors,
            fail_on_missing_text=scoring_qa_fail_on_missing_text,
            dense_dump_prob=scoring_qa_dense_dump_prob,
        ))

    pipeline.add_stage(
        ResultWriterStage(
            output_path=str(output_path),
            valid_only=valid_only,
            image_parent=str(image_parent) if image_parent else None,
            single_file=False,
            append=resume,
        )
    )

    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR dense pipeline (NemotronOCR-v2 multilingual)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to a JSONL file or directory of JSONL files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--tar-base-path",
        type=str,
        default=None,
        help="Base directory for tar shards (defaults to parent of --input-path)",
    )
    parser.add_argument(
        "--image-parent",
        type=str,
        default=None,
        help="Make image paths in output relative to this directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose reader logging",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip images already present in the output file",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        default=False,
        help="Exclude invalid records from the output",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of Xenna workers for model stages (default: autoscaler decides)",
    )
    parser.add_argument(
        "--nemotron-model-dir",
        type=str,
        default=None,
        help=(
            "Path to the NemotronOCR-v2 model directory. "
            "If not set, downloads from HuggingFace (nvidia/nemotron-ocr-v2)."
        ),
    )
    parser.add_argument(
        "--run-scoring-qa",
        action="store_true",
        default=False,
        help=(
            "After OCR, run Gemini bbox scoring + QA generation. "
            "Scores every bbox (bbox_match 0–10, text_errors), filters low-quality "
            "bboxes, and generates multi-turn QA or dense-dump conversations. "
            "Requires NVINFERENCE_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--scoring-qa-model-id",
        type=str,
        default="gcp/google/gemini-3-flash-preview",
        help="NVIDIA Inference API model ID for scoring QA stage.",
    )
    parser.add_argument(
        "--scoring-qa-min-bbox-match",
        type=int,
        default=7,
        help="Minimum bbox_match score (0-10) for a bbox to be considered valid.",
    )
    parser.add_argument(
        "--scoring-qa-max-text-errors",
        type=int,
        default=0,
        help="Maximum text_errors count for a bbox to be considered valid.",
    )
    parser.add_argument(
        "--scoring-qa-fail-on-missing-text",
        action="store_true",
        default=False,
        help="Mark the whole image invalid when Gemini reports missing text regions.",
    )
    parser.add_argument(
        "--scoring-qa-dense-dump-prob",
        type=float,
        default=0.10,
        help="Probability of dense dump conversation for complete-OCR images (default: 0.10).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help=(
            "Directory used by start_prometheus_grafana.py. "
            "If Prometheus/Grafana are running here, Ray metrics will be wired in automatically. "
            "Defaults to the per-user temp directory."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Human-readable label for this run. Sets SLURM_JOB_NAME, which Xenna attaches as "
            "the xenna_job_name label on the ray_pipeline_input_tasks metric in Prometheus. "
            "Note: the Grafana session dropdown shows Ray's auto-generated SessionName "
            "(session_YYYY-MM-DD_HH-MM-SS_PID); this flag does not change that."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = create_ocr_pipeline(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        tar_base_path=Path(args.tar_base_path) if args.tar_base_path else None,
        image_parent=Path(args.image_parent) if args.image_parent else None,
        verbose=args.verbose,
        resume=args.resume,
        valid_only=args.valid_only,
        num_workers=args.num_workers,
        nemotron_model_dir=Path(args.nemotron_model_dir) if args.nemotron_model_dir else None,
        run_scoring_qa=args.run_scoring_qa,
        scoring_qa_model_id=args.scoring_qa_model_id,
        scoring_qa_min_bbox_match=args.scoring_qa_min_bbox_match,
        scoring_qa_max_text_errors=args.scoring_qa_max_text_errors,
        scoring_qa_fail_on_missing_text=args.scoring_qa_fail_on_missing_text,
        scoring_qa_dense_dump_prob=args.scoring_qa_dense_dump_prob,
    )

    logger.info("\n" + pipeline.describe())

    if args.run_name:
        os.environ["SLURM_JOB_NAME"] = args.run_name

    client = RayClient(metrics_dir=args.metrics_dir)
    client.start()

    executor = XennaExecutor()

    logger.info("Starting OCR pipeline...")
    start = time.perf_counter()
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - start

    client.stop()

    logger.info(f"Pipeline completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Tasks processed: {len(output_tasks)}")

    merged = merge_output_shards(Path(args.output_path))
    logger.info(f"Output: {merged}")


if __name__ == "__main__":
    main()
