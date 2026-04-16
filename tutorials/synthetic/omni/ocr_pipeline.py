"""OCR dense pipeline entry point.

Stages:
  1. ocr_nemotron   NemotronOCR-v2 (multilingual) word-level OCR → populates ocr_dense
  2. verification   Gemini API bounding-box verification (optional)
  3. scoring_qa     Combined Gemini bbox scoring + multi-turn QA generation (optional)
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
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.synthetic.omni.io import (
    JsonlPipelineOutputReaderStage,
    JsonlTarImageReaderStage,
    ResultWriterStage,
    SkipProcessedStage,
)
from nemo_curator.stages.synthetic.omni.ocr_conversationalize import OCRConversationalizeStage
from nemo_curator.stages.synthetic.omni.ocr_dense_qa import OCRDenseQAStage
from nemo_curator.stages.synthetic.omni.ocr_nemotron_v2 import OCRNemotronV2Stage
from nemo_curator.stages.synthetic.omni.ocr_verification import OCRVerificationStage
from nemo_curator.stages.synthetic.omni.ocr_scoring_verification import OCRScoringVerificationStage
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
    run_verification: bool = False,
    verification_model_id: str = "gcp/google/gemini-3-flash-preview",
    run_scoring_verification: bool = False,
    scoring_verification_model_id: str = "gcp/google/gemini-3-flash-preview",
    scoring_min_bbox_match: int = 7,
    scoring_max_text_errors: int = 0,
    scoring_fail_on_missing_text: bool = True,
    run_conversationalize: bool = False,
    run_dense_qa: bool = False,
    run_scoring_qa: bool = False,
    scoring_qa_model_id: str = "gcp/google/gemini-3-flash-preview",
    scoring_qa_min_bbox_match: int = 7,
    scoring_qa_max_text_errors: int = 0,
    scoring_qa_fail_on_missing_text: bool = False,
) -> Pipeline:
    """Create the OCR pipeline using NemotronOCR-v2 (multilingual) on every image.

    Reads images from JSONL+tar shards, runs NemotronOCR-v2 word-level OCR on
    every image, and writes a JSONL with ``ocr_dense`` populated.

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
        run_verification: If True, run Gemini verification on OCR output.
        run_scoring_verification: If True, run per-bbox Gemini scoring.
        run_conversationalize: If True, convert OCR output to SFT conversation format.
        run_dense_qa: If True, generate multi-turn QA conversations.
        run_scoring_qa: If True, run combined Gemini scoring + QA generation.

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
            process_all=True,
        )
    )

    if run_verification:
        pipeline.add_stage(OCRVerificationStage(model_id=verification_model_id))

    if run_scoring_verification:
        pipeline.add_stage(OCRScoringVerificationStage(
            model_id=scoring_verification_model_id,
            min_bbox_match=scoring_min_bbox_match,
            max_text_errors=scoring_max_text_errors,
            fail_on_missing_text=scoring_fail_on_missing_text,
        ))

    if run_conversationalize:
        pipeline.add_stage(OCRConversationalizeStage())

    if run_dense_qa:
        pipeline.add_stage(OCRDenseQAStage())

    if run_scoring_qa:
        pipeline.add_stage(OCRScoringQAStage(
            model_id=scoring_qa_model_id,
            min_bbox_match=scoring_qa_min_bbox_match,
            max_text_errors=scoring_qa_max_text_errors,
            fail_on_missing_text=scoring_qa_fail_on_missing_text,
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
        "--run-verification",
        action="store_true",
        default=False,
        help=(
            "After OCR, run Gemini 3 verification to validate bounding boxes. "
            "Requires NVINFERENCE_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--verification-model-id",
        type=str,
        default="gcp/google/gemini-3-flash-preview",
        help="NVIDIA Inference API model ID for verification",
    )
    parser.add_argument(
        "--run-scoring-verification",
        action="store_true",
        default=False,
        help=(
            "After OCR, run per-bbox scoring verification using Gemini. "
            "Assigns bbox_match (0-10) and text_errors per bbox."
        ),
    )
    parser.add_argument(
        "--scoring-verification-model-id",
        type=str,
        default="gcp/google/gemini-3-flash-preview",
        help="NVIDIA Inference API model ID for scoring verification.",
    )
    parser.add_argument(
        "--scoring-min-bbox-match",
        type=int,
        default=7,
        help="Minimum bbox_match score (0-10) for a bbox to be considered valid.",
    )
    parser.add_argument(
        "--scoring-max-text-errors",
        type=int,
        default=0,
        help="Maximum text_errors count for a bbox to be considered valid.",
    )
    parser.add_argument(
        "--scoring-no-fail-on-missing-text",
        action="store_true",
        default=False,
        help="Do not mark the image invalid when Gemini reports missing text regions.",
    )
    parser.add_argument(
        "--run-conversationalize",
        action="store_true",
        default=False,
        help="Convert word-level OCR output into SFT conversation format.",
    )
    parser.add_argument(
        "--run-dense-qa",
        action="store_true",
        default=False,
        help=(
            "Generate multi-turn QA conversations (up to 100 pairs per image). "
            "Mutually exclusive with --run-conversationalize."
        ),
    )
    parser.add_argument(
        "--run-scoring-qa",
        action="store_true",
        default=False,
        help=(
            "Combined Gemini bbox scoring + multi-turn QA generation in one stage. "
            "Replaces --run-scoring-verification + --run-dense-qa."
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
        run_verification=args.run_verification,
        verification_model_id=args.verification_model_id,
        run_scoring_verification=args.run_scoring_verification,
        scoring_verification_model_id=args.scoring_verification_model_id,
        scoring_min_bbox_match=args.scoring_min_bbox_match,
        scoring_max_text_errors=args.scoring_max_text_errors,
        scoring_fail_on_missing_text=not args.scoring_no_fail_on_missing_text,
        run_conversationalize=args.run_conversationalize,
        run_dense_qa=args.run_dense_qa,
        run_scoring_qa=args.run_scoring_qa,
        scoring_qa_model_id=args.scoring_qa_model_id,
        scoring_qa_min_bbox_match=args.scoring_qa_min_bbox_match,
        scoring_qa_max_text_errors=args.scoring_qa_max_text_errors,
        scoring_qa_fail_on_missing_text=args.scoring_qa_fail_on_missing_text,
    )

    logger.info("\n" + pipeline.describe())

    executor = XennaExecutor()

    logger.info("Starting OCR pipeline...")
    start = time.perf_counter()
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - start

    logger.info(f"Pipeline completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Tasks processed: {len(output_tasks)}")


if __name__ == "__main__":
    main()
