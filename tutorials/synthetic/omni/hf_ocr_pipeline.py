"""OCR dense pipeline for HuggingFace datasets.

Reads images from a HuggingFace dataset (Hub or local), runs NemotronOCR-v2
word-level OCR on every image, and optionally generates multi-turn QA
conversations via Gemini bbox scoring.

Stages:
  1. hf_dataset_image_reader   Extract images from HF dataset to local dir
  2. ocr_nemotron_v2           NemotronOCR-v2 word-level OCR
  3. ocr_scoring_qa            Gemini bbox scoring + QA generation (optional)
  4. result_writer             Write output JSONL
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.omni.io import (
    HFDatasetImageReaderStage,
    ResultWriterStage,
    SkipProcessedStage,
    merge_output_shards,
)
from nemo_curator.stages.synthetic.omni.ocr_nemotron_v2 import OCRNemotronV2Stage
from nemo_curator.stages.synthetic.omni.ocr_scoring_qa import OCRScoringQAStage
from nemo_curator.tasks.ocr import OCRData


def create_hf_ocr_pipeline(
    dataset_name: str,
    image_dir: Path,
    output_path: Path,
    hf_split: str = "train",
    hf_config: str | None = None,
    hf_image_column: str = "image",
    hf_id_column: str | None = None,
    hf_limit: int | None = None,
    image_parent: Path | None = None,
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
    """Create the OCR pipeline reading images from a HuggingFace dataset.

    Args:
        dataset_name: HF Hub dataset id (e.g. ``"textvqa"``) or a local path.
            Local paths are detected automatically:
            - Directory with ``dataset_info.json`` → ``load_from_disk()``.
            - Any other directory → ``imagefolder`` loader.
        image_dir: Directory where extracted JPEG images are cached.
            Already-present images are reused across runs (idempotent).
        output_path: Output directory for result JSONL shards.
        hf_split: Dataset split to load (default: ``"train"``).
        hf_config: Optional HF dataset configuration / subset name.
        hf_image_column: Column holding the PIL image (default: ``"image"``).
        hf_id_column: Column used as image_id; falls back to row index if None.
            When multiple rows share the same id (VQA-style), only the first
            occurrence is processed.
        hf_limit: Cap the number of images to download and process.
            For Hub datasets the limit is embedded in the split-slice string
            (e.g. ``"train[:1000]"``) so only those records are fetched.
        image_parent: If set, image paths in the output are made relative to
            this directory.
        resume: Skip images already written to ``output_path``.
        valid_only: Exclude invalid records from the output.
        num_workers: Xenna worker count override for model stages.
        nemotron_model_dir: NemotronOCR-v2 model directory; downloads from HF
            Hub (``nvidia/nemotron-ocr-v2``) if None.
        run_scoring_qa: Run Gemini bbox scoring + QA generation after OCR.
        scoring_qa_model_id: NVIDIA Inference API model ID.
        scoring_qa_min_bbox_match: Minimum bbox_match (0–10) to keep a bbox.
        scoring_qa_max_text_errors: Maximum text_errors count to keep a bbox.
        scoring_qa_fail_on_missing_text: Mark image invalid when Gemini reports
            missing text regions.
        scoring_qa_dense_dump_prob: Probability of dense-dump vs multi-turn QA.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    pipeline = Pipeline(name="ocr-nemotron-hf")

    pipeline.add_stage(
        HFDatasetImageReaderStage(
            dataset_name=dataset_name,
            image_dir=image_dir,
            split=hf_split,
            config_name=hf_config,
            image_column=hf_image_column,
            id_column=hf_id_column,
            limit=hf_limit,
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
        pipeline.add_stage(
            OCRScoringQAStage(
                model_id=scoring_qa_model_id,
                min_bbox_match=scoring_qa_min_bbox_match,
                max_text_errors=scoring_qa_max_text_errors,
                fail_on_missing_text=scoring_qa_fail_on_missing_text,
                dense_dump_prob=scoring_qa_dense_dump_prob,
            )
        )

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
        description="OCR pipeline for HuggingFace datasets (NemotronOCR-v2 + optional Gemini QA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- HF dataset source ---
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset id (e.g. 'textvqa') or local path",
    )
    parser.add_argument(
        "--hf-image-dir",
        type=str,
        required=True,
        help="Directory where extracted JPEG images are cached",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="Dataset configuration / subset name",
    )
    parser.add_argument(
        "--hf-image-column",
        type=str,
        default="image",
        help="Column holding the PIL image",
    )
    parser.add_argument(
        "--hf-id-column",
        type=str,
        default=None,
        help="Column to use as image_id (falls back to row index if not set)",
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=None,
        help="Cap the number of images downloaded/processed",
    )

    # --- Output ---
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for result JSONL shards",
    )
    parser.add_argument(
        "--image-parent",
        type=str,
        default=None,
        help="Make image paths in output relative to this directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip images already present in the output directory",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        default=False,
        help="Exclude invalid records from the output",
    )

    # --- Model ---
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Xenna worker count for model stages (default: autoscaler decides)",
    )
    parser.add_argument(
        "--nemotron-model-dir",
        type=str,
        default=None,
        help="NemotronOCR-v2 model directory (downloads from HF if not set)",
    )

    # --- Scoring QA ---
    parser.add_argument(
        "--run-scoring-qa",
        action="store_true",
        default=False,
        help=(
            "Run Gemini bbox scoring + QA generation after OCR. "
            "Requires NVINFERENCE_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--scoring-qa-model-id",
        type=str,
        default="gcp/google/gemini-3-flash-preview",
        help="NVIDIA Inference API model ID for scoring QA",
    )
    parser.add_argument(
        "--scoring-qa-min-bbox-match",
        type=int,
        default=7,
        help="Minimum bbox_match score (0-10) to keep a bbox",
    )
    parser.add_argument(
        "--scoring-qa-max-text-errors",
        type=int,
        default=0,
        help="Maximum text_errors count to keep a bbox",
    )
    parser.add_argument(
        "--scoring-qa-fail-on-missing-text",
        action="store_true",
        default=False,
        help="Mark image invalid when Gemini reports missing text regions",
    )
    parser.add_argument(
        "--scoring-qa-dense-dump-prob",
        type=float,
        default=0.10,
        help="Probability of dense-dump conversation for complete-OCR images",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = create_hf_ocr_pipeline(
        dataset_name=args.hf_dataset,
        image_dir=Path(args.hf_image_dir),
        output_path=Path(args.output_path),
        hf_split=args.hf_split,
        hf_config=args.hf_config,
        hf_image_column=args.hf_image_column,
        hf_id_column=args.hf_id_column,
        hf_limit=args.hf_limit,
        image_parent=Path(args.image_parent) if args.image_parent else None,
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

    executor = XennaExecutor()

    logger.info("Starting HF OCR pipeline...")
    start = time.perf_counter()
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - start

    logger.info(f"Pipeline completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Tasks processed: {len(output_tasks)}")

    merged = merge_output_shards(Path(args.output_path))
    logger.info(f"Output: {merged}")


if __name__ == "__main__":
    main()
