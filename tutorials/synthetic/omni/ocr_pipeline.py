"""OCR mixed dense pipeline entry point.

Stages:
  1. routing        Language detection → labels each image "qwen" / "rtx" / "skip"
  2. ocr_qwen       Qwen3-VL word-level OCR → populates ocr_qwen_dense for "qwen"-routed images
  3. ocr_nemotron   NemotronOCR-v2 word-level OCR → populates ocr_qwen_dense for "rtx"-routed images
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
from nemo_curator.stages.synthetic.omni.ocr_language_route import OCRLanguageRoutingStage
from nemo_curator.stages.synthetic.omni.ocr_nemotron_v2 import OCRNemotronV2Stage
from nemo_curator.stages.synthetic.omni.ocr_qwen import OCRQwenStage
from nemo_curator.stages.synthetic.omni.ocr_verification import OCRVerificationStage
from nemo_curator.tasks.ocr import OCRData


def _normalize_input_paths(
    input_path: Path | str | list[Path] | list[str],
) -> str | list[str]:
    if isinstance(input_path, (list, tuple)):
        return [str(p) for p in input_path]
    return str(input_path)


def create_routing_pipeline(
    input_path: Path | str | list[Path] | list[str],
    output_path: Path,
    tar_base_path: Path | None = None,
    image_parent: Path | None = None,
    verbose: bool = False,
    resume: bool = False,
    valid_only: bool = False,
    num_workers: int | None = None,
    model_id: str | None = None,
    run_ocr: bool = False,
    run_nemotron: bool = False,
    nemotron_model_dir: Path | None = None,
    run_verification: bool = False,
    verification_model_id: str = "gcp/google/gemini-3-pro",
) -> Pipeline:
    """Create the OCR pipeline (routing + optional Qwen/NemotronOCR stages).

    Reads images from JSONL+tar shards, runs Qwen3-VL language detection on each,
    and writes a JSONL with ``ocr_language_route`` set to "qwen", "rtx", or "skip".
    When ``run_ocr=True``, also runs Qwen3-VL word-level OCR on "qwen"-routed images
    and populates ``ocr_qwen_dense``.
    When ``run_nemotron=True``, also runs NemotronOCR-v2 on "rtx"-routed images and
    populates ``ocr_qwen_dense`` (same field, same schema).

    Note: ``valid_only=False`` by default so "rtx" images (which are valid but not
    yet processable) are preserved in the output for future stages.

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
        valid_only: If True, exclude invalid ("skip") records from the output.
            Default False so "rtx" images are kept for later stages.
        num_workers: Override the number of Xenna workers for all model stages.
            If None, the Xenna autoscaler decides.
        model_id: HuggingFace model ID for both routing and Qwen OCR stages.
            Defaults to Qwen3-VL-32B.
        run_ocr: If True, add OCRQwenStage after routing to run word-level OCR
            on "qwen"-routed images.
        run_nemotron: If True, add OCRNemotronV2Stage to run word-level OCR on
            "rtx"-routed images.  Requires ``nemotron-ocr`` to be installed.
        nemotron_model_dir: Path to the NemotronOCR-v2 model directory
            (e.g. ``<repo>/v2_english``).  If None, downloads from HuggingFace.
        run_verification: If True, add OCRVerificationStage after OCR stages to
            validate bounding boxes using Gemini 3 Pro.  Requires
            ``NVINFERENCE_API_KEY`` to be set.
        verification_model_id: NVIDIA Inference API model ID for verification.
            Defaults to ``gcp/google/gemini-3-pro``.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    if run_ocr and run_nemotron:
        name = "ocr-full"
    elif run_ocr:
        name = "ocr-routing-qwen"
    elif run_nemotron:
        name = "ocr-routing-nemotron"
    else:
        name = "ocr-routing"
    pipeline = Pipeline(name=name)

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

    model_kwargs: dict = {}
    if model_id is not None:
        model_kwargs["model_id"] = model_id
    pipeline.add_stage(OCRLanguageRoutingStage(num_workers=num_workers, **model_kwargs))

    if run_ocr:
        pipeline.add_stage(OCRQwenStage(num_workers=num_workers, **model_kwargs))

    if run_nemotron:
        pipeline.add_stage(
            OCRNemotronV2Stage(
                model_dir=nemotron_model_dir,
                num_workers=num_workers,
            )
        )

    if run_verification:
        pipeline.add_stage(OCRVerificationStage(model_id=verification_model_id))

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
        description="OCR mixed dense pipeline",
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
        help=(
            "Exclude invalid (skip) records from the output. "
            "By default, 'rtx'-routed images are kept even though no RTX stage runs yet."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of Xenna workers for the routing stage (default: autoscaler decides)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace model ID for the routing and OCR stages (default: Qwen/Qwen3-VL-32B-Instruct)",
    )
    parser.add_argument(
        "--run-ocr",
        action="store_true",
        default=False,
        help=(
            "After routing, run Qwen3-VL word-level OCR on 'qwen'-routed images. "
            "Populates ocr_qwen_dense in the output."
        ),
    )
    parser.add_argument(
        "--run-nemotron",
        action="store_true",
        default=False,
        help=(
            "After routing, run NemotronOCR-v2 word-level OCR on 'rtx'-routed images. "
            "Populates ocr_qwen_dense in the output. Requires nemotron-ocr to be installed."
        ),
    )
    parser.add_argument(
        "--nemotron-model-dir",
        type=str,
        default=None,
        help=(
            "Path to the NemotronOCR-v2 model directory (e.g. <repo>/v2_english). "
            "If not set, downloads from HuggingFace (nvidia/nemotron-ocr-v2)."
        ),
    )
    parser.add_argument(
        "--run-verification",
        action="store_true",
        default=False,
        help=(
            "After OCR stages, run Gemini 3 Pro verification to validate bounding boxes. "
            "Requires NVINFERENCE_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--verification-model-id",
        type=str,
        default="gcp/google/gemini-3-pro",
        help="NVIDIA Inference API model ID for verification (default: gcp/google/gemini-3-pro)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = create_routing_pipeline(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        tar_base_path=Path(args.tar_base_path) if args.tar_base_path else None,
        image_parent=Path(args.image_parent) if args.image_parent else None,
        verbose=args.verbose,
        resume=args.resume,
        valid_only=args.valid_only,
        num_workers=args.num_workers,
        model_id=args.model_id,
        run_ocr=args.run_ocr,
        run_nemotron=args.run_nemotron,
        nemotron_model_dir=Path(args.nemotron_model_dir) if args.nemotron_model_dir else None,
        run_verification=args.run_verification,
        verification_model_id=args.verification_model_id,
    )

    logger.info("\n" + pipeline.describe())

    executor = XennaExecutor()

    logger.info("Starting OCR routing pipeline...")
    start = time.perf_counter()
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - start

    logger.info(f"Pipeline completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Tasks processed: {len(output_tasks)}")


if __name__ == "__main__":
    main()
