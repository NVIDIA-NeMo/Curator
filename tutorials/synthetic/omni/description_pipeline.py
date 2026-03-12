import time
from pathlib import Path

import click
from loguru import logger
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.synthetic.omni.description import ImageCaptioningData, DescriptionStage
from nemo_curator.stages.synthetic.omni.description_output import DescriptionOutputStage
from nemo_curator.stages.synthetic.omni.description_validator import DescriptionValidatorStage, DescriptionValidatedData
from nemo_curator.stages.synthetic.omni.io import (
    JsonlPipelineOutputReaderStage,
    JsonlTarImageReaderStage,
    ResultWriterStage,
    SkipProcessedStage,
)

import argparse

def _normalize_input_paths(
    input_path: Path | str | list[Path] | list[str],
) -> str | list[str]:
    """Normalize input_path to str or list[str] for FilePartitioningStage."""
    if isinstance(input_path, (list, tuple)):
        return [str(p) for p in input_path]
    return str(input_path)


def create_description_pipeline(
    input_path: Path | str | list[Path] | list[str],
    output_path: Path,
    tar_base_path: Path | None = None,
    image_parent: Path | None = None,
    verbose: bool = False,
    resume: bool = False,
    valid_only: bool = True,
    description_num_workers: int | None = None,
) -> Pipeline:
    """Create the description generation pipeline (Part 1).

    Args:
        input_path: Path to a JSONL file, a directory containing JSONL files,
            or a list of JSONL file paths.
        output_path: Path to output JSONL file.
        tar_base_path: Base path for tar shards.
        image_parent: Parent directory to make image paths relative.
        verbose: Enable verbose logging.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    pipeline = Pipeline(name="description-gen")

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
            task_type=ImageCaptioningData,
        )
    )

    if resume:
        pipeline.add_stage(SkipProcessedStage(
            output_path=output_path,
            image_parent=image_parent,
        ))

    pipeline.add_stage(DescriptionStage(num_workers=description_num_workers))

    pipeline.add_stage(ResultWriterStage(
        output_path=str(output_path),
        valid_only=valid_only,
        image_parent=str(image_parent) if image_parent else None,
        single_file=False,
        append=resume,
    ))

    return pipeline


def create_validation_pipeline(
    input_path: Path | str | list[Path] | list[str],
    output_path: Path,
    image_parent: Path | None = None,
    verbose: bool = False,
    resume: bool = False,
    valid_only: bool = True,
) -> Pipeline:
    """Create the validation pipeline (Part 2).

    Args:
        input_path: Path to a JSONL file from Part 1, a directory containing
            such JSONL files, or a list of JSONL file paths.
        output_path: Path to output JSONL file.
        image_parent: Parent directory to make image paths relative.
        verbose: Enable verbose logging.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    pipeline = Pipeline(name="description-val")

    file_paths = _normalize_input_paths(input_path)
    pipeline.add_stage(
        FilePartitioningStage(
            file_paths=file_paths,
            file_extensions=[".jsonl"],
            files_per_partition=1,
        )
    )

    pipeline.add_stage(
        JsonlPipelineOutputReaderStage(
            verbose=verbose,
            task_type=DescriptionValidatedData,
        )
    )

    if resume:
        pipeline.add_stage(SkipProcessedStage(
            output_path=output_path,
            image_parent=image_parent,
        ))

    pipeline.add_stage(DescriptionValidatorStage())

    pipeline.add_stage(ResultWriterStage(
        output_path=str(output_path),
        valid_only=valid_only,
        image_parent=str(image_parent) if image_parent else None,
        single_file=False,
        append=resume,
    ))

    return pipeline

def create_output_pipeline(
    input_path: Path,
    output_path: Path,
    image_parent: Path | None = None,
    verbose: bool = False,
    resume: bool = False,
    valid_only: bool = True,
) -> Pipeline:
    """Create the output pipeline: validated descriptions -> conversation format.

    Args:
        input_path: Path to JSONL file with validated descriptions.
        output_path: Path to output JSONL file.
        image_parent: Parent directory to make image paths relative.
        verbose: Enable verbose logging.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    pipeline = Pipeline(name="description-output")

    pipeline.add_stage(JsonlPipelineOutputReaderStage(
        jsonl_path=input_path,
        verbose=verbose,
        task_type=DescriptionValidatedData,
    ))

    if resume:
        pipeline.add_stage(SkipProcessedStage(
            output_path=output_path,
            image_parent=image_parent,
        ))

    pipeline.add_stage(DescriptionOutputStage())

    pipeline.add_stage(ResultWriterStage(
        output_path=str(output_path),
        valid_only=valid_only,
        image_parent=str(image_parent) if image_parent else None,
        single_file=False,
        append=resume,
    ))

    return pipeline

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic multilingual Q&A data using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to a JSONL file, or a directory containing JSONL files",
    )
    parser.add_argument("--output-path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--tar-base-path", type=str, help="Base path for tar shards")
    parser.add_argument("--image-parent", type=str, help="Parent directory to make image paths relative")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose logging")
    parser.add_argument("--resume", type=bool, default=False, help="Resume from last processed image")
    parser.add_argument("--valid-only", type=bool, default=True, help="Only process valid images")
    parser.add_argument("--run_validation", action="store_true", help="Run validation pipeline")
    parser.add_argument("--run_output", action="store_true", help="Run output pipeline")
    parser.add_argument(
        "--description-num-workers",
        type=int,
        default=None,
        help="Number of workers for description stage (e.g. 8 for 8 GPUs). If unset, Xenna autoscaler decides.",
    )
    return parser.parse_args()

def main() -> None:
    """Main function to run the description generation pipeline."""
    args = parse_args()

    if args.run_validation:
        pipeline = create_validation_pipeline(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            image_parent=Path(args.image_parent) if args.image_parent else None,
            verbose=args.verbose,
            resume=args.resume,
            valid_only=args.valid_only,
        )
    elif args.run_output:
        pipeline = create_output_pipeline(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            image_parent=Path(args.image_parent) if args.image_parent else None,
            verbose=args.verbose,
            resume=args.resume,
            valid_only=args.valid_only,
        )
    else:
        pipeline = create_description_pipeline(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            tar_base_path=Path(args.tar_base_path) if args.tar_base_path else None,
            image_parent=Path(args.image_parent) if args.image_parent else None,
            verbose=args.verbose,
            resume=args.resume,
            valid_only=args.valid_only,
            description_num_workers=args.description_num_workers,
        )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    start = time.perf_counter()
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - start
    print(f"\nPipeline completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Tasks processed: {len(output_tasks)}")


if __name__ == "__main__":
    main()
