from pathlib import Path

import click
from loguru import logger
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline

from nemo_curator.stages.synthetic.omni.description import ImageCaptioningData, DescriptionStage
from nemo_curator.stages.synthetic.omni.io import JsonlTarImageReaderStage, ResultWriterStage, SkipProcessedStage
import argparse

def create_description_pipeline(
    input_path: Path,
    output_path: Path,
    tar_base_path: Path | None = None,
    image_parent: Path | None = None,
    verbose: bool = False,
    resume: bool = False,
    valid_only: bool = True,
) -> Pipeline:
    """Create the description generation pipeline (Part 1).

    Args:
        input_path: Path to JSONL file with tar image references.
        output_path: Path to output JSONL file.
        tar_base_path: Base path for tar shards.
        image_parent: Parent directory to make image paths relative.
        verbose: Enable verbose logging.

    Returns:
        Configured NeMo Curator Pipeline.
    """
    pipeline = Pipeline(name="description-gen")

    pipeline.add_stage(JsonlTarImageReaderStage(
        jsonl_path=str(input_path),
        tar_base_path=str(tar_base_path or input_path.parent),
        verbose=verbose,
        task_type=ImageCaptioningData,
    ))

    if resume:
        pipeline.add_stage(SkipProcessedStage(
            output_path=output_path,
            image_parent=image_parent,
        ))

    pipeline.add_stage(DescriptionStage(cuda_devices=[0]))

    pipeline.add_stage(ResultWriterStage(
        output_path=str(output_path),
        valid_only=valid_only,
        image_parent=str(image_parent) if image_parent else None,
        single_file=True,
        append=resume,
    ))

    return pipeline

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic multilingual Q&A data using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-path", type=str, required=True, help="Path to JSONL file with tar image references")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--tar-base-path", type=str, help="Base path for tar shards")
    parser.add_argument("--image-parent", type=str, help="Parent directory to make image paths relative")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose logging")
    parser.add_argument("--resume", type=bool, default=False, help="Resume from last processed image")
    parser.add_argument("--valid-only", type=bool, default=True, help="Only process valid images")
    return parser.parse_args()

def main() -> None:
    """Main function to run the description generation pipeline."""
    args = parse_args()
    pipeline = create_description_pipeline(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        tar_base_path=Path(args.tar_base_path) if args.tar_base_path else None,
        image_parent=Path(args.image_parent) if args.image_parent else None,
        verbose=args.verbose,
        resume=args.resume,
        valid_only=args.valid_only,
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
# Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    pipeline.run(executor)

    # Print results
    print("\nPipeline completed!")


