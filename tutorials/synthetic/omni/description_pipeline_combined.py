import time
from pathlib import Path

import click
from loguru import logger
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.synthetic.omni.description import ImageCaptioningData, DescriptionStage
from nemo_curator.stages.synthetic.omni.description_output import DescriptionOutputStage
from nemo_curator.stages.synthetic.omni.description_validator import DescriptionValidatorStage, DescriptionValidatedData
from nemo_curator.stages.synthetic.omni.io import JsonlTarImageReaderStage, ResultWriterStage, SkipProcessedStage

import argparse
from nemo_curator.core.client import RayClient

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
    validator_model_id: str = "gcp/google/gemini-3-flash-preview",
) -> Pipeline:
    """Create the combined description pipeline (description → validation → output → write).

    Args:
        input_path: Path to a JSONL file, a directory containing JSONL files,
            or a list of JSONL file paths.
        output_path: Path to output JSONL file.
        tar_base_path: Base path for tar shards.
        image_parent: Parent directory to make image paths relative.
        verbose: Enable verbose logging.

    Returns:
        Configured NeMo Curator Pipeline.

    Note:
        Xenna streaming mode does allow overlap (Validation can run while Description
        is still producing). Overlap is improved when Validation's batch_size is <=
        Description's (e.g. 16); otherwise the next stage waits for a full batch
        and stays idle (see DescriptionValidatorStage.batch_size).
        With the raydata backend, the executor repartitions before actor stages so
        blocks are smaller and downstream stages (e.g. Validator) receive output
        incrementally and can overlap with Description.
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

    pipeline.add_stage(DescriptionValidatorStage(model_id=validator_model_id))

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
    parser.add_argument(
        "--description-num-workers",
        type=int,
        default=None,
        help="Number of workers for description stage (e.g. 8 for 8 GPUs). If unset, Xenna autoscaler decides.",
    )
    parser.add_argument(
        "--validator-model-id",
        type=str,
        default="gcp/google/gemini-3-flash-preview",
        help="NVIDIA Inference API model ID for the validator stage.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["xenna", "raydata"],
        default="xenna",
        help="Execution backend: 'xenna' (default) or 'raydata' (Ray Data, experimental).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        help="Directory for Prometheus/Grafana metrics. Start Prometheus first with the same path: "
        "python -m nemo_curator.metrics.start_prometheus_grafana --yes --metrics_dir <path>",
    )
    parser.add_argument(
        "--dashboard-host",
        type=str,
        default="127.0.0.1",
        help="Ray dashboard bind address. Use 0.0.0.0 when accessing via SSH tunnel from another machine (e.g. SLURM).",
    )
    return parser.parse_args()

def main() -> None:
    """Main function to run the description generation pipeline."""
    args = parse_args()

    if args.metrics_dir is not None:
        client = RayClient(metrics_dir=args.metrics_dir, ray_dashboard_host=args.dashboard_host)
        client.start()
    else:
        client = None

    pipeline = create_description_pipeline(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        tar_base_path=Path(args.tar_base_path) if args.tar_base_path else None,
        image_parent=Path(args.image_parent) if args.image_parent else None,
        verbose=args.verbose,
        resume=args.resume,
        valid_only=args.valid_only,
        description_num_workers=args.description_num_workers,
        validator_model_id=args.validator_model_id,
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")
    # Create executor
    if args.backend == "raydata":
        executor = RayDataExecutor()
    else:
        executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    start = time.perf_counter()
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - start
    print(f"\nPipeline completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Tasks processed: {len(output_tasks)}")
    
    if client is not None:
        client.stop()


if __name__ == "__main__":
    main()
