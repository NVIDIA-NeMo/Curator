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

"""Slurm array tutorial: split a large file collection across array tasks.

Each Slurm array task processes its own slice of the input files.
With 2000 JSONL or Parquet files and --array=0-19, each of the 20 jobs gets
~100 files.

Array partitioning parameters
------------------------------
shard_index          Which shard this job processes.
                     Default: SLURM_ARRAY_TASK_ID env var.
total_shards         Total number of shards (i.e. array width).
                     Default: SLURM_ARRAY_TASK_COUNT env var.
minimum_shard_index  Offset added to the hash-assigned shard before
                     comparing with shard_index. Use when the array does
                     not start at 0. E.g. --array=1-20 requires
                     minimum_shard_index=1 so shard IDs 1-20 match task IDs 1-20.
                     Default: 0. No env var fallback — must be set explicitly.

Usage (local smoke test against a small sample directory)::

    # Simulate task 0 of 4 locally (zero-indexed array)
    SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=4 \\
        python tutorials/slurm/array_pipeline.py \\
            --input-dir /path/to/input/directory \\
            --output-dir /path/to/output/directory

    # Non-zero-indexed array: tasks 1-4, minimum_shard_index=1
    python tutorials/slurm/array_pipeline.py \\
        --input-dir /path/to/input/directory \\
        --output-dir /path/to/output/directory \\
        --shard-index 1 --total-shards 4 --minimum-shard-index 1

    # Use Parquet input/output instead of the default JSONL:
    python tutorials/slurm/array_pipeline.py \\
        --input-dir /path/to/input/directory \\
        --input-file-type parquet \\
        --output-dir /path/to/output/directory \\
        --output-file-type parquet \\
        --shard-index 0 --total-shards 4

    # Or let the sbatch script set the env vars:
    sbatch --array=0-19 tutorials/slurm/submit_array.sh
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import socket
import tempfile
import uuid
from pathlib import Path

from loguru import logger

from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


METADATA_DIRNAME = ".nemo_curator_metadata"
SLURM_ARRAY_RETRY_DIRNAME = ".slurm_array_retry"


def _safe_token(value: object) -> str:
    """Convert a value to a conservative filename token."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def _resolve_int_arg(value: int | None, env_var: str) -> int | None:
    """Resolve an optional CLI integer from an environment variable."""
    if value is not None:
        return value
    env_value = os.environ.get(env_var)
    return int(env_value) if env_value is not None else None


def _is_driver_process(use_slurm: bool) -> bool:
    """Return True for the process that should run the pipeline and own retry metadata."""
    return not use_slurm or os.environ.get("SLURM_NODEID", "0") == "0"


def _retry_manifest_prefix(
    shard_index: int | None,
    total_shards: int | None,
    minimum_shard_index: int,
) -> str:
    return (
        f"manifest_shard-{_safe_token(shard_index)}_"
        f"total-{_safe_token(total_shards)}_"
        f"min-{_safe_token(minimum_shard_index)}_"
    )


def _retry_manifest_payload(
    shard_index: int | None,
    total_shards: int | None,
    minimum_shard_index: int,
    status: str,
    error: BaseException | None = None,
) -> dict[str, object]:
    payload = {
        "shard_index": shard_index,
        "total_shards": total_shards,
        "minimum_shard_index": minimum_shard_index,
        "status": status,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_node_id": os.environ.get("SLURM_NODEID"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }

    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)

    return payload


def write_retry_manifest(
    checkpoint_path: str,
    shard_index: int | None,
    total_shards: int | None,
    minimum_shard_index: int,
    status: str,
    error: BaseException | None = None,
    manifest_file: Path | None = None,
) -> Path:
    """Write a retry manifest using a unique name and atomic rename."""
    retry_dir = Path(checkpoint_path, METADATA_DIRNAME, SLURM_ARRAY_RETRY_DIRNAME).absolute()
    retry_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.datetime.now(datetime.timezone.utc)
    manifest = _retry_manifest_payload(
        shard_index=shard_index,
        total_shards=total_shards,
        minimum_shard_index=minimum_shard_index,
        status=status,
        error=error,
    )

    if manifest_file is None:
        timestamp = created_at.strftime("%Y%m%d%H%M%S%f")
        manifest_file = retry_dir / (
            _retry_manifest_prefix(shard_index, total_shards, minimum_shard_index)
            + f"job-{_safe_token(os.environ.get('SLURM_JOB_ID', 'local'))}_"
            + f"task-{_safe_token(os.environ.get('SLURM_ARRAY_TASK_ID', 'local'))}_"
            + f"node-{_safe_token(os.environ.get('SLURM_NODEID', '0'))}_"
            + f"pid-{os.getpid()}_"
            + f"{timestamp}_"
            + f"{uuid.uuid4().hex}.json"
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=retry_dir,
            prefix=f".{manifest_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(manifest, tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, manifest_file)
    except Exception:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
        raise

    return manifest_file


def remove_retry_manifests(
    checkpoint_path: str,
    shard_index: int | None,
    total_shards: int | None,
    minimum_shard_index: int,
) -> None:
    """Remove retry manifests for a shard after successful completion."""
    retry_dir = Path(checkpoint_path, METADATA_DIRNAME, SLURM_ARRAY_RETRY_DIRNAME).absolute()
    if not retry_dir.exists():
        return

    pattern = _retry_manifest_prefix(shard_index, total_shards, minimum_shard_index) + "*.json"
    for manifest_file in retry_dir.glob(pattern):
        with contextlib.suppress(FileNotFoundError):
            manifest_file.unlink()


def build_pipeline(
    input_dir: str,
    input_file_type: str,
    output_dir: str,
    output_file_type: str,
    files_per_partition: int,
    shard_index: int | None,
    total_shards: int | None,
    minimum_shard_index: int,
) -> Pipeline:
    pipeline = Pipeline(
        name="slurm_array_demo",
        description=(
            "Read files from input directory assigned to this Slurm array task "
            "and write them out to output directory."
        ),
    )

    # enable_array_partitioning=True reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
    # from the environment by default. Explicit shard_index / total_shards / minimum_shard_index
    # override those env vars — useful for non-Slurm schedulers or local testing.
    if input_file_type == "jsonl":
        pipeline.add_stage(
            JsonlReader(
                file_paths=input_dir,
                files_per_partition=files_per_partition,
                enable_array_partitioning=True,
                shard_index=shard_index,
                total_shards=total_shards,
                minimum_shard_index=minimum_shard_index,
            )
        )
    elif input_file_type == "parquet":
        pipeline.add_stage(
            ParquetReader(
                file_paths=input_dir,
                files_per_partition=files_per_partition,
                enable_array_partitioning=True,
                shard_index=shard_index,
                total_shards=total_shards,
                minimum_shard_index=minimum_shard_index,
            )
        )
    else:
        raise ValueError(f"Unsupported input file type: {input_file_type}")

    if output_file_type == "jsonl":
        pipeline.add_stage(JsonlWriter(output_dir))
    elif output_file_type == "parquet":
        pipeline.add_stage(ParquetWriter(output_dir))
    else:
        raise ValueError(f"Unsupported output file type: {output_file_type}")

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Slurm array file-partitioning demo")
    parser.add_argument("--input-dir", required=True, help="Directory containing input files")
    parser.add_argument(
        "--input-file-type",
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Type of input files (default: jsonl)",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write output files")
    parser.add_argument(
        "--output-file-type",
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Type of output files (default: jsonl)",
    )
    parser.add_argument(
        "--files-per-partition",
        type=int,
        default=1,
        help="Files grouped into each FileGroupTask (default: 1)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard to process. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of shards. Defaults to SLURM_ARRAY_TASK_COUNT.",
    )
    parser.add_argument(
        "--minimum-shard-index",
        type=int,
        default=0,
        help=(
            "Offset added to the hash-assigned shard before comparison. "
            "Set to match the first task ID when the array does not start at 0 "
            "(e.g. --array=1-20 requires --minimum-shard-index=1). Default: 0."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path for checkpoint metadata. Slurm array retry manifests are written under "
            "<checkpoint_path>/.nemo_curator_metadata/.slurm_array_retry/. Defaults to None."
        ),
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Use SlurmRayClient for multi-node srun jobs.",
    )
    args = parser.parse_args()

    ray_client = SlurmRayClient() if args.slurm else RayClient()
    shard_index = args.shard_index
    total_shards = args.total_shards
    minimum_shard_index = args.minimum_shard_index
    retry_manifest_file = None
    is_driver_process = _is_driver_process(args.slurm)
    should_manage_retry_manifest = args.checkpoint_path is not None and is_driver_process

    try:
        shard_index = _resolve_int_arg(args.shard_index, "SLURM_ARRAY_TASK_ID")
        total_shards = _resolve_int_arg(args.total_shards, "SLURM_ARRAY_TASK_COUNT")

        if should_manage_retry_manifest:
            retry_manifest_file = write_retry_manifest(
                checkpoint_path=args.checkpoint_path,
                shard_index=shard_index,
                total_shards=total_shards,
                minimum_shard_index=minimum_shard_index,
                status="pending",
            )
            logger.info(f"Wrote pending Slurm array retry manifest to {retry_manifest_file}")

        ray_client.start()

        pipeline = build_pipeline(
            args.input_dir,
            args.input_file_type,
            args.output_dir,
            args.output_file_type,
            args.files_per_partition,
            shard_index=shard_index,
            total_shards=total_shards,
            minimum_shard_index=minimum_shard_index,
        )
        logger.info(f"\n{pipeline.describe()}")

        pipeline.run()

        if should_manage_retry_manifest:
            try:
                remove_retry_manifests(
                    checkpoint_path=args.checkpoint_path,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    minimum_shard_index=minimum_shard_index,
                )
            except Exception as cleanup_error:
                logger.error(f"Pipeline succeeded but failed to remove retry manifest: {cleanup_error}")
    except Exception as e:
        if should_manage_retry_manifest:
            try:
                manifest_file = write_retry_manifest(
                    checkpoint_path=args.checkpoint_path,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    minimum_shard_index=minimum_shard_index,
                    status="failed",
                    error=e,
                    manifest_file=retry_manifest_file,
                )
                logger.error(f"Wrote Slurm array retry manifest to {manifest_file}")
            except Exception as manifest_error:
                logger.error(f"Failed to write Slurm array retry manifest: {manifest_error}")

        logger.error(f"Error running pipeline: {e}")
        raise
    finally:
        if is_driver_process:
            ray_client.stop()


if __name__ == "__main__":
    main()
