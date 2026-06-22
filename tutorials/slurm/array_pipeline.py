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

Each Slurm array task processes its own slice of the source tasks emitted by the reader.
With 2000 JSONL or Parquet files and --array=0-19, each of the 20 jobs gets
~100 files.

Array partitioning parameters
------------------------------
shard_index          Which shard this job processes.
total_shards         Total number of shards (i.e. array width).
minimum_shard_index  Offset added to the hash-assigned shard before
                     comparing with shard_index. Use when the array does
                     not start at 0. E.g. --array=1-20 requires
                     minimum_shard_index=1 so shard IDs 1-20 match task IDs 1-20.
                     Default: 0.

Usage (local smoke test against a small sample directory)::

    # Simulate task 0 of 4 locally (zero-indexed array)
    python tutorials/slurm/array_pipeline.py \\
        --input-dir /path/to/input/directory \\
        --output-dir /path/to/output/directory \\
        --shard-index 0 --total-shards 4

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

    # Or let the sbatch script read Slurm env vars and pass explicit args:
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

from nemo_curator.backends.base import (
    FAILED_TASKS_DIR_ENV_VAR,
    SLURM_ARRAY_ENABLED_ENV_VAR,
    SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR,
)
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter

METADATA_DIRNAME = ".nemo_curator_metadata"
SLURM_ARRAY_RETRY_DIRNAME = ".slurm_array_retry"
FAILED_TASK_MARKER_PATTERN = "failed_task_*.json"
MAX_FAILED_TASK_MARKERS_IN_MANIFEST = 10


def _safe_token(value: object) -> str:
    """Convert a value to a conservative filename token."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def _parse_int_or_env_name(value: str) -> int | str:
    """Parse an integer value or keep an environment variable name."""
    try:
        return int(value)
    except ValueError:
        return value


def _resolve_int_or_env_name(value: int | str, label: str) -> int:
    """Resolve an integer or an environment variable name containing an integer."""
    if isinstance(value, int):
        return value

    env_value = os.environ.get(value)
    if env_value is None:
        msg = f"{label} references environment variable {value}, but it is not set"
        raise ValueError(msg)
    try:
        return int(env_value)
    except ValueError as e:
        msg = f"{label} references environment variable {value}, which must contain an integer, got {env_value!r}"
        raise ValueError(msg) from e


def configure_slurm_array_source_filtering(
    shard_index: int,
    total_shards: int,
    minimum_shard_index: int,
) -> None:
    """Enable adapter-level source-task filtering for this Slurm array task."""
    os.environ[SLURM_ARRAY_ENABLED_ENV_VAR] = "1"
    os.environ[SLURM_ARRAY_SHARD_INDEX_ENV_VAR] = str(shard_index)
    os.environ[SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR] = str(total_shards)
    os.environ[SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR] = str(minimum_shard_index)


def _is_driver_process(use_slurm: bool) -> bool:
    """Return True for the process that should run the pipeline and own retry metadata."""
    return not use_slurm or os.environ.get("SLURM_NODEID", "0") == "0"


def _retry_manifest_prefix(
    shard_index: int,
    total_shards: int,
    minimum_shard_index: int,
) -> str:
    return (
        f"manifest_shard-{_safe_token(shard_index)}_"
        f"total-{_safe_token(total_shards)}_"
        f"min-{_safe_token(minimum_shard_index)}_"
    )


def _retry_manifest_payload(  # noqa: PLR0913
    shard_index: int,
    total_shards: int,
    minimum_shard_index: int,
    status: str,
    created_at: datetime.datetime,
    error: BaseException | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "shard_index": shard_index,
        "total_shards": total_shards,
        "minimum_shard_index": minimum_shard_index,
        "status": status,
        "created_at": created_at.isoformat(),
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

    if extra is not None:
        payload.update(extra)

    return payload


def write_retry_manifest(  # noqa: PLR0913
    checkpoint_path: str,
    shard_index: int | None,
    total_shards: int | None,
    minimum_shard_index: int,
    status: str,
    error: BaseException | None = None,
    manifest_file: Path | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a retry manifest using a unique name and atomic rename."""
    retry_dir = Path(checkpoint_path, METADATA_DIRNAME, SLURM_ARRAY_RETRY_DIRNAME).absolute()
    retry_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.datetime.now(datetime.UTC)
    manifest = _retry_manifest_payload(
        shard_index=shard_index,
        total_shards=total_shards,
        minimum_shard_index=minimum_shard_index,
        status=status,
        created_at=created_at,
        error=error,
        extra=extra,
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


def failed_task_marker_files() -> list[Path]:
    """Return FailedTask marker files written by BaseStageAdapter for this job."""
    failed_tasks_dir = os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    if not failed_tasks_dir:
        return []

    marker_dir = Path(failed_tasks_dir).absolute()
    if not marker_dir.exists():
        return []

    return sorted(path for path in marker_dir.glob(FAILED_TASK_MARKER_PATTERN) if path.is_file())


def failed_task_manifest_metadata(marker_files: list[Path]) -> dict[str, object]:
    marker_dir = marker_files[0].parent if marker_files else os.environ.get(FAILED_TASKS_DIR_ENV_VAR)
    sample_marker_files = marker_files[:MAX_FAILED_TASK_MARKERS_IN_MANIFEST]
    return {
        "failed_task_marker_dir": str(marker_dir),
        "failed_task_marker_count": len(marker_files),
        "failed_task_marker_files": [str(path) for path in sample_marker_files],
        "failed_task_marker_files_truncated": len(marker_files) > len(sample_marker_files),
    }


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
) -> Pipeline:
    pipeline = Pipeline(
        name="slurm_array_demo",
        description=(
            "Read files from input directory assigned to this Slurm array task and write them out to output directory."
        ),
    )

    if input_file_type == "jsonl":
        pipeline.add_stage(
            JsonlReader(
                file_paths=input_dir,
                files_per_partition=files_per_partition,
            )
        )
    elif input_file_type == "parquet":
        pipeline.add_stage(
            ParquetReader(
                file_paths=input_dir,
                files_per_partition=files_per_partition,
            )
        )
    else:
        msg = f"Unsupported input file type: {input_file_type}"
        raise ValueError(msg)

    if output_file_type == "jsonl":
        pipeline.add_stage(JsonlWriter(output_dir))
    elif output_file_type == "parquet":
        pipeline.add_stage(ParquetWriter(output_dir))
    else:
        msg = f"Unsupported output file type: {output_file_type}"
        raise ValueError(msg)

    return pipeline


def main() -> None:  # noqa: PLR0915
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
        type=_parse_int_or_env_name,
        required=True,
        help="Shard to process, as an integer or environment variable name.",
    )
    parser.add_argument(
        "--total-shards",
        type=_parse_int_or_env_name,
        required=True,
        help="Total number of shards, as an integer or environment variable name.",
    )
    parser.add_argument(
        "--minimum-shard-index",
        type=_parse_int_or_env_name,
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

    shard_index = _resolve_int_or_env_name(args.shard_index, "shard_index")
    total_shards = _resolve_int_or_env_name(args.total_shards, "total_shards")
    minimum_shard_index = _resolve_int_or_env_name(args.minimum_shard_index, "minimum_shard_index")
    configure_slurm_array_source_filtering(
        shard_index=shard_index,
        total_shards=total_shards,
        minimum_shard_index=minimum_shard_index,
    )

    ray_client = SlurmRayClient() if args.slurm else RayClient()
    retry_manifest_file = None
    is_driver_process = _is_driver_process(args.slurm)
    should_manage_retry_manifest = args.checkpoint_path is not None and is_driver_process

    try:
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
        )
        logger.info(f"\n{pipeline.describe()}")

        pipeline.run()

        failed_task_markers = failed_task_marker_files()
        if failed_task_markers:
            if should_manage_retry_manifest:
                manifest_file = write_retry_manifest(
                    checkpoint_path=args.checkpoint_path,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    minimum_shard_index=minimum_shard_index,
                    status="failed_tasks",
                    manifest_file=retry_manifest_file,
                    extra=failed_task_manifest_metadata(failed_task_markers),
                )
                logger.warning(
                    "Pipeline completed without raising, but found "
                    f"{len(failed_task_markers)} FailedTask marker(s). "
                    f"Keeping retry manifest at {manifest_file}."
                )
            else:
                logger.warning(
                    f"Pipeline completed without raising, but found {len(failed_task_markers)} FailedTask marker(s)."
                )
            return

        if should_manage_retry_manifest:
            try:
                remove_retry_manifests(
                    checkpoint_path=args.checkpoint_path,
                    shard_index=shard_index,
                    total_shards=total_shards,
                    minimum_shard_index=minimum_shard_index,
                )
            except Exception as cleanup_error:  # noqa: BLE001
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
            except Exception as manifest_error:  # noqa: BLE001
                logger.error(f"Failed to write Slurm array retry manifest: {manifest_error}")

        logger.error(f"Error running pipeline: {e}")
        raise
    finally:
        if is_driver_process:
            ray_client.stop()


if __name__ == "__main__":
    main()
