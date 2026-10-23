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

"""
JSONL/Parquet pipeline used by the Slurm array tutorial.
Refer to the README for more details.
"""

from __future__ import annotations

import argparse

from loguru import logger

from nemo_curator.backends.slurm_array import is_slurm_array_driver_process
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


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
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path for checkpoint metadata. Slurm array completion manifests are written under "
            "<checkpoint_path>/.nemo_curator_metadata/.slurm_array_completion/. Defaults to None."
        ),
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Use SlurmRayClient for multi-node srun jobs.",
    )
    args = parser.parse_args()

    ray_client = SlurmRayClient() if args.slurm else RayClient()

    pipeline = build_pipeline(
        args.input_dir,
        args.input_file_type,
        args.output_dir,
        args.output_file_type,
        args.files_per_partition,
    )
    logger.info(f"\n{pipeline.describe()}")

    try:
        ray_client.start()
        pipeline.run(checkpoint_path=args.checkpoint_path)
    except Exception as e:
        logger.error(f"Error running pipeline; shard remains incomplete: {e}")
        raise
    finally:
        if is_slurm_array_driver_process():
            ray_client.stop()


if __name__ == "__main__":
    main()
