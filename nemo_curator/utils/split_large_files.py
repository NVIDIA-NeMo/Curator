# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
import ray
from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.utils.file_utils import get_all_file_paths_under


def _split_table(table: pa.Table, target_size: int) -> list[pa.Table]:
    # Split table into two chunks
    tables = [table.slice(0, table.num_rows // 2), table.slice(table.num_rows // 2, table.num_rows)]
    results = []
    for t in tables:
        if t.nbytes > target_size:
            # If still above the target size, continue spliting until chunks
            # are below the target size
            results.extend(_split_table(t, target_size=target_size))
        else:
            results.append(t)
    return results


def _write_table_to_file(table: pa.Table, outdir: str, output_prefix: str, ext: str, file_idx: int) -> int:
    output_file = os.path.join(outdir, f"{output_prefix}_{file_idx}{ext}")
    pq.write_table(table, output_file)
    logger.debug(f"Saved {output_file} (~{table.nbytes / (1024 * 1024):.2f} MB)")
    return file_idx + 1


@ray.remote
def split_parquet_file_by_size(input_file: str, outdir: str, target_size_mb: int) -> None:
    root, ext = os.path.splitext(input_file)
    if not ext:
        ext = ".parquet"
    outfile_prefix = os.path.basename(root)

    logger.info(f"""Splitting parquet file...

Input file: {input_file}
Output directory: {outdir}
Target size: {target_size_mb} MB
""")

    pf = pq.ParquetFile(input_file)
    num_row_groups = pf.num_row_groups
    target_size_bytes = target_size_mb * 1024 * 1024
    file_idx = 0
    row_group_idx = 0

    # Loop over all row groups in the file, splitting or merging row groups as needed
    # to hit the target size.
    while row_group_idx < num_row_groups:
        current_size = 0
        row_groups_to_write = []

        while row_group_idx < num_row_groups and current_size < target_size_bytes:
            row_group = pf.read_row_group(row_group_idx)

            if row_group.nbytes > target_size_bytes:
                # Large row group case. Split into smaller chunks to get below target size.
                chunks = _split_table(row_group, target_size=target_size_bytes)
                for chunk in chunks:
                    file_idx = _write_table_to_file(
                        chunk, outdir=outdir, output_prefix=outfile_prefix, ext=ext, file_idx=file_idx
                    )
                row_group_idx += 1
            elif row_group.nbytes + current_size > target_size_bytes:
                # Adding the current row group will push over the desired target size, so
                # write current batch to a file.
                break
            else:
                # Case where we need to merge smaller row groups into a single table
                row_groups_to_write.append(row_group)
                current_size += row_group.nbytes
                row_group_idx += 1

        if row_groups_to_write:
            sub_table = pa.concat_tables(row_groups_to_write)
            file_idx = _write_table_to_file(
                sub_table, outdir=outdir, output_prefix=outfile_prefix, ext=ext, file_idx=file_idx
            )


def parse_args(args: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile", type=str, required=True, help="Path to input file, or directory of files, to split"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to store split files")
    parser.add_argument("--target-size-mb", type=int, default=128, help="Target size (in MB) of split output files")
    return parser.parse_args(args)


def main(args: argparse.ArgumentParser | None = None) -> None:
    args = parse_args(args)

    files = get_all_file_paths_under(args.infile)
    if not files:
        logger.error(f"No file(s) found at '{args.infile}'")
        return

    os.makedirs(args.outdir, exist_ok=True)
    with RayClient():
        ray.get(
            [
                split_parquet_file_by_size.remote(input_file=f, outdir=args.outdir, target_size_mb=args.target_size_mb)
                for f in files
            ]
        )


if __name__ == "__main__":
    main()
