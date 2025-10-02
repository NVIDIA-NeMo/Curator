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


def _split_pyarrow_table(table: pa.Table, target_size_mb: int) -> list[pa.Table]:
    target_size_bytes = target_size_mb * 1024 * 1024
    # Slice full table into two chunks
    tables = [table.slice(0, table.num_rows // 2), table.slice(table.num_rows // 2, table.num_rows)]
    results = []
    for t in tables:
        if t.nbytes > target_size_bytes:
            # If above the target size, continue chunking until chunks
            # are below the target size
            results.extend(_split_pyarrow_table(t, target_size_mb=target_size_mb))
        else:
            results.append(t)
    return results


def _split_parquet_file_by_size(infile: str, outdir: str, target_size_mb: int, verbose: bool) -> None:
    root, ext = os.path.splitext(infile)
    if not ext:
        ext = ".parquet"
    outfile_prefix = os.path.basename(root)
    table = pq.read_table(infile)
    if verbose:
        print(f"""Splitting file into smaller ones...

Input file: {infile} (~{table.nbytes / (1024 * 1024):.2f} MB)
Output directory: {outdir}
Target size: {target_size_mb} MB
""")
    results = _split_pyarrow_table(table, target_size_mb=target_size_mb)
    if verbose:
        print(f"Splitting into {len(results)} output files")
    for idx, t in enumerate(results):
        # Write chunk to a new Parquet file
        output_file = os.path.join(outdir, f"{outfile_prefix}_{idx}{ext}")
        pq.write_table(t, output_file)
        if verbose:
            print(f"Saved {output_file} with approximate size {t.nbytes / (1024 * 1024):.2f} MB")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    _split_parquet_file_by_size(
        infile=args.infile, outdir=args.outdir, target_size_mb=args.target_size_mb, verbose=args.verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True, help="Path to input file to split")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to store split files")
    parser.add_argument("--target-size-mb", type=int, default=128, help="Target size (in MB) of split output files")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    main(args)
