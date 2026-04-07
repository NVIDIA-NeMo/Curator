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

from __future__ import annotations

import argparse
import json
import posixpath
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import get_all_file_paths_under


def _storage_options(storage_options: dict[str, Any] | None) -> dict[str, Any]:
    return storage_options if storage_options is not None else {}


def _basename_and_ext(path: str) -> tuple[str, str]:
    """Basename and extension for local paths and fsspec URIs (e.g. s3://bucket/key/file.jsonl)."""
    name = posixpath.basename(path.rstrip("/"))
    root, ext = posixpath.splitext(name)
    return root, ext


def _join_out_path(output_path: str, filename: str, storage_options: dict[str, Any]) -> str:
    """Join output directory and filename using the target filesystem (local or remote)."""
    fs, root = url_to_fs(str(output_path), **storage_options)
    joined = fs.sep.join([root.rstrip(fs.sep), filename])
    return fs.unstrip_protocol(joined) if is_remote_url(str(output_path)) else joined


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


def _write_table_to_file(table: pa.Table, output_file: str, storage_options: dict[str, Any]) -> None:
    with fsspec.open(output_file, "wb", **storage_options) as out_f:
        pq.write_table(table, out_f)
    logger.debug("Saved {} (~{:.2f} MB)", output_file, table.nbytes / (1024 * 1024))


@ray.remote
def split_parquet_file_by_size(
    input_file: str, output_path: str, target_size_mb: int, storage_options: dict[str, Any] | None = None
) -> None:
    root, ext = _basename_and_ext(input_file)
    if not ext:
        ext = ".parquet"
    outfile_prefix = root

    logger.info(
        "Splitting parquet file...\n\nInput file: {}\nOutput directory: {}\nTarget size: {} MB\n",
        input_file,
        output_path,
        target_size_mb,
    )

    so = _storage_options(storage_options)
    with fsspec.open(str(input_file), "rb", **so) as in_f:
        pf = pq.ParquetFile(in_f)
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
                    # Flush any pending small row groups first to preserve order.
                    if row_groups_to_write:
                        sub_table = (
                            row_groups_to_write[0]
                            if len(row_groups_to_write) == 1
                            else pa.concat_tables(row_groups_to_write)
                        )
                        out_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
                        _write_table_to_file(sub_table, out_file, so)
                        file_idx += 1
                        row_groups_to_write = []
                        current_size = 0
                    # Now write the large row group's chunks.
                    chunks = _split_table(row_group, target_size=target_size_bytes)
                    for chunk in chunks:
                        out_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
                        _write_table_to_file(chunk, out_file, so)
                        file_idx += 1
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
                sub_table = (
                    row_groups_to_write[0] if len(row_groups_to_write) == 1 else pa.concat_tables(row_groups_to_write)
                )
                out_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
                _write_table_to_file(sub_table, out_file, so)
                file_idx += 1


def _flush_jsonl_chunk(lines: list[bytes], output_file: str, storage_options: dict[str, Any]) -> None:
    with fsspec.open(output_file, "wb", **storage_options) as out_f:
        out_f.writelines(lines)
    nbytes = sum(len(line) for line in lines)
    logger.debug("Saved {} (~{:.2f} MB)", output_file, nbytes / (1024 * 1024))


@ray.remote
def split_jsonl_file_by_size(
    input_file: str, output_path: str, target_size_mb: int, storage_options: dict[str, Any] | None = None
) -> None:
    # Stream line-by-line in binary mode (O(line size) memory). Lines larger than the target are
    # written alone and may exceed the target. JSONL records cannot be split mid-line.
    root, ext = _basename_and_ext(input_file)
    if not ext:
        ext = ".jsonl"
    outfile_prefix = root

    logger.info(
        "Splitting jsonl file...\n\nInput file: {}\nOutput directory: {}\nTarget size: {} MB\n",
        input_file,
        output_path,
        target_size_mb,
    )

    target_size_bytes = target_size_mb * 1024 * 1024
    file_idx = 0
    chunk_lines: list[bytes] = []
    chunk_bytes = 0

    so = _storage_options(storage_options)
    with fsspec.open(str(input_file), "rb", **so) as in_f:
        for line in in_f:
            line_len = len(line)
            if line_len > target_size_bytes:
                if chunk_lines:
                    out_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
                    _flush_jsonl_chunk(chunk_lines, out_file, so)
                    chunk_lines = []
                    chunk_bytes = 0
                    file_idx += 1
                output_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
                _flush_jsonl_chunk([line], output_file, so)
                logger.warning(
                    "Single line ({} bytes) exceeds target ({} bytes); wrote as its own shard: {}",
                    line_len,
                    target_size_bytes,
                    output_file,
                )
                file_idx += 1
                continue

            if chunk_bytes + line_len > target_size_bytes and chunk_lines:
                out_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
                _flush_jsonl_chunk(chunk_lines, out_file, so)
                chunk_lines = []
                chunk_bytes = 0
                file_idx += 1

            chunk_lines.append(line)
            chunk_bytes += line_len

    if chunk_lines:
        out_file = _join_out_path(output_path, f"{outfile_prefix}_{file_idx}{ext}", so)
        _flush_jsonl_chunk(chunk_lines, out_file, so)


def parse_args(args: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to input file, or directory of files, to split"
    )
    parser.add_argument(
        "--file-type", type=str, required=True, help="Type of file to split", choices=["parquet", "jsonl"]
    )
    parser.add_argument("--output-path", type=str, required=True, help="Output directory to store split files")
    parser.add_argument("--target-size-mb", type=int, default=128, help="Target size (in MB) of split output files")
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help="Optional JSON object of fsspec storage options (credentials, endpoint_url, etc.) for remote paths.",
    )
    return parser.parse_args(args)


def main(args: argparse.ArgumentParser | None = None) -> None:
    args = parse_args(args)

    storage_options: dict[str, Any] | None = json.loads(args.storage_options) if args.storage_options else None

    files = get_all_file_paths_under(args.input_path, keep_extensions=args.file_type, storage_options=storage_options)
    if not files:
        logger.error("No file(s) found at '{}'", args.input_path)
        return

    out_fs, out_root = url_to_fs(str(args.output_path), **_storage_options(storage_options))
    out_fs.makedirs(out_root, exist_ok=True)
    _handlers = {"parquet": split_parquet_file_by_size, "jsonl": split_jsonl_file_by_size}

    with RayClient():
        ray.get(
            [
                _handlers[args.file_type].remote(
                    input_file=f,
                    output_path=args.output_path,
                    target_size_mb=args.target_size_mb,
                    storage_options=storage_options,
                )
                for f in files
            ]
        )


if __name__ == "__main__":
    main()
