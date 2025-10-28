# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shuffle and reshard webdataset tar files with index-based random access."""

import tarfile
import math
import os
import random
import time
import tqdm
import click
from pathlib import Path
from typing import Optional


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--first-page-idx", type=int, default=0)
@click.option("--last-page-idx", type=int, default=None)
@click.option("--shard-id", type=int, default=0)
@click.option("--num-shards", type=int, default=1)
@click.option("--seed", type=int, default=0)
def shuffle(
    input_dir: Path,
    output_dir: Path,
    first_page_idx: int,
    last_page_idx: Optional[int],
    shard_id: int,
    num_shards: int,
    seed: int,
):
    """Shuffle webdataset tars."""
    shards = sorted(input_dir.glob("shard_*.tar"))
    if not shards:
        raise ValueError(f"No shards found in {input_dir}")

    if first_page_idx < 0:
        raise click.BadParameter("--first-page-idx is negative")
    if last_page_idx is not None and last_page_idx < first_page_idx:
        raise click.BadParameter("--last-page-idx must be larger than --first-page-idx")

    os.makedirs(output_dir, exist_ok=True)

    in_shard_count = len(shards)
    dist_index_wds(shards, shard_id, num_shards)
    in_shard_sizes = [os.path.getsize(f"{path}.idx") // 8 - 1 for path in shards]

    total_page_count = sum(in_shard_sizes)
    if last_page_idx is None:
        last_page_idx = total_page_count - 1
    if first_page_idx >= total_page_count:
        raise click.BadParameter(
            f"--first-page-idx is too large, {first_page_idx} >= {total_page_count}"
        )
    if last_page_idx >= total_page_count:
        raise click.BadParameter(
            f"--last-page-idx is too large, {last_page_idx} >= {total_page_count}"
        )

    rng = random.Random(seed)
    input_permutation = sorted(range(total_page_count), key=lambda _: rng.random())
    input_permutation = input_permutation[first_page_idx : last_page_idx + 1]

    # all other shards are same size except last one is smaller
    input_size = len(input_permutation)
    out_shard_size = math.ceil(input_size / num_shards)
    out_shard_start = shard_id * out_shard_size
    if shard_id == num_shards - 1:
        out_shard_size = input_size - out_shard_start
    output_ids = input_permutation[out_shard_start : out_shard_start + out_shard_size]

    def read_chunk(path: str, idx: int) -> bytes:
        with open(f"{path}.idx", "rb") as index_fd:
            index_fd.seek(idx * 8)
            buf = index_fd.read(8)
            if len(buf) != 8:
                raise IndexError(f"chunk start index out of range: {path}.idx")
            start = int.from_bytes(buf, byteorder="little")
            assert start % 512 == 0, f"invalid start position: {path}.idx"
            buf = index_fd.read(8)
            if len(buf) != 8:
                raise IndexError(f"chunk end index out of range: {path}.idx")
            end = int.from_bytes(buf, byteorder="little")
            assert end % 512 == 0, f"invalid end position: {path}.idx"
        with open(path, "rb") as fd:
            fd.seek(start)
            chunk = fd.read(end - start)
            if len(chunk) != end - start:
                raise IndexError(f"tar file is truncated: {path}")
            assert not chunk.startswith(b"\x00" * 1024), "FIXME end marker found"
            return chunk

    outpath = f"{output_dir}/shard_{shard_id:06d}.tar"
    with open(outpath, "wb") as output_fd:
        with open(f"{outpath}.idx", "wb") as output_index_fd:
            for pos in tqdm.tqdm(output_ids, desc=f"{shard_id} shuffle"):
                for src in range(in_shard_count):
                    if pos < in_shard_sizes[src]:
                        break
                    pos -= in_shard_sizes[src]
                assert 0 <= pos < in_shard_sizes[src], f"invalid position: {pos}"
                chunk = read_chunk(shards[src], pos)
                offset = output_fd.tell()
                assert offset % 512 == 0, f"invalid tar offset: {offset}"
                output_index_fd.write(offset.to_bytes(8, byteorder="little"))
                output_fd.write(chunk)
            # write end marker
            offset = output_fd.tell()
            assert offset % 512 == 0, "invalid end marker position"
            output_index_fd.write(offset.to_bytes(8, byteorder="little"))
            output_fd.write(b"\x00" * 1024)


def dist_index_wds(paths: list[str], rank: int, world_size: int) -> None:
    """Create index files for multiple webdataset tars in parallel."""
    for path in paths[rank::world_size]:
        if not os.path.exists(f"{path}.idx"):
            index_wds_tarfile(path)
    for path in paths:
        while not os.path.exists(f"{path}.idx"):
            time.sleep(1)


def index_wds_tarfile(path: str) -> None:
    """Create a (energon-compatible) index file for webdataset tar."""
    # avoid race condition on index file
    tmp_path = f"{path}.idx.tmp.{random.randint(100000, 999999)}"
    with open(path, "rb") as fd:
        with open(tmp_path, "wb") as index_fd:
            prev_key = None
            with tarfile.open(fileobj=fd) as tar:
                for member in tqdm.tqdm(tar, desc="Indexing"):
                    curr_key = member.name.split(".")[0]
                    if curr_key != prev_key:
                        index_fd.write(member.offset.to_bytes(8, "little"))
                        prev_key = curr_key
                # write end marker
                index_fd.write(tar.offset.to_bytes(8, "little"))
    os.rename(tmp_path, f"{path}.idx")


if __name__ == "__main__":
    shuffle()
