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

"""Sample random Wikipedia pages from tar.gz archives and deduplicate by identifier."""

import json
import os
import tarfile
from random import Random
from typing import BinaryIO

import click
import tqdm
import zstandard as zstd


@click.command()
@click.argument("tar_path", type=click.Path(exists=True))
@click.argument("outpath", type=click.Path())
@click.option("--sample-size", type=int, default=1010000)
@click.option("--seed", type=int, default=0)
def reservoir_sample_pages(tar_path: str, outpath: str, sample_size: int, seed: int):
    """
    Deduplicate and sample N random pages from Wikipedia NDJSON files.
    Use reservoir sampling algorithm.

    Args:
        tar_path: Path to the tar.gz file containing NDJSON files
        outpath: Path to save sampled pages (.zst)
        sample_size: Number of pages to sample
    """
    offsets = []
    total_pages = 0
    skipped = 0
    duplicates = 0
    rng = Random(seed)
    seen = set()

    # the 2nd stage is I/O bound so compress reservoir items on disk
    compressor = zstd.ZstdCompressor()

    def write_record(f: BinaryIO, line: bytes):
        compressed_data = compressor.compress(line)
        f.write(len(compressed_data).to_bytes(8, "little"))
        f.write(compressed_data)

    progress = tqdm.tqdm(total=os.path.getsize(tar_path), desc="reading")
    with open(f"{outpath}.tmp", "wb") as fr:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".ndjson"):
                    continue
                f = tar.extractfile(member)
                for line in f:
                    # exclude short/stub documents
                    if len(line) < 20000:
                        skipped += 1
                        continue
                    # exclude duplicates
                    identifier = json.loads(line)["identifier"]
                    assert isinstance(identifier, int), "expected integer identifier"
                    if identifier in seen:
                        duplicates += 1
                        continue
                    seen.add(identifier)
                    total_pages += 1
                    assert line.endswith(b"\n"), "no trailing newline"
                    if len(offsets) < sample_size:
                        offsets.append(fr.tell())
                        write_record(fr, line)
                    else:
                        j = rng.randint(0, total_pages - 1)
                        if j < sample_size:
                            offsets[j] = fr.tell()
                            write_record(fr, line)
                f.close()
                progress.update(tar.fileobj.fileobj.tell() - progress.n)
    print(
        f"Processed {total_pages} pages ({duplicates} duplicates, {skipped} skipped), sampled {len(offsets)} pages"
    )

    decompressor = zstd.ZstdDecompressor()

    def read_record(f: BinaryIO) -> bytes:
        length_bytes = f.read(8)
        assert len(length_bytes) == 8, "expected 8 bytes"
        length = int.from_bytes(length_bytes, "little")
        compressed_data = f.read(length)
        assert len(compressed_data) == length, f"expected {length} bytes"
        return decompressor.decompress(compressed_data)

    with zstd.open(outpath, "wb") as f, open(f"{outpath}.tmp", "rb") as fr:
        for offset in tqdm.tqdm(offsets, desc="writing"):
            fr.seek(offset)
            f.write(read_record(fr))
    os.unlink(f"{outpath}.tmp")


if __name__ == "__main__":
    reservoir_sample_pages()
