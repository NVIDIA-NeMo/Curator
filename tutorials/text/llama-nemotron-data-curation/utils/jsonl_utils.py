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

import itertools
import os
from collections.abc import Generator


def stream_jsonl_files(jsonl_dir: str) -> Generator[str, None, None]:
    # Confusingly, Ray's write_json function names the files with the .json extension,
    # but the actual files are .jsonl
    files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".json"))
    for fname in files:
        with open(os.path.join(jsonl_dir, fname)) as f:
            for line in f:
                yield line.rstrip("\n")


def chunked(gen: Generator[str, None, None], size: int = 100) -> Generator[list[str], None, None]:
    while True:
        chunk = list(itertools.islice(gen, size))
        if not chunk:  # generator exhausted
            break
        yield chunk  # may be < size on last chunk


def interleave_datasets(dir1: str, dir2: str, out_path: str, chunk_size: int = 1) -> None:
    gen1 = chunked(stream_jsonl_files(dir1), chunk_size)
    gen2 = chunked(stream_jsonl_files(dir2), chunk_size)

    with open(out_path, "w") as out:
        for chunk1, chunk2 in itertools.zip_longest(gen1, gen2, fillvalue=[]):
            # write chunk1 (may be empty if dir1 is exhausted)
            out.writelines(line + "\n" for line in chunk1)
            # write chunk2 (may be empty if dir2 is exhausted)
            out.writelines(line + "\n" for line in chunk2)

    print(f"Interleaved datasets from {dir1} and {dir2} into {out_path} with adaptive chunk size up to {chunk_size}")
