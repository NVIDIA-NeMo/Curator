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

import itertools
import os
from collections.abc import Generator

import ray


@ray.remote
def split_jsonl_by_size(input_file: str, target_size_mb: int, output_dir: str, output_prefix: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    target_size = target_size_mb * 1024 * 1024
    file_count = 0
    bytes_written = 0
    out = None

    base_name = os.path.basename(input_file)  # e.g., "chat.jsonl"
    base_name_no_ext = os.path.splitext(base_name)[0]  # "chat"

    with open(input_file, encoding="utf-8") as infile:
        for line in infile:
            if out is None or bytes_written + len(line.encode("utf-8")) > target_size:
                if out:
                    out.close()
                out_path = os.path.join(output_dir, f"{output_prefix}_{base_name_no_ext}_{file_count}.jsonl")
                out = open(out_path, "w", encoding="utf-8")  # noqa: SIM115
                file_count += 1
                bytes_written = 0
            out.write(line)
            bytes_written += len(line.encode("utf-8"))

    if out:
        out.close()


def stream_jsonl_files(jsonl_dir: str) -> Generator[str, None, None]:
    # Confusingly, Ray's write_json function names the files with the .json extension,
    # but the actual files are .jsonl
    files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".json"))
    for fname in files:
        with open(os.path.join(jsonl_dir, fname)) as f:
            for line in f:
                yield line.rstrip("\n")


def interleave_datasets(dir1: str, dir2: str, out_path: str) -> None:
    gen1 = stream_jsonl_files(dir1)
    gen2 = stream_jsonl_files(dir2)

    with open(out_path, "w") as out:
        for line1, line2 in itertools.zip_longest(gen1, gen2):
            if line1 is not None:
                out.write(line1 + "\n")
            if line2 is not None:
                out.write(line2 + "\n")

    print(f"Interleaved datasets from directories {dir1} and {dir2} into file {out_path}")
