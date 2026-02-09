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

"""Convert a CSV file to JSONL (one JSON object per line)."""

import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL")
    parser.add_argument("--input_file", required=True, help="Input CSV path")
    parser.add_argument("--output_dir", required=True, help="Output directory (JSONL files written inside)")
    parser.add_argument("--records_per_jsonl_file", type=int, default=100, help="Records per JSONL file (default: 100)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep=",", encoding="utf-8")

    os.makedirs(args.output_dir, exist_ok=True)
    n = args.records_per_jsonl_file
    for i, start in enumerate(range(0, len(df), n)):
        chunk = df.iloc[start : start + n]
        path = os.path.join(args.output_dir, f"{i:06d}.jsonl")
        chunk.to_json(path, orient="records", lines=True, force_ascii=False, date_format="iso")


if __name__ == "__main__":
    main()
