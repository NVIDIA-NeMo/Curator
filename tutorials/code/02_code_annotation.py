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

"""Stage 2: Code Annotation with Quality Metrics.

Compute quality metrics for code files using NeMo Curator's CodeAnnotation
stage with a distributed Ray pipeline.

Prerequisites:
    pip install nemo-curator[code]
    python tutorials/code/00_download_data.py

Usage:
    python 02_code_annotation.py [--input_dir PATH] [--output_dir PATH]
"""

import argparse
from pathlib import Path

import pandas as pd

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.code import CodeAnnotation
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter


def print_summary(output_dir: Path) -> None:
    """Print summary of annotation results."""
    result_files = list(output_dir.glob("02_annotated/*.jsonl"))
    if not result_files:
        return

    df = pd.concat([pd.read_json(f, lines=True) for f in result_files])
    print(f"\nProcessed {len(df)} files with {len(df.columns)} columns")
    print(f"Total bytes: {df['basic_num_bytes'].sum():,}")
    print(f"Total lines: {df['basic_num_lines'].sum():,}")
    print(f"Avg comment fraction: {df['ors_comment_lines_frac'].mean():.2%}")

    token_cols = [c for c in df.columns if c.startswith("num_tokens_")]
    if token_cols:
        print(f"Total tokens: {df[token_cols[0]].sum():,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Code Annotation")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="tutorials/code/output/input_data/data",
        help="Input directory with JSONL files (run 00_download_data.sh first)",
    )
    parser.add_argument("--output_dir", type=str, default="tutorials/code/output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    input_files = list(input_dir.glob("*/data.json"))
    if not input_files:
        print(f"Error: No JSONL files found in: {input_dir}")
        print("Run './tutorials/code/00_download_data.sh' first to download the dataset.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ray_client = RayClient()
    ray_client.start()

    pipeline = Pipeline(
        name="Code Annotation",
        stages=[
            JsonlReader(file_paths=[str(f) for f in input_files]),
            # Use 'ext' column (file extension) from the-stack-smol-xs dataset
            CodeAnnotation(
                content_column="content",
                filename_column="ext",
                detect_language=True,
                basic_stats=True,
                software_metrics=True,  # Slower but provides complexity metrics
                opencoder_metrics=True,
                tokenize=True,
            ),
            JsonlWriter(path=str(output_dir / "02_annotated")),
        ],
    )

    print(f"Processing {len(input_files)} files from: {input_dir}")
    pipeline.run()

    ray_client.stop()

    print_summary(output_dir)
    print(f"\nResults saved to: {output_dir / '02_annotated'}")


if __name__ == "__main__":
    main()
