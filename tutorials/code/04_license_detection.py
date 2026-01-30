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

"""Stage 4: License Detection for Code Files.

Detect software licenses in code files using NeMo Curator's
CodeLicenseDetection stage with a distributed Ray pipeline.

Prerequisites:
    pip install nemo-curator[code]
    python tutorials/code/00_download_data.py

Usage:
    python 04_license_detection.py [--input_dir PATH] [--output_dir PATH]
"""

import argparse
from pathlib import Path

import pandas as pd

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.code import CodeLicenseDetection
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter


def print_summary(output_dir: Path) -> None:
    """Print summary of license detection results."""
    result_files = list(output_dir.glob("04_license_detected/*.jsonl"))
    if not result_files:
        return

    df = pd.concat([pd.read_json(f, lines=True) for f in result_files])
    with_license = df["has_license"].sum()
    print(f"\nProcessed {len(df)} files")
    print(f"Files with license: {with_license} ({100 * with_license / len(df):.1f}%)")

    all_licenses = {}
    for licenses in df["licenses"]:
        if isinstance(licenses, list):
            for lic in licenses:
                all_licenses[lic] = all_licenses.get(lic, 0) + 1

    if all_licenses:
        print("\nTop licenses:")
        for lic, count in sorted(all_licenses.items(), key=lambda x: -x[1])[:10]:
            print(f"  {lic}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: License Detection")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="tutorials/code/output/input_data/data",
        help="Input directory with JSONL files (run 00_download_data.sh first)",
    )
    parser.add_argument("--output_dir", type=str, default="tutorials/code/output")
    parser.add_argument("--timeout", type=int, default=100, help="Detection timeout per file (seconds)")
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
        name="Code License Detection",
        stages=[
            JsonlReader(file_paths=[str(f) for f in input_files]),
            CodeLicenseDetection(content_column="content", detection_timeout=args.timeout),
            JsonlWriter(path=str(output_dir / "04_license_detected")),
        ],
    )

    print(f"Processing {len(input_files)} files from: {input_dir}")
    pipeline.run()

    ray_client.stop()

    print_summary(output_dir)
    print(f"\nResults saved to: {output_dir / '04_license_detected'}")


if __name__ == "__main__":
    main()
