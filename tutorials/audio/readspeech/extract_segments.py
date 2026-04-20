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

"""
Segment Extraction CLI — thin wrapper around the package implementation.

Reads manifest JSONL file(s) and extracts audio segments from original files.
Automatically detects the pipeline combo from the manifest schema and applies
the appropriate extraction strategy.

Each segment is saved with naming convention:
  With speaker separation:    {original_filename}_speaker_{x}_segment_{y}.{format}
  Without speaker separation: {original_filename}_segment_{y}.{format}

Usage:
    python extract_segments.py --manifest manifest.jsonl --output-dir extracted/
    python extract_segments.py --manifest /path/to/result_dir/ --output-dir out/
    python extract_segments.py --manifest result_dir/ --output-dir out/ --output-format flac

See ``nemo_curator.stages.audio.io.extract_segments`` for the full API.
"""

import argparse
import os
import sys

from loguru import logger

from nemo_curator.stages.audio.io.extract_segments import (
    DEFAULT_OUTPUT_FORMAT,
    SegmentExtractionStage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract audio segments from original files based on manifest")
    parser.add_argument(
        "--manifest", "-m", required=True, help="Path to manifest.jsonl file or directory containing .jsonl files"
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for extracted segments")
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["wav", "flac", "ogg"],
        help="Output audio format (default: wav)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    if not os.path.exists(args.manifest):
        logger.error(f"Manifest path not found: {args.manifest}")
        return 1

    logger.info(f"Output format: {args.output_format}")
    stage = SegmentExtractionStage(output_dir=args.output_dir, output_format=args.output_format)
    stage.extract_from_manifest(input_path=args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
