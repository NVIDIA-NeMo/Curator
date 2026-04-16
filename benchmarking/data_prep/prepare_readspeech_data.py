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

"""One-time data preparation for ReadSpeech audio benchmarks.

Downloads the DNS Challenge Read Speech dataset (~4.88 GB) and extracts the
WAV files to a persistent location. This script is NOT part of the nightly
benchmark YAML -- it is run once (or whenever the dataset needs refreshing).

The extracted dataset can then be referenced in benchmark YAML configs via
the ``datasets_path`` placeholder, avoiding repeated downloads.

Example usage:

    # Download and extract to datasets directory
    python prepare_readspeech_data.py --output-path /path/to/datasets/read_speech

    # Verify existing download without re-downloading
    python prepare_readspeech_data.py --output-path /path/to/datasets/read_speech --verify-only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from loguru import logger

from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage


def count_wavs(directory: Path) -> int:
    """Count WAV files recursively under a directory."""
    count = 0
    for _root, _dirs, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(".wav"))
    return count


def find_wav_directory(base_dir: Path) -> Path | None:
    """Find the directory containing WAV files under base_dir."""
    if not base_dir.exists():
        return None

    if glob.glob(str(base_dir / "*.wav")):
        return base_dir

    known_subdirs = [
        "read_speech",
        "mnt/dnsv5/clean/read_speech",
        "data/mnt/dnsv5/clean/read_speech",
    ]
    for subdir in known_subdirs:
        check_path = base_dir / subdir
        if check_path.exists() and glob.glob(str(check_path / "*.wav")):
            return check_path

    for root, _dirs, files in os.walk(base_dir):
        if any(f.endswith(".wav") for f in files):
            return Path(root)

    return None


def verify_dataset(output_path: Path) -> bool:
    """Verify the dataset exists and report statistics."""
    wav_dir = find_wav_directory(output_path)
    if not wav_dir:
        logger.error(f"No WAV files found under {output_path}")
        return False

    wav_count = count_wavs(wav_dir)
    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _dirs, files in os.walk(wav_dir)
        for f in files
        if f.endswith(".wav")
    )
    total_size_gb = total_size / (1024**3)

    logger.info("=" * 60)
    logger.info("ReadSpeech Dataset Verification")
    logger.info("=" * 60)
    logger.info(f"  Location:   {wav_dir}")
    logger.info(f"  WAV files:  {wav_count}")
    logger.info(f"  Total size: {total_size_gb:.2f} GB")
    logger.info("=" * 60)

    if wav_count == 0:
        logger.error("Dataset appears empty")
        return False

    logger.success(f"Dataset verified: {wav_count} WAV files ({total_size_gb:.2f} GB)")
    return True


def download_dataset(output_path: Path) -> bool:
    """Download and extract the ReadSpeech dataset using the existing stage logic."""
    stage = CreateInitialManifestReadSpeechStage(
        raw_data_dir=str(output_path),
        max_samples=-1,
        auto_download=True,
    )

    try:
        extracted_dir = stage.download_and_extract()
        wav_count = count_wavs(Path(extracted_dir))
        logger.success(f"Dataset ready: {wav_count} WAV files at {extracted_dir}")
        return True
    except Exception:
        logger.exception("Dataset download/extraction failed")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DNS Challenge Read Speech dataset for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset: DNS Challenge Read Speech (Track 1 Headset)
Source: https://github.com/microsoft/DNS-Challenge

Contains 14,279 clean read speech WAV files at 48kHz (19.3 hours, ~4.88 GB download).

After running this script, reference the output path in your benchmark YAML:
  datasets:
    - name: "read_speech"
      formats:
        - type: "wav"
          path: "{datasets_path}/read_speech"
        """,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory to download and extract the dataset into",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset without downloading",
    )

    args = parser.parse_args()
    output_path = args.output_path.resolve()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.verify_only:
        logger.info(f"Verifying dataset at: {output_path}")
        return 0 if verify_dataset(output_path) else 1

    logger.info(f"Preparing ReadSpeech dataset at: {output_path}")

    existing_wav_dir = find_wav_directory(output_path)
    if existing_wav_dir:
        wav_count = count_wavs(existing_wav_dir)
        logger.info(f"Dataset already exists: {wav_count} WAV files at {existing_wav_dir}")
        logger.info("Use --verify-only to check, or delete the directory to re-download")
        return 0

    if not download_dataset(output_path):
        return 1

    return 0 if verify_dataset(output_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
