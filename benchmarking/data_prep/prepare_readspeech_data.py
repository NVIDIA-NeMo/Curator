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

Downloads the DNS Challenge Read Speech dataset and extracts the WAV files
to a persistent location. This script is NOT part of the nightly benchmark
YAML -- it is run once (or whenever the dataset needs refreshing).

The full dataset has 21 parts (partaa-partau), each ~4.88 GB (~102 GB total
archive, ~299 GB extracted). By default only 1 part is downloaded (~4.88 GB,
14,279 WAV files). Use --num-parts to download more.

The extracted dataset can then be referenced in benchmark YAML configs via
the ``datasets_path`` placeholder, avoiding repeated downloads.

Example usage:

    # Download 1 part (default, ~4.88 GB)
    python prepare_readspeech_data.py --output-path /path/to/datasets/read_speech

    # Download first 10 parts (~48.8 GB)
    python prepare_readspeech_data.py --output-path /path/to/datasets/read_speech --num-parts 10

    # Download all 21 parts (~102 GB)
    python prepare_readspeech_data.py --output-path /path/to/datasets/read_speech --num-parts 21

    # Verify existing download without re-downloading
    python prepare_readspeech_data.py --output-path /path/to/datasets/read_speech --verify-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from nemo_curator.stages.audio.datasets.file_utils import download_file
from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage

DNS_READSPEECH_BASE_URL = (
    "https://dnschallengepublic.blob.core.windows.net/dns5archive/"
    "V5_training_dataset/Track1_Headset/read_speech.tgz."
)

DNS_READSPEECH_PARTS = [
    "partaa", "partab", "partac", "partad", "partae", "partaf", "partag",
    "partah", "partai", "partaj", "partak", "partal", "partam", "partan",
    "partao", "partap", "partaq", "partar", "partas", "partat", "partau",
]


def collect_all_wavs(base_dir: Path) -> list[str]:
    """Recursively collect all WAV file paths under base_dir."""
    wav_files = []
    if not base_dir.exists():
        return wav_files
    for root, _dirs, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    return wav_files


def verify_dataset(output_path: Path) -> bool:
    """Verify the dataset exists and report statistics by scanning all WAVs recursively."""
    all_wavs = collect_all_wavs(output_path)
    if not all_wavs:
        logger.error(f"No WAV files found under {output_path}")
        return False

    total_size = sum(os.path.getsize(f) for f in all_wavs)
    total_size_gb = total_size / (1024**3)

    wav_dirs = sorted({os.path.dirname(f) for f in all_wavs})

    logger.info("=" * 60)
    logger.info("ReadSpeech Dataset Verification")
    logger.info("=" * 60)
    logger.info(f"  Base path:      {output_path}")
    logger.info(f"  WAV files:      {len(all_wavs)}")
    logger.info(f"  Total size:     {total_size_gb:.2f} GB")
    logger.info(f"  Directories:    {len(wav_dirs)}")
    for d in wav_dirs:
        dir_count = sum(1 for f in all_wavs if os.path.dirname(f) == d)
        logger.info(f"    {d} ({dir_count} files)")
    logger.info("=" * 60)

    logger.success(f"Dataset verified: {len(all_wavs)} WAV files ({total_size_gb:.2f} GB) across {len(wav_dirs)} directories")
    return True


def _download_single_part(output_path: Path) -> bool:
    """Download single-part dataset using the existing stage logic."""
    stage = CreateInitialManifestReadSpeechStage(
        raw_data_dir=str(output_path),
        max_samples=-1,
        auto_download=True,
    )
    extracted_dir = stage.download_and_extract()
    all_wavs = collect_all_wavs(Path(extracted_dir))
    logger.success(f"Dataset ready: {len(all_wavs)} WAV files at {extracted_dir}")
    return True


def _download_parts(output_path: Path, parts: list[str]) -> list[str]:
    """Download archive parts, skipping already-downloaded ones. Returns list of file paths."""
    downloaded_files = []
    num_parts = len(parts)
    for i, part in enumerate(parts, 1):
        url = DNS_READSPEECH_BASE_URL + part
        filename = f"read_speech.tgz.{part}"
        filepath = output_path / filename

        if filepath.exists() and filepath.stat().st_size > 0:
            logger.info(f"  [{i}/{num_parts}] Already downloaded: {filename} ({filepath.stat().st_size / (1024**3):.2f} GB)")
        else:
            logger.info(f"  [{i}/{num_parts}] Downloading {filename}...")
            download_file(url, str(output_path), verbose=True)
            logger.info(f"  [{i}/{num_parts}] Downloaded: {filepath.stat().st_size / (1024**3):.2f} GB")
        downloaded_files.append(str(filepath))
    return downloaded_files


def _combine_and_extract(output_path: Path, downloaded_files: list[str]) -> bool:
    """Combine split archive parts, extract, and clean up."""
    combined_archive = output_path / "read_speech.tgz"
    logger.info(f"Combining {len(downloaded_files)} parts into {combined_archive.name}...")
    with open(str(combined_archive), "wb") as outfile:
        for fpath in downloaded_files:
            with open(fpath, "rb") as infile:
                while True:
                    chunk = infile.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)

    logger.info(f"Combined archive: {combined_archive.stat().st_size / (1024**3):.2f} GB")
    logger.info("Extracting...")

    result = subprocess.run(  # noqa: S603
        ["tar", "-xzf", str(combined_archive), "-C", str(output_path), "--ignore-zeros"],  # noqa: S607
        capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 2):
        logger.error(f"Extraction failed (exit code {result.returncode}): {result.stderr[:500]}")
        return False

    for fpath in downloaded_files:
        os.remove(fpath)
    os.remove(str(combined_archive))
    logger.info("Cleaned up archive files")
    return True


def download_dataset(output_path: Path, num_parts: int = 1) -> bool:
    """Download and extract the ReadSpeech dataset.

    Args:
        output_path: Directory to download and extract into.
        num_parts: Number of archive parts to download (1-21). Each part is ~4.88 GB.
    """
    try:
        if num_parts == 1:
            return _download_single_part(output_path)

        parts = DNS_READSPEECH_PARTS[:num_parts]
        total_size_gb = num_parts * 4.88
        logger.info("=" * 60)
        logger.info(f"DNS Challenge 5 - Read Speech Download ({num_parts} parts, ~{total_size_gb:.1f} GB)")
        logger.info(f"Downloading to: {output_path}")
        logger.info("=" * 60)

        os.makedirs(str(output_path), exist_ok=True)
        downloaded_files = _download_parts(output_path, parts)

        if not _combine_and_extract(output_path, downloaded_files):
            return False

        all_wavs = collect_all_wavs(output_path)
        if not all_wavs:
            logger.error("No WAV files found after extraction")
            return False

        wav_dirs = sorted({os.path.dirname(f) for f in all_wavs})
        logger.success(f"Dataset ready: {len(all_wavs)} WAV files across {len(wav_dirs)} directories")

    except Exception:
        logger.exception("Dataset download/extraction failed")
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DNS Challenge Read Speech dataset for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset: DNS Challenge Read Speech (Track 1 Headset)
Source: https://github.com/microsoft/DNS-Challenge

The full dataset has 21 parts, each ~4.88 GB (~102 GB total, ~299 GB extracted).
Part 1 alone contains 14,279 WAV files at 48kHz (19.3 hours).

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
        "--num-parts",
        type=int,
        default=1,
        choices=range(1, 22),
        metavar="[1-21]",
        help="Number of archive parts to download (default: 1, each ~4.88 GB, max: 21 for full ~102 GB)",
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

    existing_wavs = collect_all_wavs(output_path)
    if existing_wavs:
        wav_dirs = sorted({os.path.dirname(f) for f in existing_wavs})
        logger.info(f"Dataset already exists: {len(existing_wavs)} WAV files across {len(wav_dirs)} directories")
        logger.info("Use --verify-only to check, or delete the directory to re-download")
        return 0

    if not download_dataset(output_path, num_parts=args.num_parts):
        return 1

    return 0 if verify_dataset(output_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
