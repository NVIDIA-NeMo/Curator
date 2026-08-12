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

"""One-time data preparation for FLEURS audio benchmarks.

Downloads the FLEURS transcript + audio archive from Hugging Face
(``google/fleurs``) once and lays it out persistently so the nightly benchmark
can consume it with ``--no-auto-download`` instead of re-fetching from Hugging
Face on every run (repeated unauthenticated pulls are what triggered HTTP 429
rate-limiting in CI).

This script is NOT part of the nightly benchmark YAML -- it is run once (or
whenever the dataset needs refreshing) on the benchmarking machine.

Layout produced (consumed by ``CreateInitialManifestFleursStage`` with
``auto_download=False`` and ``raw_data_dir=<output-path>``)::

    <output-path>/<lang>/<split>.tsv     # transcript
    <output-path>/<lang>/<split>/*.wav   # extracted audio

Example usage::

    # Stage the default Armenian train split used by the nightly benchmark
    python prepare_fleurs_data.py --output-path /path/to/datasets/fleurs

    # Stage a specific language/split
    python prepare_fleurs_data.py --output-path /path/to/datasets/fleurs \\
        --lang hy_am --split test

    # Verify an existing staged dataset without downloading
    python prepare_fleurs_data.py --output-path /path/to/datasets/fleurs \\
        --lang hy_am --split train --verify-only

After running this script, reference the parent output path in your benchmark YAML::

    datasets:
      - name: "fleurs_hy_am"
        formats:
          - type: "files"
            path: "{datasets_path}/fleurs"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage


def _count_wavs(directory: Path) -> int:
    """Count ``.wav`` files under ``directory`` recursively."""
    if not directory.is_dir():
        return 0
    return sum(1 for _root, _dirs, files in os.walk(directory) for f in files if f.endswith(".wav"))


def verify_dataset(lang_dir: Path, split: str) -> bool:
    """Verify the staged transcript + audio exist and report statistics."""
    tsv_path = lang_dir / f"{split}.tsv"
    audio_root = lang_dir / split

    if not tsv_path.is_file():
        logger.error(f"Transcript not found: {tsv_path}")
        return False
    if not audio_root.is_dir():
        logger.error(f"Audio directory not found: {audio_root}")
        return False

    wav_count = _count_wavs(audio_root)
    if wav_count == 0:
        logger.error(f"No WAV files found under {audio_root}")
        return False

    num_lines = sum(1 for _ in tsv_path.open(encoding="utf-8"))
    logger.info("=" * 60)
    logger.info("FLEURS Dataset Verification")
    logger.info("=" * 60)
    logger.info(f"  Language dir: {lang_dir}")
    logger.info(f"  Transcript:   {tsv_path.name} ({num_lines} lines)")
    logger.info(f"  Audio dir:    {audio_root}")
    logger.info(f"  WAV files:    {wav_count}")
    logger.info("=" * 60)
    logger.success(f"Dataset verified: {wav_count} WAV files, {num_lines} transcript lines")
    return True


def stage_dataset(output_path: Path, lang: str, split: str, cache_dir: str | None) -> bool:
    """Download + extract FLEURS into ``<output_path>/<lang>`` for offline benchmark use.

    Reuses ``CreateInitialManifestFleursStage.download_extract_files`` so the
    download/extract logic stays in one place. That call also stages the
    transcript next to the extracted audio, so the dataset can later be found
    with ``auto_download=False``.
    """
    lang_dir = output_path / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"FLEURS Download (lang={lang}, split={split})")
    logger.info(f"Staging to: {lang_dir}")
    logger.info("=" * 60)

    stage = CreateInitialManifestFleursStage(
        lang=lang,
        split=split,
        raw_data_dir=str(output_path),
        cache_dir=cache_dir,
        auto_download=True,
    )
    # download_extract_files stages the transcript at <lang_dir>/<split>.tsv and
    # extracts the audio into <lang_dir>/<split>/, returning both final paths.
    tsv_path, audio_root = stage.download_extract_files(stage.language_data_dir())
    logger.info(f"Transcript staged at {tsv_path}")

    wav_count = _count_wavs(Path(audio_root))
    logger.success(f"Dataset ready: {wav_count} WAV files at {audio_root}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the FLEURS dataset for benchmarking (one-time staging).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Base directory to stage FLEURS into; a <lang>/ subdir is created under it.",
    )
    parser.add_argument("--lang", default="hy_am", help="Language code (default: hy_am)")
    parser.add_argument("--split", default="train", help="Dataset split: train/dev/test (default: train)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir used during the one-time download.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the existing staged dataset without downloading.",
    )

    args = parser.parse_args()
    output_path = args.output_path.resolve()
    lang_dir = output_path / args.lang

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.verify_only:
        logger.info(f"Verifying staged FLEURS dataset at: {lang_dir}")
        return 0 if verify_dataset(lang_dir, args.split) else 1

    if (lang_dir / f"{args.split}.tsv").is_file() and (lang_dir / args.split).is_dir():
        logger.info(f"Dataset already staged at {lang_dir} for split '{args.split}'")
        logger.info("Use --verify-only to check, or delete the directory to re-stage.")
        return 0 if verify_dataset(lang_dir, args.split) else 1

    if not stage_dataset(output_path, args.lang, args.split, args.cache_dir):
        return 1

    return 0 if verify_dataset(lang_dir, args.split) else 1


if __name__ == "__main__":
    raise SystemExit(main())
