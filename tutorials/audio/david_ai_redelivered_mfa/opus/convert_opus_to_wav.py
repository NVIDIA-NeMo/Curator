#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Convert an Opus directory tree to mono PCM WAV while preserving relative paths."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionResult:
    source: Path
    destination: Path
    status: str
    error: str = ""


def destination_for(source: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / source.relative_to(input_dir)).with_suffix(".wav")


def convert_one(
    source: Path,
    destination: Path,
    *,
    ffmpeg: str,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> ConversionResult:
    if destination.is_file() and not overwrite:
        return ConversionResult(source, destination, "skipped")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{threading.get_ident()}.tmp.wav"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(temp),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, text=True)
        if result.returncode != 0:
            return ConversionResult(source, destination, "failed", result.stderr[-1000:])
        os.replace(temp, destination)
        return ConversionResult(source, destination, "converted")
    except OSError as exc:
        return ConversionResult(source, destination, "failed", str(exc))
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing .opus files")
    parser.add_argument("output_dir", type=Path, help="Destination directory for .wav files")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ffmpeg",
        default=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        help="ffmpeg executable (default: $FFMPEG_BIN or ffmpeg)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        parser.error(f"ffmpeg executable not found: {args.ffmpeg}")

    sources = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".opus"
    )
    if not sources:
        print(f"No Opus files found under {input_dir}")
        return 0

    workers = max(1, args.workers)
    counts = {"converted": 0, "skipped": 0, "failed": 0}
    failures: list[ConversionResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                convert_one,
                source,
                destination_for(source, input_dir, output_dir),
                ffmpeg=ffmpeg,
                sample_rate=args.sample_rate,
                channels=args.channels,
                overwrite=args.overwrite,
            )
            for source in sources
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result.status] += 1
            if result.status == "failed":
                failures.append(result)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"Progress {completed}/{len(futures)}: "
                    f"converted={counts['converted']} skipped={counts['skipped']} "
                    f"failed={counts['failed']}"
                )

    for failure in failures:
        print(f"FAILED {failure.source} -> {failure.destination}: {failure.error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
