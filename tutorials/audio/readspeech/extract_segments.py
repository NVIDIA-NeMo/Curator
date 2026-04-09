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
Segment Extraction Script

Reads manifest jsonl file(s) and extracts audio segments from original files.
Automatically detects the pipeline combo from the manifest schema and applies
the appropriate extraction strategy:

  Combo 1 (no VAD, no speaker):
    Extracts the full file as a single segment (start=0, end=file duration).
    Output: {original_filename}_segment_000.{format}

  Combo 2 (VAD only):
    Extracts each VAD speech segment by original_start_ms / original_end_ms.
    Output: {original_filename}_segment_{NNN}.{format}
    Segments are numbered in ascending order of start time.

  Combo 3 (speaker only):
    Extracts each speaking interval from diar_segments per speaker.
    Output: {original_filename}_speaker_{X}_segment_{NNN}.{format}
    Segments are numbered per speaker in ascending order.

  Combo 4 (VAD + speaker):
    Extracts each speaker-segment by original_start_ms / original_end_ms.
    Output: {original_filename}_speaker_{X}_segment_{NNN}.{format}
    Segments are numbered per speaker in ascending order of start time.

Input can be:
  - A single manifest.jsonl file
  - A directory containing multiple .jsonl files

Supports configurable output format: wav, flac, ogg (via soundfile).

Usage:
    python extract_segments.py --manifest manifest.jsonl --output-dir extracted/
    python extract_segments.py --manifest /path/to/result_dir/ --output-dir out/
    python extract_segments.py --manifest result_dir/ --output-dir out/ --output-format flac
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

DEFAULT_OUTPUT_FORMAT = "wav"

SOUNDFILE_FORMATS = {
    "wav": "PCM_16",
    "flac": "PCM_16",
    "ogg": "VORBIS",
}

_CSV_STRUCTURAL_KEYS = frozenset(
    {
        "filename",
        "original_file",
        "original_start_ms",
        "original_end_ms",
        "duration_ms",
        "start_sec",
        "end_sec",
        "duration",
        "segment_index",
        "speaker_id",
        "num_speakers",
        "speaking_duration",
        "diar_segments",
    }
)


def _extract_scores(entry: dict) -> dict:
    """Extract quality/filter score fields from a manifest entry.

    Returns all keys that are not structural CSV columns (timestamps,
    duration, speaker info), with float values rounded for readability.
    Since TimestampMapper already whitelist-filters the manifest output,
    anything remaining is a quality score or user-defined field.
    """
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items() if k not in _CSV_STRUCTURAL_KEYS}


def load_manifest(manifest_path: str) -> list:
    """Load a single manifest.jsonl file and return list of entries."""
    entries = []
    with open(manifest_path) as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {manifest_path}: {e}")
    return entries


def load_manifests(input_path: str, output_dir: str) -> list:
    """Load entries from a single jsonl file or a directory of jsonl files."""
    if os.path.isfile(input_path):
        return load_manifest(input_path)

    if not os.path.isdir(input_path):
        logger.error(f"Input path not found: {input_path}")
        return []

    jsonl_files = sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
    if not jsonl_files:
        logger.error(f"No .jsonl files found in {input_path}")
        return []

    logger.info(f"Found {len(jsonl_files)} jsonl files in {input_path}")

    all_entries = []
    for jf in jsonl_files:
        all_entries.extend(load_manifest(jf))

    logger.info(f"Combined {len(all_entries)} entries from {len(jsonl_files)} file(s)")

    if all_entries:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, "manifest.jsonl")
        with open(combined_path, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in all_entries)
        logger.info(f"Saved combined manifest to {combined_path}")

    return all_entries


def detect_combo(entries: list) -> int:
    """Detect which pipeline combo produced the manifest.

    Returns 2, 3, or 4.  Since TimestampMapper always emits
    ``original_start_ms``/``original_end_ms``, combos 1 and 2 are
    indistinguishable and both use timestamp-based extraction.

    Returns:
        2: segments by timestamps (combos 1 and 2)
        3: speaker diarization segments
        4: speaker-segments by timestamps
    """
    if not entries:
        return 2

    first = entries[0]
    has_speaker = "speaker_id" in first
    has_diar = "diar_segments" in first

    if has_speaker and has_diar:
        return 3
    if has_speaker:
        return 4
    return 2


def _write_segment(output_path: str, audio: np.ndarray, sample_rate: int, output_format: str) -> None:
    """Write a single audio segment to disk."""
    sf.write(output_path, audio, sample_rate, subtype=SOUNDFILE_FORMATS[output_format])


def _read_segment(filepath: str, start_ms: int, end_ms: int, sample_rate: int) -> np.ndarray:
    """Read a slice of audio from a file."""
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    audio, _ = sf.read(filepath, start=start_sample, stop=end_sample, dtype="float32")
    return audio


# ------------------------------------------------------------------
# Combos 1 & 2: extract segments by timestamps
# ------------------------------------------------------------------


def extract_segments_by_timestamps(
    entries: list, output_dir: str, output_format: str
) -> tuple[int, float, dict[str, int], list[dict]]:
    """Extract segments by original_start_ms / original_end_ms, sorted by start time."""
    by_file = defaultdict(list)
    for entry in entries:
        original_file = entry.get("original_file", "")
        by_file[original_file].append(entry)

    extracted = 0
    total_dur = 0.0
    speaker_counts: dict[str, int] = {}
    metadata_rows: list[dict] = []

    for original_file, segments in by_file.items():
        if not os.path.exists(original_file):
            logger.error(f"Original file not found: {original_file}")
            continue

        info = sf.info(original_file)
        original_name = Path(original_file).stem
        logger.info(f"\nProcessing: {original_name} ({len(segments)} segments)")

        segments.sort(key=lambda x: x.get("original_start_ms", 0))

        for i, seg in enumerate(segments):
            start_ms = seg.get("original_start_ms", 0)
            end_ms = seg.get("original_end_ms", 0)
            dur = seg.get("duration", (end_ms - start_ms) / 1000)

            out_filename = f"{original_name}_segment_{i:03d}.{output_format}"
            output_path = os.path.join(output_dir, out_filename)

            try:
                audio = _read_segment(original_file, start_ms, end_ms, info.samplerate)
                _write_segment(output_path, audio, info.samplerate, output_format)
                extracted += 1
                total_dur += dur
                logger.debug(f"  {out_filename} ({start_ms}-{end_ms}ms, {dur:.2f}s)")

                metadata_rows.append(
                    {
                        "filename": out_filename,
                        "original_file": original_file,
                        "segment_index": i,
                        "start_sec": round(start_ms / 1000, 3),
                        "end_sec": round(end_ms / 1000, 3),
                        "duration": round(dur, 3),
                        **_extract_scores(seg),
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"  Failed to extract {out_filename}: {e}")

    return extracted, total_dur, speaker_counts, metadata_rows


# ------------------------------------------------------------------
# Combo 3: speaker only -- extract each diar_segment per speaker
# ------------------------------------------------------------------


def extract_speaker_diar_segments(entries: list, output_dir: str, output_format: str) -> tuple[int, float, dict, list[dict]]:
    """Extract individual speaking intervals from diar_segments per speaker."""
    by_file = defaultdict(list)
    for entry in entries:
        original_file = entry.get("original_file", "")
        by_file[original_file].append(entry)

    extracted = 0
    total_dur = 0.0
    speaker_counts: dict[str, int] = defaultdict(int)
    metadata_rows: list[dict] = []

    for original_file, speaker_entries in by_file.items():
        if not os.path.exists(original_file):
            logger.error(f"Original file not found: {original_file}")
            continue

        info = sf.info(original_file)
        original_name = Path(original_file).stem
        logger.info(f"\nProcessing: {original_name} ({len(speaker_entries)} speakers)")

        speaker_entries.sort(key=lambda x: x.get("speaker_id", ""))

        for entry in speaker_entries:
            speaker_id = entry.get("speaker_id", "unknown")
            speaker_num = speaker_id.replace("speaker_", "") if "speaker_" in speaker_id else speaker_id
            num_speakers = entry.get("num_speakers", 0)
            diar_segments = entry.get("diar_segments", [])

            scores = _extract_scores(entry)

            if not diar_segments:
                logger.warning(f"  {speaker_id}: no diar_segments, skipping")
                continue

            diar_segments_sorted = sorted(diar_segments, key=lambda x: x[0])

            logger.info(f"  {speaker_id}: {len(diar_segments_sorted)} speaking intervals")

            for j, (start_sec, end_sec) in enumerate(diar_segments_sorted):
                start_ms = int(start_sec * 1000)
                end_ms = int(end_sec * 1000)
                dur = end_sec - start_sec

                out_filename = f"{original_name}_speaker_{speaker_num}_segment_{j:03d}.{output_format}"
                output_path = os.path.join(output_dir, out_filename)

                try:
                    audio = _read_segment(original_file, start_ms, end_ms, info.samplerate)
                    _write_segment(output_path, audio, info.samplerate, output_format)
                    extracted += 1
                    total_dur += dur
                    speaker_counts[speaker_id] += 1
                    logger.debug(f"    {out_filename} ({start_sec:.2f}-{end_sec:.2f}s, {dur:.2f}s)")

                    metadata_rows.append(
                        {
                            "filename": out_filename,
                            "original_file": original_file,
                            "speaker_id": speaker_id,
                            "num_speakers": num_speakers,
                            "segment_index": j,
                            "start_sec": round(start_sec, 3),
                            "end_sec": round(end_sec, 3),
                            "duration": round(dur, 3),
                            **scores,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"    Failed to extract {out_filename}: {e}")

    return extracted, total_dur, speaker_counts, metadata_rows


# ------------------------------------------------------------------
# Combo 4: VAD + speaker -- extract each speaker-segment by timestamps
# ------------------------------------------------------------------


def extract_speaker_segments_by_timestamps(entries: list, output_dir: str, output_format: str) -> tuple[int, float, dict, list[dict]]:
    """Extract speaker-segments using original_start_ms / original_end_ms."""
    by_file = defaultdict(list)
    for entry in entries:
        original_file = entry.get("original_file", "")
        by_file[original_file].append(entry)

    extracted = 0
    total_dur = 0.0
    speaker_counts: dict[str, int] = defaultdict(int)
    metadata_rows: list[dict] = []

    for original_file, segments in by_file.items():
        if not os.path.exists(original_file):
            logger.error(f"Original file not found: {original_file}")
            continue

        info = sf.info(original_file)
        original_name = Path(original_file).stem
        logger.info(f"\nProcessing: {original_name} ({len(segments)} speaker-segments)")

        segments.sort(key=lambda x: (x.get("speaker_id", ""), x.get("original_start_ms", 0)))

        per_speaker_count: dict[str, int] = defaultdict(int)

        for seg in segments:
            speaker_id = seg.get("speaker_id", "unknown")
            speaker_num = speaker_id.replace("speaker_", "") if "speaker_" in speaker_id else speaker_id
            num_speakers = seg.get("num_speakers", 0)
            start_ms = seg.get("original_start_ms", 0)
            end_ms = seg.get("original_end_ms", 0)
            dur = seg.get("duration", (end_ms - start_ms) / 1000)

            seg_idx = per_speaker_count[speaker_id]
            per_speaker_count[speaker_id] += 1

            out_filename = f"{original_name}_speaker_{speaker_num}_segment_{seg_idx:03d}.{output_format}"
            output_path = os.path.join(output_dir, out_filename)

            try:
                audio = _read_segment(original_file, start_ms, end_ms, info.samplerate)
                _write_segment(output_path, audio, info.samplerate, output_format)
                extracted += 1
                total_dur += dur
                speaker_counts[speaker_id] += 1
                logger.debug(f"  {out_filename} ({start_ms}-{end_ms}ms, {dur:.2f}s)")

                metadata_rows.append(
                    {
                        "filename": out_filename,
                        "original_file": original_file,
                        "speaker_id": speaker_id,
                        "num_speakers": num_speakers,
                        "segment_index": seg_idx,
                        "start_sec": round(start_ms / 1000, 3),
                        "end_sec": round(end_ms / 1000, 3),
                        "duration": round(dur, 3),
                        **_extract_scores(seg),
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"  Failed to extract {out_filename}: {e}")

    return extracted, total_dur, speaker_counts, metadata_rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def _write_metadata_csv(output_dir: str, metadata_rows: list[dict]) -> str:
    """Write metadata.csv from collected metadata rows."""
    if not metadata_rows:
        return ""

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in metadata_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path = os.path.join(output_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(metadata_rows)

    return csv_path


def extract_segments(input_path: str, output_dir: str, output_format: str = DEFAULT_OUTPUT_FORMAT) -> None:
    """Extract segments from original audio files based on manifest."""
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading manifest: {input_path}")
    entries = load_manifests(input_path, output_dir)
    logger.info(f"Found {len(entries)} entries total")

    if not entries:
        logger.error("No entries found in manifest")
        return

    combo = detect_combo(entries)
    combo_names = {
        2: "Segments by timestamps",
        3: "Speaker diarization segments",
        4: "Speaker-segments by timestamps",
    }
    logger.info(f"Detected: {combo_names[combo]}")

    extractors = {
        2: extract_segments_by_timestamps,
        3: extract_speaker_diar_segments,
        4: extract_speaker_segments_by_timestamps,
    }
    total_extracted, total_dur, speaker_counts, metadata_rows = extractors[combo](entries, output_dir, output_format)

    csv_path = _write_metadata_csv(output_dir, metadata_rows)

    summary = {
        "manifest_path": input_path,
        "output_dir": output_dir,
        "total_segments": total_extracted,
        "total_duration_sec": round(total_dur, 2),
        "output_format": output_format,
    }
    if speaker_counts:
        summary["segments_by_speaker"] = dict(speaker_counts)

    summary_path = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Combo: {combo_names[combo]}")
    logger.info(f"  Total segments: {total_extracted}")
    logger.info(f"  Total duration: {total_dur:.2f}s ({total_dur / 60:.1f} min)")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Format: {output_format}")
    if speaker_counts:
        logger.info("  Segments by speaker:")
        for speaker, count in sorted(speaker_counts.items()):
            logger.info(f"    {speaker}: {count} segments")
    if csv_path:
        logger.info(f"  Metadata CSV: {csv_path}")
    logger.info(f"  Summary: {summary_path}")


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
    extract_segments(input_path=args.manifest, output_dir=args.output_dir, output_format=args.output_format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
