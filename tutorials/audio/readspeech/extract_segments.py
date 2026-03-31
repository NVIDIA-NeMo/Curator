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
Each segment is saved with naming convention:
  With speaker separation:    {original_filename}_speaker_{x}_segment_{y}.{format}
  Without speaker separation: {original_filename}_segment_{y}.{format}

Input can be:
  - A single manifest.jsonl file
  - A directory containing multiple .jsonl files (from pipeline executor output)

When given a directory, all .jsonl files are combined into a single
manifest.jsonl in the output directory with escaped paths (\\/) cleaned up.

Supports configurable output format:
  wav/flac/ogg via soundfile (no extra dependencies)
  mp3/m4a via pydub + ffmpeg (optional, install separately)

Usage:
    python extract_segments.py --manifest manifest.jsonl --output-dir extracted_segments/
    python extract_segments.py --manifest /path/to/result_dir/ --output-dir out/
    python extract_segments.py --manifest /path/to/result_dir/ --output-dir out/ --output-format flac
"""

import argparse
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

PYDUB_FORMATS = {"mp3", "m4a"}


def load_manifest(manifest_path: str) -> list:
    """Load a single manifest.jsonl file and return list of segment entries."""
    segments = []
    with open(manifest_path) as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                segment = json.loads(line)
                segments.append(segment)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {manifest_path}: {e}")
    return segments


def load_manifests(input_path: str, output_dir: str) -> list:
    """
    Load segments from a single jsonl file or a directory of jsonl files.

    When input_path is a directory, all .jsonl files are combined and a
    merged manifest.jsonl is saved in output_dir with escaped paths fixed.
    """
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

    all_segments = []
    skipped_files = 0
    for jf in jsonl_files:
        segs = load_manifest(jf)
        if not segs:
            skipped_files += 1
            continue
        all_segments.extend(segs)

    if skipped_files:
        logger.info(f"Skipped {skipped_files} empty jsonl file(s)")
    logger.info(f"Combined {len(all_segments)} segments from {len(jsonl_files) - skipped_files} file(s)")

    if all_segments:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, "manifest.jsonl")
        with open(combined_path, "w") as f:
            f.writelines(json.dumps(seg) + "\n" for seg in all_segments)
        logger.info(f"Saved combined manifest to {combined_path}")

    return all_segments


def _write_segment(
    output_path: str, segment_audio: np.ndarray, sample_rate: int, audio_data: np.ndarray, output_format: str
) -> None:
    """Write a single audio segment to disk."""
    if output_format in SOUNDFILE_FORMATS:
        sf.write(output_path, segment_audio, sample_rate, subtype=SOUNDFILE_FORMATS[output_format])
    else:
        from pydub import AudioSegment

        pcm16 = (segment_audio * 32767).astype(np.int16)
        audio_seg = AudioSegment(
            data=pcm16.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1 if audio_data.ndim == 1 else audio_data.shape[1],
        )
        audio_seg.export(output_path, format=output_format)


def _process_file_segments(
    original_file: str,
    file_segments: list,
    output_dir: str,
    output_format: str,
    speaker_counts: dict,
) -> tuple[int, float]:
    """Process all segments for a single original file. Returns (extracted_count, duration_sec)."""
    original_name = Path(original_file).stem
    logger.info(f"\nProcessing: {original_name}")
    logger.info(f"  Original file: {original_file}")
    logger.info(f"  Segments to extract: {len(file_segments)}")

    try:
        audio_data, sample_rate = sf.read(original_file, dtype="float32")
        total_samples = len(audio_data) if audio_data.ndim == 1 else audio_data.shape[0]
        logger.info(f"  Original duration: {total_samples / sample_rate:.2f}s")
    except Exception as e:  # noqa: BLE001
        logger.error(f"  Failed to load audio: {e}")
        return 0, 0.0

    has_speakers = any("speaker_id" in seg for seg in file_segments)
    if has_speakers:
        file_segments.sort(key=lambda x: (x.get("speaker_id", ""), x.get("original_start_ms", 0)))
    else:
        file_segments.sort(key=lambda x: x.get("original_start_ms", 0))

    segment_counts = defaultdict(int)
    extracted = 0
    duration_total = 0.0

    for seg in file_segments:
        start_ms = seg.get("original_start_ms", 0)
        end_ms = seg.get("original_end_ms", 0)
        speaker_id = seg.get("speaker_id")
        duration_sec = seg.get("duration_sec", (end_ms - start_ms) / 1000)

        count_key = speaker_id if speaker_id else "__all__"
        segment_num = segment_counts[count_key]
        segment_counts[count_key] += 1

        if speaker_id:
            speaker_num = speaker_id.replace("speaker_", "") if "speaker_" in speaker_id else speaker_id
            output_filename = f"{original_name}_speaker_{speaker_num}_segment_{segment_num:03d}.{output_format}"
        else:
            output_filename = f"{original_name}_segment_{segment_num:03d}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            segment_audio = (
                audio_data[start_sample:end_sample] if audio_data.ndim == 1 else audio_data[start_sample:end_sample, :]
            )
            _write_segment(output_path, segment_audio, sample_rate, audio_data, output_format)
            extracted += 1
            duration_total += duration_sec
            if speaker_id:
                speaker_counts[speaker_id] += 1
            logger.debug(f"  Extracted: {output_filename} ({duration_sec:.2f}s)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"  Failed to extract segment {segment_num}: {e}")

    logger.info(f"  Extracted {sum(segment_counts.values())} segments from this file")
    return extracted, duration_total


def extract_segments(input_path: str, output_dir: str, output_format: str = DEFAULT_OUTPUT_FORMAT) -> None:
    """
    Extract segments from original audio files based on manifest.

    Args:
        input_path: Path to manifest.jsonl file or directory of .jsonl files
        output_dir: Directory to save extracted segments
        output_format: Output audio format (wav, flac, ogg, mp3, m4a). Default: wav.
            wav/flac/ogg use soundfile (no extra deps). mp3/m4a require pydub + ffmpeg.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading manifest: {input_path}")
    segments = load_manifests(input_path, output_dir)
    logger.info(f"Found {len(segments)} segments total")

    if not segments:
        logger.error("No segments found in manifest")
        return

    segments_by_file = defaultdict(list)
    for seg in segments:
        original_file = seg.get("original_file")
        if original_file:
            segments_by_file[original_file].append(seg)

    logger.info(f"Segments span {len(segments_by_file)} original file(s)")

    total_extracted = 0
    total_duration_sec = 0.0
    speaker_counts: dict[str, int] = defaultdict(int)

    for original_file, file_segments in segments_by_file.items():
        if not os.path.exists(original_file):
            logger.error(f"Original file not found: {original_file}")
            continue
        extracted, duration = _process_file_segments(
            original_file,
            file_segments,
            output_dir,
            output_format,
            speaker_counts,
        )
        total_extracted += extracted
        total_duration_sec += duration

    summary = {
        "manifest_path": input_path,
        "output_dir": output_dir,
        "total_segments": total_extracted,
        "total_duration_sec": round(total_duration_sec, 2),
        "segments_by_speaker": dict(speaker_counts),
        "original_files_processed": len(segments_by_file),
    }

    summary_path = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total segments extracted: {total_extracted}")
    logger.info(f"Total duration: {total_duration_sec:.2f}s ({total_duration_sec / 60:.1f} min)")
    logger.info(f"Output directory: {output_dir}")
    if speaker_counts:
        logger.info("\nSegments by speaker:")
        for speaker, count in sorted(speaker_counts.items()):
            logger.info(f"  {speaker}: {count} segments")
    logger.info(f"\nSummary saved to: {summary_path}")


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
        choices=["wav", "flac", "ogg", "mp3", "m4a"],
        help="Output audio format (default: wav). mp3/m4a require pydub + ffmpeg.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
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
