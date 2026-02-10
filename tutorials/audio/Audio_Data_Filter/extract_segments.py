#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Segment Extraction Script

Reads a manifest.jsonl file and extracts audio segments from original files.
Each segment is saved with naming convention: {original_filename}_speaker_{x}_segment_{y}.{format}

Supports multiple input formats: wav, mp3, flac, ogg, m4a, aac, wma, opus, webm
Supports configurable output format.

Usage:
    python extract_segments.py --manifest manifest.jsonl --output-dir extracted_segments/
    python extract_segments.py --manifest manifest.jsonl --output-dir out/ --output-format flac
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from loguru import logger
from pydub import AudioSegment

# Default output format
DEFAULT_OUTPUT_FORMAT = "wav"


def load_manifest(manifest_path: str) -> list:
    """Load manifest.jsonl file and return list of segment entries."""
    segments = []
    with open(manifest_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                segment = json.loads(line)
                segments.append(segment)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
    return segments


def extract_segments(manifest_path: str, output_dir: str, output_format: str = DEFAULT_OUTPUT_FORMAT):
    """
    Extract segments from original audio files based on manifest.
    
    Args:
        manifest_path: Path to manifest.jsonl
        output_dir: Directory to save extracted segments
        output_format: Output audio format (wav, mp3, flac, ogg, m4a). Default: wav
    
    Note: Non-wav output formats require ffmpeg to be installed on the system.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load manifest
    logger.info(f"Loading manifest: {manifest_path}")
    segments = load_manifest(manifest_path)
    logger.info(f"Found {len(segments)} segments in manifest")
    
    if not segments:
        logger.error("No segments found in manifest")
        return
    
    # Group segments by original file
    segments_by_file = defaultdict(list)
    for seg in segments:
        original_file = seg.get('original_file')
        if original_file:
            segments_by_file[original_file].append(seg)
    
    logger.info(f"Segments span {len(segments_by_file)} original file(s)")
    
    # Track statistics
    total_extracted = 0
    total_duration_sec = 0
    speaker_counts = defaultdict(int)
    
    # Process each original file
    for original_file, file_segments in segments_by_file.items():
        if not os.path.exists(original_file):
            logger.error(f"Original file not found: {original_file}")
            continue
        
        # Get original filename without extension
        original_name = Path(original_file).stem
        
        logger.info(f"\nProcessing: {original_name}")
        logger.info(f"  Original file: {original_file}")
        logger.info(f"  Segments to extract: {len(file_segments)}")
        
        # Load original audio
        try:
            audio = AudioSegment.from_file(original_file)
            logger.info(f"  Original duration: {len(audio)/1000:.2f}s")
        except Exception as e:
            logger.error(f"  Failed to load audio: {e}")
            continue
        
        # Sort segments by start time for consistent ordering
        file_segments.sort(key=lambda x: (x.get('speaker_id', ''), x.get('original_start_ms', 0)))
        
        # Track segment numbers per speaker for this file
        speaker_segment_counts = defaultdict(int)
        
        # Extract each segment
        for seg in file_segments:
            start_ms = seg.get('original_start_ms', 0)
            end_ms = seg.get('original_end_ms', 0)
            speaker_id = seg.get('speaker_id', 'unknown')
            duration_sec = seg.get('duration_sec', (end_ms - start_ms) / 1000)
            
            # Get segment number for this speaker
            segment_num = speaker_segment_counts[speaker_id]
            speaker_segment_counts[speaker_id] += 1
            
            # Create output filename
            # Format: originalfilename_speaker_0_segment_000.{format}
            speaker_num = speaker_id.replace('speaker_', '') if 'speaker_' in speaker_id else speaker_id
            output_filename = f"{original_name}_speaker_{speaker_num}_segment_{segment_num:03d}.{output_format}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Extract segment
            try:
                segment_audio = audio[start_ms:end_ms]
                
                # Export segment with configurable format
                segment_audio.export(output_path, format=output_format)
                
                total_extracted += 1
                total_duration_sec += duration_sec
                speaker_counts[speaker_id] += 1
                
                logger.debug(f"  Extracted: {output_filename} ({duration_sec:.2f}s)")
                
            except Exception as e:
                logger.error(f"  Failed to extract segment {segment_num}: {e}")
        
        logger.info(f"  Extracted {sum(speaker_segment_counts.values())} segments from this file")
    
    # Save extraction summary
    summary = {
        'manifest_path': manifest_path,
        'output_dir': output_dir,
        'total_segments': total_extracted,
        'total_duration_sec': round(total_duration_sec, 2),
        'segments_by_speaker': dict(speaker_counts),
        'original_files_processed': len(segments_by_file),
    }
    
    summary_path = os.path.join(output_dir, 'extraction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total segments extracted: {total_extracted}")
    logger.info(f"Total duration: {total_duration_sec:.2f}s ({total_duration_sec/60:.1f} min)")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nSegments by speaker:")
    for speaker, count in sorted(speaker_counts.items()):
        logger.info(f"  {speaker}: {count} segments")
    logger.info(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio segments from original files based on manifest"
    )
    parser.add_argument(
        "--manifest", "-m",
        required=True,
        help="Path to manifest.jsonl file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for extracted segments"
    )
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Output audio format (default: wav). Note: non-wav formats require ffmpeg."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    
    # Validate manifest exists
    if not os.path.exists(args.manifest):
        logger.error(f"Manifest file not found: {args.manifest}")
        return 1
    
    # Extract segments
    logger.info(f"Output format: {args.output_format}")
    extract_segments(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        output_format=args.output_format
    )
    
    return 0


if __name__ == "__main__":
    exit(main())