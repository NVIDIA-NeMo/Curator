#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Detect data modality from file extensions in a directory.

This script analyzes a directory to determine the primary data modality
(text, video, image, or audio) based on file extensions.

Examples:
    python detect_modality.py /data/my_dataset
    python detect_modality.py /data/my_dataset --json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

MODALITY_EXTENSIONS = {
    "text": {".jsonl", ".parquet", ".json", ".txt", ".csv", ".tsv"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"},
}

TASK_TYPES = {
    "text": "DocumentBatch",
    "video": "VideoTask",
    "image": "ImageBatch",
    "audio": "AudioBatch",
}

RECOMMENDED_STAGES = {
    "text": [
        "JsonlReader / ParquetReader",
        "HeuristicFilters (25+ filters)",
        "QualityClassifier / DomainClassifier",
        "FuzzyDeduplicationWorkflow / ExactDuplicateIdentification",
        "JsonlWriter / ParquetWriter",
    ],
    "video": [
        "VideoReader",
        "TransNetV2ClipExtractionStage / FixedStrideExtractorStage",
        "CaptionGenerationStage (Qwen VL)",
        "CosmosEmbed1EmbeddingStage / InternVideo2EmbeddingStage",
        "ClipWriterStage",
    ],
    "image": [
        "ImageReader",
        "ImageEmbeddingStage (CLIP)",
        "AestheticFilterStage",
        "NSFWFilterStage",
    ],
    "audio": [
        "AudioReader",
        "InferenceAsrNemoStage",
        "WERCalculationStage",
    ],
}


def detect_modality(path: str, max_files: int = 10000) -> dict:
    """Detect the primary data modality from a directory.

    Args:
        path: Directory path to analyze
        max_files: Maximum number of files to scan

    Returns:
        Dictionary with modality detection results
    """
    p = Path(path)

    if not p.exists():
        return {"error": f"Path does not exist: {path}", "status": "error"}

    if p.is_file():
        ext = p.suffix.lower()
        for modality, exts in MODALITY_EXTENSIONS.items():
            if ext in exts:
                return {
                    "status": "success",
                    "modality": modality,
                    "task_type": TASK_TYPES[modality],
                    "extensions": [ext],
                    "file_count": 1,
                    "recommended_stages": RECOMMENDED_STAGES[modality],
                }
        return {
            "status": "unknown",
            "modality": "unknown",
            "extensions": [ext],
            "message": f"Unknown file extension: {ext}",
        }

    # Scan directory
    extension_counts: Counter = Counter()
    file_count = 0

    for f in p.rglob("*"):
        if file_count >= max_files:
            break
        if f.is_file() and not f.name.startswith("."):
            extension_counts[f.suffix.lower()] += 1
            file_count += 1

    if not extension_counts:
        return {
            "status": "empty",
            "modality": "unknown",
            "message": "No files found in directory",
        }

    # Determine primary modality
    modality_scores: dict[str, int] = {}
    detected_extensions: dict[str, list[str]] = {}

    for modality, exts in MODALITY_EXTENSIONS.items():
        matching_exts = set(extension_counts.keys()) & exts
        if matching_exts:
            score = sum(extension_counts[e] for e in matching_exts)
            modality_scores[modality] = score
            detected_extensions[modality] = list(matching_exts)

    if not modality_scores:
        return {
            "status": "unknown",
            "modality": "unknown",
            "extensions": list(extension_counts.keys())[:10],
            "message": "Could not identify modality from file extensions",
        }

    # Find primary modality (highest file count)
    primary_modality = max(modality_scores, key=modality_scores.get)

    result = {
        "status": "success",
        "modality": primary_modality,
        "task_type": TASK_TYPES[primary_modality],
        "extensions": detected_extensions[primary_modality],
        "file_count": modality_scores[primary_modality],
        "total_files_scanned": file_count,
        "recommended_stages": RECOMMENDED_STAGES[primary_modality],
    }

    # Check for mixed modalities
    if len(modality_scores) > 1:
        result["warning"] = "Multiple modalities detected"
        result["all_modalities"] = {
            m: {"count": c, "extensions": detected_extensions[m]}
            for m, c in modality_scores.items()
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Directory or file path to analyze")
    parser.add_argument(
        "--max-files",
        type=int,
        default=10000,
        help="Maximum files to scan (default: 10000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON only (default: human-readable)",
    )

    args = parser.parse_args()
    result = detect_modality(args.path, args.max_files)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        if result["status"] == "success":
            print(f"📁 Modality: {result['modality'].upper()}")
            print(f"   Task Type: {result['task_type']}")
            print(f"   Extensions: {', '.join(result['extensions'])}")
            print(f"   Files: {result['file_count']}")
            print()
            print("📋 Recommended Stages:")
            for stage in result["recommended_stages"]:
                print(f"   • {stage}")

            if "warning" in result:
                print()
                print(f"⚠️  {result['warning']}")
                for m, info in result.get("all_modalities", {}).items():
                    print(f"   - {m}: {info['count']} files")
        elif result["status"] == "error":
            print(f"❌ Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"❓ {result.get('message', 'Unknown modality')}")
            if "extensions" in result:
                print(f"   Found extensions: {', '.join(result['extensions'][:10])}")


if __name__ == "__main__":
    main()
