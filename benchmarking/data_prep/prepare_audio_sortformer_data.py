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

"""One-time data preparation for the audio Sortformer benchmark."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

DEFAULT_CACHE_DIR = "/tmp/curator/audio_sortformer_cache"  # noqa: S108
AMI_HF_REPO_ID = "diarizers-community/ami"
AMI_CONFIG = "sdm"
AMI_SPLITS = ("validation", "test")
AMI_SPLIT_NUM_ROWS = {"validation": 18, "test": 16}
EXPECTED_AUDIO_FILENAMES = tuple(
    f"ami_sdm_{split}_{index:03d}.wav" for split in AMI_SPLITS for index in range(AMI_SPLIT_NUM_ROWS[split])
)
MODEL_HF_REPO_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"Manifest has invalid JSON on line {line_number}: {e}"
                raise RuntimeError(msg) from e
            if not isinstance(row, dict):
                msg = f"Manifest line {line_number} is not a JSON object"
                raise TypeError(msg)
            rows.append(row)
    if not rows:
        msg = f"Manifest contains no data rows: {manifest_path}"
        raise RuntimeError(msg)
    return rows


def _validate_manifest_contract(rows: list[dict[str, Any]], label: str) -> None:
    if len(rows) != len(EXPECTED_AUDIO_FILENAMES):
        msg = f"{label} must contain exactly {len(EXPECTED_AUDIO_FILENAMES)} rows, found {len(rows)}"
        raise RuntimeError(msg)

    expected_filenames = set(EXPECTED_AUDIO_FILENAMES)
    seen_filenames: set[str] = set()
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        audio_filepath = row.get("audio_filepath")
        if not isinstance(audio_filepath, str) or not audio_filepath:
            msg = f"{label} line {line_number} must contain audio_filepath"
            raise RuntimeError(msg)
        filename = Path(audio_filepath).name
        if filename not in expected_filenames:
            msg = f"{label} line {line_number} references unexpected audio file {filename!r}"
            raise RuntimeError(msg)
        if filename in seen_filenames:
            msg = f"{label} contains duplicate audio file {filename!r}"
            raise RuntimeError(msg)
        seen_filenames.add(filename)

        audio_item_id = row.get("audio_item_id")
        if not isinstance(audio_item_id, str) or not audio_item_id:
            msg = f"{label} line {line_number} must contain a nonempty audio_item_id"
            raise RuntimeError(msg)
        if audio_item_id in seen_ids:
            msg = f"{label} contains duplicate audio_item_id {audio_item_id!r}"
            raise RuntimeError(msg)
        seen_ids.add(audio_item_id)

        duration = row.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            msg = f"{label} line {line_number} must contain a finite positive duration"
            raise RuntimeError(msg)

    if seen_filenames != expected_filenames:
        msg = f"{label} is missing expected audio files: {sorted(expected_filenames - seen_filenames)}"
        raise RuntimeError(msg)


def _write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row) + "\n")


def verify_dataset(output_path: Path) -> bool:
    manifest_path = output_path / "manifest.jsonl"
    audio_dir = output_path / "audio"
    if not manifest_path.is_file() or not audio_dir.is_dir():
        logger.error(f"Expected manifest.jsonl and audio/ under {output_path}")
        return False

    missing_audio = [
        audio_dir / filename for filename in EXPECTED_AUDIO_FILENAMES if not (audio_dir / filename).is_file()
    ]
    if missing_audio:
        logger.error(f"Missing expected audio files: {', '.join(str(path) for path in missing_audio)}")
        return False

    try:
        rows = _load_manifest_rows(manifest_path)
        _validate_manifest_contract(rows, str(manifest_path))
    except Exception as e:
        logger.error(f"Manifest validation failed: {e}")
        return False

    total_duration_hours = sum(float(row["duration"]) for row in rows) / 3600
    logger.info(f"Verified {len(rows)} unique AMI SDM files ({total_duration_hours:.3f} audio hours)")
    return True


def verify_model(model_path: Path) -> bool:
    if not model_path.is_file() or model_path.stat().st_size == 0:
        logger.error(f"Sortformer model is missing or empty: {model_path}")
        return False
    logger.info(f"Verified local Sortformer checkpoint: {model_path}")
    return True


def _write_audio_row(audio: object, target_path: Path) -> float:
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        msg = "Expected embedded WAV bytes in the AMI row"
        raise TypeError(msg)
    audio_bytes = audio["bytes"]
    info = sf.info(BytesIO(audio_bytes))
    target_path.write_bytes(audio_bytes)
    return float(info.frames) / info.samplerate


def stage_dataset(output_path: Path, cache_dir: str) -> None:
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for split in AMI_SPLITS:
        dataset = load_dataset(
            AMI_HF_REPO_ID,
            AMI_CONFIG,
            split=split,
            cache_dir=cache_dir,
            streaming=True,
        ).cast_column("audio", Audio(decode=False))
        expected_rows = AMI_SPLIT_NUM_ROWS[split]
        split_rows = 0
        for row_index, row in enumerate(dataset):
            if row_index >= expected_rows:
                msg = f"AMI SDM {split} contains more than {expected_rows} rows"
                raise RuntimeError(msg)
            audio_item_id = f"ami_sdm_{split}_{row_index:03d}"
            filename = f"{audio_item_id}.wav"
            duration = _write_audio_row(row["audio"], audio_dir / filename)
            manifest_rows.append(
                {
                    "audio_filepath": f"audio/{filename}",
                    "audio_item_id": audio_item_id,
                    "duration": duration,
                }
            )
            split_rows += 1
        if split_rows != expected_rows:
            msg = f"Expected {expected_rows} AMI SDM {split} rows, found {split_rows}"
            raise RuntimeError(msg)

    _write_manifest(manifest_rows, output_path / "manifest.jsonl")
    logger.success(f"Dataset ready: {len(manifest_rows)} unique AMI SDM meetings")


def stage_model(model_path: Path, cache_dir: str) -> None:
    downloaded_path = hf_hub_download(
        repo_id=MODEL_HF_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir=cache_dir,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    output_path = args.output_path.resolve()
    model_path = args.model_output_path.resolve()
    if args.verify_only:
        return 0 if verify_dataset(output_path) and verify_model(model_path) else 1

    dataset_ready = verify_dataset(output_path)
    if not dataset_ready:
        stage_dataset(output_path, args.cache_dir)
        dataset_ready = verify_dataset(output_path)

    model_ready = verify_model(model_path)
    if not model_ready:
        stage_model(model_path, args.cache_dir)
        model_ready = verify_model(model_path)
    return 0 if dataset_ready and model_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
