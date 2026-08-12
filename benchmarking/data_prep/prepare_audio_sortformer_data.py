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

"""Stage public AMI SDM validation/test audio and Sortformer for nightly benchmarks.

Both sources are public and ungated. Revisions, expected sizes, and the model
checksum are pinned so nightly runs use an auditable input without credentials
or runtime downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audio_sortformer_contract import (
    AUDIO_SAMPLE_RATE,
    DATASET_AUDIO_SHA256,
    DATASET_CONFIG,
    DATASET_HF_REPO_ID,
    DATASET_LICENSE,
    DATASET_NUM_ROWS,
    DATASET_REVISION,
    DATASET_SOURCE_DOWNLOAD_BYTES,
    DATASET_SPLIT_NUM_ROWS,
    DATASET_SPLIT_SOURCE_DOWNLOAD_BYTES,
    DATASET_SPLITS,
    EXPECTED_AUDIO_FILENAMES,
    MANIFEST_FILENAME,
    MAX_DATASET_DURATION_S,
    MIN_DATASET_DURATION_S,
    MODEL_FILENAME,
    MODEL_HF_REPO_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    MONO_CHANNELS,
    REFERENCE_ANNOTATIONS_SHA256,
    SOURCE_METADATA_FILENAME,
    reference_annotations_sha256,
    sha256,
    source_metadata,
    validate_model,
    validate_reference_annotations,
)

DEFAULT_CACHE_DIR = "/tmp/curator/audio_sortformer_cache"  # noqa: S108
DEFAULT_CONTAINER_DATA_PATH = "/datasets/audio_sortformer_ami_sdm"


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


def _validate_manifest_contract(rows: list[dict[str, Any]], label: str) -> None:  # noqa: C901
    if len(rows) != DATASET_NUM_ROWS:
        msg = f"{label} must contain exactly {DATASET_NUM_ROWS} rows, found {len(rows)}"
        raise RuntimeError(msg)

    expected_basenames = set(EXPECTED_AUDIO_FILENAMES)
    seen_basenames: set[str] = set()
    seen_ids: set[str] = set()
    seen_sessions: set[str] = set()
    total_duration_s = 0.0

    for line_number, row in enumerate(rows, start=1):
        audio_filepath = row.get("audio_filepath")
        if not isinstance(audio_filepath, str) or not audio_filepath:
            msg = f"{label} line {line_number} must contain audio_filepath"
            raise RuntimeError(msg)
        audio_basename = Path(audio_filepath).name
        if audio_basename not in expected_basenames:
            msg = f"{label} line {line_number} references unexpected audio file {audio_basename!r}"
            raise RuntimeError(msg)
        if audio_basename in seen_basenames:
            msg = f"{label} contains duplicate audio file {audio_basename!r}"
            raise RuntimeError(msg)
        seen_basenames.add(audio_basename)

        audio_item_id = row.get("audio_item_id")
        if not isinstance(audio_item_id, str) or not audio_item_id:
            msg = f"{label} line {line_number} must contain a nonempty audio_item_id"
            raise RuntimeError(msg)
        if audio_item_id in seen_ids:
            msg = f"{label} contains duplicate audio_item_id {audio_item_id!r}"
            raise RuntimeError(msg)
        seen_ids.add(audio_item_id)

        session_name = row.get("session_name")
        if not isinstance(session_name, str) or not session_name:
            msg = f"{label} line {line_number} must contain a nonempty session_name"
            raise RuntimeError(msg)
        if session_name in seen_sessions:
            msg = f"{label} contains duplicate session_name {session_name!r}"
            raise RuntimeError(msg)
        seen_sessions.add(session_name)

        duration = row.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            msg = f"{label} line {line_number} must contain a finite positive duration"
            raise RuntimeError(msg)
        total_duration_s += float(duration)
        validate_reference_annotations(row, float(duration), f"{label} line {line_number}")

    if seen_basenames != expected_basenames:
        missing = sorted(expected_basenames - seen_basenames)
        msg = f"{label} is missing expected audio files: {missing}"
        raise RuntimeError(msg)
    if not MIN_DATASET_DURATION_S <= total_duration_s <= MAX_DATASET_DURATION_S:
        msg = (
            f"{label} total duration {total_duration_s:.3f}s is outside the expected public AMI SDM range "
            f"[{MIN_DATASET_DURATION_S:.0f}, {MAX_DATASET_DURATION_S:.0f}]s"
        )
        raise RuntimeError(msg)


def _verify_dataset_or_raise(output_path: Path) -> tuple[list[dict[str, Any]], float]:
    manifest_path = output_path / MANIFEST_FILENAME
    audio_dir = output_path / "audio"
    metadata_path = output_path / SOURCE_METADATA_FILENAME
    if not manifest_path.is_file() or not audio_dir.is_dir() or not metadata_path.is_file():
        msg = f"Expected manifest, audio directory, and source metadata under {output_path}"
        raise FileNotFoundError(msg)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != source_metadata():
        msg = f"Source metadata does not match the pinned public inputs: {metadata_path}"
        raise RuntimeError(msg)

    rows = _load_manifest_rows(manifest_path)
    _validate_manifest_contract(rows, str(manifest_path))
    actual_reference_sha256 = reference_annotations_sha256(rows)
    if actual_reference_sha256 != REFERENCE_ANNOTATIONS_SHA256:
        msg = (
            "Staged Sortformer reference annotation SHA-256 mismatch: "
            f"expected {REFERENCE_ANNOTATIONS_SHA256}, found {actual_reference_sha256}"
        )
        raise RuntimeError(msg)
    rows_by_basename = {Path(row["audio_filepath"]).name: row for row in rows}

    actual_audio_files = {path.name for path in audio_dir.glob("*.wav")}
    expected_audio_files = set(EXPECTED_AUDIO_FILENAMES)
    if actual_audio_files != expected_audio_files:
        missing = sorted(expected_audio_files - actual_audio_files)
        unexpected = sorted(actual_audio_files - expected_audio_files)
        msg = f"Staged audio mismatch; missing={missing}, unexpected={unexpected}"
        raise RuntimeError(msg)

    total_duration_s = 0.0
    for audio_filename in EXPECTED_AUDIO_FILENAMES:
        audio_path = audio_dir / audio_filename
        info = sf.info(audio_path)
        if info.samplerate != AUDIO_SAMPLE_RATE or info.channels != MONO_CHANNELS:
            msg = f"Expected mono 16 kHz WAV, found {info.channels} channels at {info.samplerate} Hz: {audio_path}"
            raise RuntimeError(msg)
        duration_s = float(info.frames) / info.samplerate
        manifest_duration_s = float(rows_by_basename[audio_filename]["duration"])
        if not math.isclose(duration_s, manifest_duration_s, abs_tol=1e-3):
            msg = (
                f"Manifest duration {manifest_duration_s:.6f}s does not match WAV duration "
                f"{duration_s:.6f}s: {audio_path}"
            )
            raise RuntimeError(msg)
        total_duration_s += duration_s
        expected_sha256 = DATASET_AUDIO_SHA256[audio_filename]
        actual_sha256 = sha256(audio_path)
        if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
            msg = f"Staged audio SHA-256 mismatch for {audio_path}: expected {expected_sha256}, found {actual_sha256}"
            raise RuntimeError(msg)

    if not MIN_DATASET_DURATION_S <= total_duration_s <= MAX_DATASET_DURATION_S:
        msg = f"Staged audio duration is outside the expected public AMI SDM range: {total_duration_s:.3f}s"
        raise RuntimeError(msg)
    return rows, total_duration_s


def verify_dataset(output_path: Path) -> bool:
    try:
        rows, total_duration_s = _verify_dataset_or_raise(output_path)
    except Exception as e:
        logger.error(f"Sortformer dataset validation failed: {e}")
        return False

    logger.info("=" * 60)
    logger.info("Audio Sortformer Dataset Verification")
    logger.info("=" * 60)
    logger.info(f"  Public source: {DATASET_HF_REPO_ID}@{DATASET_REVISION}")
    logger.info(f"  License:       {DATASET_LICENSE}")
    logger.info(f"  Source bytes:  {DATASET_SOURCE_DOWNLOAD_BYTES:,}")
    logger.info(f"  Manifest rows: {len(rows)}")
    logger.info(f"  Audio hours:   {total_duration_s / 3600:.3f}")
    logger.info("=" * 60)
    return True


def verify_model(model_output_path: Path) -> bool:
    try:
        validate_model(model_output_path)
    except Exception as e:
        logger.error(f"Sortformer model validation failed: {e}")
        return False
    logger.info("=" * 60)
    logger.info("Audio Sortformer Model Verification")
    logger.info("=" * 60)
    logger.info(f"  Public source: {MODEL_HF_REPO_ID}@{MODEL_REVISION}")
    logger.info(f"  Model path:    {model_output_path}")
    logger.info(f"  Size:          {MODEL_SIZE_BYTES:,} bytes")
    logger.info(f"  SHA-256:       {MODEL_SHA256}")
    logger.info("=" * 60)
    return True


def _write_audio_row(audio: object, target_path: Path) -> float:
    if not isinstance(audio, dict):
        msg = f"Expected a decode-disabled datasets Audio value, got {type(audio).__name__}"
        raise TypeError(msg)
    audio_bytes = audio.get("bytes")
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        msg = "Expected embedded WAV bytes in the pinned AMI Parquet row"
        raise RuntimeError(msg)
    info = sf.info(BytesIO(audio_bytes))
    if info.samplerate != AUDIO_SAMPLE_RATE or info.channels != MONO_CHANNELS or info.subtype != "PCM_16":
        msg = (
            f"Expected mono 16 kHz PCM-16 AMI audio, found {info.channels} channels at "
            f"{info.samplerate} Hz with subtype {info.subtype}"
        )
        raise RuntimeError(msg)

    target_path.write_bytes(audio_bytes)
    return float(info.frames) / info.samplerate


def stage_dataset(output_path: Path, cache_dir: str, container_data_path: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}-staging-", dir=output_path.parent) as staging_dir:
        staging_path = Path(staging_dir)
        _stage_dataset(staging_path, cache_dir, container_data_path)
        _verify_dataset_or_raise(staging_path)
        if output_path.exists():
            shutil.rmtree(output_path)
        staging_path.replace(output_path)


def _stage_dataset(output_path: Path, cache_dir: str, container_data_path: str) -> None:
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True)

    logger.info("=" * 60)
    logger.info("Audio Sortformer Public AMI SDM Download")
    logger.info(f"Repo:       {DATASET_HF_REPO_ID}@{DATASET_REVISION}")
    logger.info(f"Config:     {DATASET_CONFIG}")
    logger.info(
        f"Splits:     {', '.join(DATASET_SPLITS)} ({DATASET_NUM_ROWS} rows, {DATASET_SOURCE_DOWNLOAD_BYTES:,} bytes)"
    )
    logger.info(f"Staging to: {output_path}")
    logger.info("=" * 60)

    manifest_rows: list[dict[str, Any]] = []
    audio_container_dir = Path(container_data_path) / "audio"

    for split in DATASET_SPLITS:
        expected_rows = DATASET_SPLIT_NUM_ROWS[split]
        logger.info(
            f"Loading {split}: {expected_rows} rows, {DATASET_SPLIT_SOURCE_DOWNLOAD_BYTES[split]:,} source bytes"
        )
        dataset = load_dataset(
            DATASET_HF_REPO_ID,
            DATASET_CONFIG,
            split=split,
            revision=DATASET_REVISION,
            cache_dir=cache_dir,
            streaming=True,
            token=False,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLE_RATE, decode=False))
        split_rows = 0
        for row_index, row in enumerate(dataset):
            if row_index >= expected_rows:
                msg = f"Pinned AMI SDM {split} split contains more than the expected {expected_rows} rows"
                raise RuntimeError(msg)
            audio_item_id = f"ami_sdm_{split}_{row_index:03d}"
            target_audio_path = audio_dir / f"{audio_item_id}.wav"
            logger.info(f"Staging {audio_item_id}")
            duration_s = _write_audio_row(row["audio"], target_audio_path)
            manifest_rows.append(
                {
                    "audio_filepath": str(audio_container_dir / target_audio_path.name),
                    "audio_item_id": audio_item_id,
                    "session_name": audio_item_id,
                    "duration": duration_s,
                    "timestamps_start": [float(value) for value in row["timestamps_start"]],
                    "timestamps_end": [float(value) for value in row["timestamps_end"]],
                    "speakers": [str(value) for value in row["speakers"]],
                }
            )
            split_rows += 1
        if split_rows != expected_rows:
            msg = f"Pinned AMI SDM {split} split must contain {expected_rows} rows, found {split_rows}"
            raise RuntimeError(msg)

    split_label = "+".join(DATASET_SPLITS)
    _validate_manifest_contract(manifest_rows, f"{DATASET_HF_REPO_ID}/{DATASET_CONFIG}/{split_label}")
    actual_reference_sha256 = reference_annotations_sha256(manifest_rows)
    if actual_reference_sha256 != REFERENCE_ANNOTATIONS_SHA256:
        msg = (
            "Downloaded Sortformer reference annotation SHA-256 mismatch: "
            f"expected {REFERENCE_ANNOTATIONS_SHA256}, found {actual_reference_sha256}"
        )
        raise RuntimeError(msg)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / MANIFEST_FILENAME).open("w", encoding="utf-8") as manifest_file:
        for row in manifest_rows:
            manifest_file.write(json.dumps(row) + "\n")
    (output_path / SOURCE_METADATA_FILENAME).write_text(
        json.dumps(source_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.success(f"Dataset ready: {len(manifest_rows)} unique public AMI SDM meetings")


def stage_model(model_output_path: Path, cache_dir: str) -> None:
    logger.info("=" * 60)
    logger.info("Audio Sortformer Public Model Download")
    logger.info(f"Repo:       {MODEL_HF_REPO_ID}@{MODEL_REVISION}")
    logger.info(f"Filename:   {MODEL_FILENAME} ({MODEL_SIZE_BYTES:,} bytes)")
    logger.info(f"Staging to: {model_output_path}")
    logger.info("=" * 60)

    downloaded_path = Path(
        hf_hub_download(
            repo_id=MODEL_HF_REPO_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            token=False,
        )
    )
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, model_output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage public AMI SDM validation/test data and Sortformer weights for nightly runs"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory for the staged AMI manifest, source metadata, and audio files.",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        required=True,
        help=f"Path for the staged {MODEL_FILENAME} checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache used only during one-time public input staging.",
    )
    parser.add_argument(
        "--container-data-path",
        default=DEFAULT_CONTAINER_DATA_PATH,
        help="Container-visible dataset path written into the staged manifest.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify pinned data and model files without downloading them.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    output_path = args.output_path.resolve()
    model_output_path = args.model_output_path.resolve()
    if args.verify_only:
        dataset_valid = verify_dataset(output_path)
        model_valid = verify_model(model_output_path)
        return 0 if dataset_valid and model_valid else 1

    dataset_valid = verify_dataset(output_path)
    if not dataset_valid:
        stage_dataset(output_path, args.cache_dir, args.container_data_path)
        dataset_valid = verify_dataset(output_path)
    if not dataset_valid:
        return 1

    model_valid = verify_model(model_output_path)
    if not model_valid:
        stage_model(model_output_path, args.cache_dir)
        model_valid = verify_model(model_output_path)
    return 0 if model_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
