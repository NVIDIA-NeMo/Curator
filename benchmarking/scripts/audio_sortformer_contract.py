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

"""Pinned public input contract shared by Sortformer prep and execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import soundfile as sf

DATASET_HF_REPO_ID = "diarizers-community/ami"
DATASET_REVISION = "8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0"  # pragma: allowlist secret
DATASET_CONFIG = "sdm"
DATASET_SPLITS = ("validation", "test")
DATASET_SPLIT_NUM_ROWS = {"validation": 18, "test": 16}
DATASET_NUM_ROWS = sum(DATASET_SPLIT_NUM_ROWS.values())
DATASET_SOURCE_DOWNLOAD_BYTES = 2_034_736_248
DATASET_DECODED_AUDIO_BYTES = 2_157_687_735
DATASET_PUBLISHED_TOTAL_DURATION_S = 67_427.088
DATASET_PUBLISHED_MEAN_DURATION_S = DATASET_PUBLISHED_TOTAL_DURATION_S / DATASET_NUM_ROWS
DATASET_LICENSE = "cc-by-4.0"
MIN_DATASET_DURATION_S = 67_426.0
MAX_DATASET_DURATION_S = 67_428.0
REFERENCE_ANNOTATIONS_SHA256 = (
    "ad548f866d578402a03dc6e10fb92c613092f862e7ca0ec0592a8e74c114ad99"  # pragma: allowlist secret
)

MODEL_HF_REPO_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"  # pragma: allowlist secret
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"
MODEL_SIZE_BYTES = 471_367_680
MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"  # pragma: allowlist secret
MODEL_LICENSE = "nvidia-open-model-license"

MANIFEST_FILENAME = "manifest.jsonl"
EXPECTED_AUDIO_FILENAMES = tuple(
    f"ami_sdm_{split}_{index:03d}.wav" for split in DATASET_SPLITS for index in range(DATASET_SPLIT_NUM_ROWS[split])
)
AUDIO_CORPUS_SHA256 = "6acd64df9e893666d8c60f787c2413e4f47dc42efe3c61542d4d7392ec6a2c43"  # pragma: allowlist secret
AUDIO_SAMPLE_RATE = 16_000
MONO_CHANNELS = 1
TIMESTAMP_TOLERANCE_S = 1e-3


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as input_file:
        return hashlib.file_digest(input_file, "sha256").hexdigest()


def audio_corpus_sha256(audio_dir: Path) -> str:
    """Hash the ordered filename/file-digest pairs for the complete corpus."""
    corpus_digest = hashlib.sha256()
    for filename in EXPECTED_AUDIO_FILENAMES:
        corpus_digest.update(filename.encode())
        corpus_digest.update(b"\0")
        with (audio_dir / filename).open("rb") as audio_file:
            corpus_digest.update(hashlib.file_digest(audio_file, "sha256").digest())
    return corpus_digest.hexdigest()


def reference_annotations_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash semantic manifest fields independently of path prefixes."""
    canonical_rows = [
        {
            "audio_filename": Path(row["audio_filepath"]).name,
            "audio_item_id": str(row["audio_item_id"]),
            "session_name": str(row["session_name"]),
            "duration": float(row["duration"]),
            "timestamps_start": [float(value) for value in row["timestamps_start"]],
            "timestamps_end": [float(value) for value in row["timestamps_end"]],
            "speakers": [str(value) for value in row["speakers"]],
        }
        for row in rows
    ]
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_model(model_path: Path) -> None:
    """Require the exact pinned public Sortformer checkpoint."""
    if not model_path.is_file():
        msg = f"Sortformer model not found: {model_path}"
        raise FileNotFoundError(msg)
    actual_size = model_path.stat().st_size
    if actual_size != MODEL_SIZE_BYTES:
        msg = f"Sortformer model size mismatch: expected {MODEL_SIZE_BYTES}, found {actual_size}"
        raise RuntimeError(msg)
    actual_sha256 = sha256(model_path)
    if actual_sha256 != MODEL_SHA256:
        msg = f"Sortformer model SHA-256 mismatch: expected {MODEL_SHA256}, found {actual_sha256}"
        raise RuntimeError(msg)


def validate_reference_annotations(row: dict[str, Any], duration_s: float, label: str) -> None:
    """Validate the AMI reference turns retained for semantic scoring."""
    starts = row.get("timestamps_start")
    ends = row.get("timestamps_end")
    speakers = row.get("speakers")
    if not isinstance(starts, list) or not isinstance(ends, list) or not isinstance(speakers, list):
        msg = f"{label} must contain timestamps_start, timestamps_end, and speakers lists"
        raise TypeError(msg)
    if not starts or len(starts) != len(ends) or len(starts) != len(speakers):
        msg = f"{label} reference annotation lists must have the same nonzero length"
        raise RuntimeError(msg)

    for segment_index, (start, end, speaker) in enumerate(zip(starts, ends, speakers, strict=True)):
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not 0 <= float(start) < float(end) <= duration_s + TIMESTAMP_TOLERANCE_S
        ):
            msg = f"{label} has invalid reference segment {segment_index}: start={start!r}, end={end!r}"
            raise RuntimeError(msg)
        if not isinstance(speaker, str) or not speaker:
            msg = f"{label} has an empty reference speaker at segment {segment_index}"
            raise RuntimeError(msg)

    num_speakers = len(set(speakers))
    if not 3 <= num_speakers <= 4:  # noqa: PLR2004
        msg = f"{label} must contain 3-4 AMI speakers, found {num_speakers}"
        raise RuntimeError(msg)


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
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


def write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row) + "\n")


def validate_manifest(rows: list[dict[str, Any]], label: str) -> float:  # noqa: C901
    """Validate identities, durations, annotations, and complete unique coverage."""
    if len(rows) != DATASET_NUM_ROWS:
        msg = f"{label} must contain exactly {DATASET_NUM_ROWS} rows, found {len(rows)}"
        raise RuntimeError(msg)

    expected_files = set(EXPECTED_AUDIO_FILENAMES)
    seen_files: set[str] = set()
    seen_ids: set[str] = set()
    seen_sessions: set[str] = set()
    total_duration_s = 0.0
    for line_number, row in enumerate(rows, start=1):
        audio_filepath = row.get("audio_filepath")
        if not isinstance(audio_filepath, str) or not audio_filepath:
            msg = f"{label} line {line_number} must contain audio_filepath"
            raise RuntimeError(msg)
        audio_filename = Path(audio_filepath).name
        if audio_filename not in expected_files:
            msg = f"{label} line {line_number} references unexpected audio file {audio_filename!r}"
            raise RuntimeError(msg)
        if audio_filename in seen_files:
            msg = f"{label} contains duplicate audio file {audio_filename!r}"
            raise RuntimeError(msg)
        seen_files.add(audio_filename)

        for field, seen in (("audio_item_id", seen_ids), ("session_name", seen_sessions)):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                msg = f"{label} line {line_number} must contain a nonempty {field}"
                raise RuntimeError(msg)
            if value in seen:
                msg = f"{label} contains duplicate {field} {value!r}"
                raise RuntimeError(msg)
            seen.add(value)

        duration = row.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            msg = f"{label} line {line_number} must contain a finite positive duration"
            raise RuntimeError(msg)
        total_duration_s += float(duration)
        validate_reference_annotations(row, float(duration), f"{label} line {line_number}")

    if seen_files != expected_files:
        msg = f"{label} is missing expected audio files: {sorted(expected_files - seen_files)}"
        raise RuntimeError(msg)
    if not MIN_DATASET_DURATION_S <= total_duration_s <= MAX_DATASET_DURATION_S:
        msg = f"{label} total duration is outside the expected AMI SDM range: {total_duration_s:.3f}s"
        raise RuntimeError(msg)
    if reference_annotations_sha256(rows) != REFERENCE_ANNOTATIONS_SHA256:
        msg = f"{label} reference annotation SHA-256 mismatch"
        raise RuntimeError(msg)
    return total_duration_s


def validate_staged_dataset(root: Path) -> tuple[list[dict[str, Any]], float]:
    """Validate the exact staged public workload once for prep or execution."""
    manifest_path = root / MANIFEST_FILENAME
    audio_dir = root / "audio"
    if not manifest_path.is_file() or not audio_dir.is_dir():
        msg = f"Expected {MANIFEST_FILENAME} and audio/ under {root}"
        raise FileNotFoundError(msg)

    rows = load_manifest(manifest_path)
    total_duration_s = validate_manifest(rows, str(manifest_path))
    actual_files = {path.name for path in audio_dir.glob("*.wav")}
    expected_files = set(EXPECTED_AUDIO_FILENAMES)
    if actual_files != expected_files:
        msg = (
            f"Staged audio mismatch; missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
        raise RuntimeError(msg)

    rows_by_filename = {Path(row["audio_filepath"]).name: row for row in rows}
    for filename in EXPECTED_AUDIO_FILENAMES:
        audio_path = audio_dir / filename
        info = sf.info(audio_path)
        if info.samplerate != AUDIO_SAMPLE_RATE or info.channels != MONO_CHANNELS:
            msg = f"Expected mono 16 kHz WAV, found {info.channels} channels at {info.samplerate} Hz: {audio_path}"
            raise RuntimeError(msg)
        duration_s = float(info.frames) / info.samplerate
        if not math.isclose(duration_s, float(rows_by_filename[filename]["duration"]), abs_tol=TIMESTAMP_TOLERANCE_S):
            msg = f"Manifest duration does not match staged WAV: {audio_path}"
            raise RuntimeError(msg)

    actual_corpus_sha256 = audio_corpus_sha256(audio_dir)
    if actual_corpus_sha256 != AUDIO_CORPUS_SHA256:
        msg = f"Staged audio corpus SHA-256 mismatch: expected {AUDIO_CORPUS_SHA256}, found {actual_corpus_sha256}"
        raise RuntimeError(msg)
    return rows, total_duration_s
