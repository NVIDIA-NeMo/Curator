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

"""Stage the pinned public Hindi FLEURS cohort for the Indic ASR benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download
from loguru import logger

from nemo_curator.stages.audio.datasets.file_utils import extract_archive

HF_REPO_ID = "google/fleurs"  # CC-BY-4.0
HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # pragma: allowlist secret
HF_CONFIG = "hi_in"
HF_SPLIT = "train"
TRANSCRIPT_FILENAME = f"data/{HF_CONFIG}/{HF_SPLIT}.tsv"
AUDIO_ARCHIVE_FILENAME = f"data/{HF_CONFIG}/audio/{HF_SPLIT}.tar.gz"
TRANSCRIPT_SHA256 = "fa15b11ca73fd4e8ccb6403f58cde3ac5bbdf27d9654799b3c85c26375489c78"  # pragma: allowlist secret
AUDIO_ARCHIVE_SHA256 = "bb6f52bfb27ca91c54539480111163eb9245305008cfe0526c6f4af3bfdd04e9"  # pragma: allowlist secret
EXPECTED_NUM_ROWS = 2120
EXPECTED_TOTAL_FRAMES = 383_370_240
SAMPLE_RATE = 16_000
MIN_TRANSCRIPT_FIELDS = 6
DEFAULT_CACHE_DIR = "/tmp/curator/audio_indic_asr_cache"  # noqa: S108


@dataclass(frozen=True)
class FleursTranscriptRow:
    """One validated row from the pinned FLEURS TSV."""

    source_id: str
    filename: str
    raw_text: str
    text: str
    num_frames: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_transcript(transcript_path: Path) -> list[FleursTranscriptRow]:
    rows: list[FleursTranscriptRow] = []
    filenames: set[str] = set()
    with transcript_path.open(encoding="utf-8") as transcript_file:
        for line_number, line in enumerate(transcript_file, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < MIN_TRANSCRIPT_FIELDS:
                msg = f"Invalid FLEURS transcript row {transcript_path}:{line_number}"
                raise RuntimeError(msg)
            source_id, filename, raw_text, text, _characters, frame_count = parts[:6]
            try:
                num_frames = int(frame_count)
            except ValueError as e:
                msg = f"Invalid frame count at {transcript_path}:{line_number}"
                raise RuntimeError(msg) from e
            if (
                not source_id
                or not filename.endswith(".wav")
                or Path(filename).name != filename
                or filename in filenames
                or not raw_text.strip()
                or not text.strip()
                or num_frames <= 0
            ):
                msg = f"Invalid or duplicate FLEURS transcript row {transcript_path}:{line_number}"
                raise RuntimeError(msg)
            filenames.add(filename)
            rows.append(
                FleursTranscriptRow(
                    source_id=source_id,
                    filename=filename,
                    raw_text=raw_text,
                    text=text,
                    num_frames=num_frames,
                )
            )
    if not rows:
        msg = f"FLEURS transcript contains no rows: {transcript_path}"
        raise RuntimeError(msg)
    return rows


def _manifest_row(row: FleursTranscriptRow) -> dict[str, object]:
    return {
        "audio_filepath": f"audio/{row.filename}",
        "audio_item_id": f"fleurs_hi_in_train_{Path(row.filename).stem}",
        "corpus": "FLEURS",
        "duration": row.num_frames / SAMPLE_RATE,
        "fleurs_source_id": row.source_id,
        "raw_text": row.raw_text,
        "sampling_rate": SAMPLE_RATE,
        "source_lang": "hi",
        "text": row.text,
    }


def _validate_rows_and_audio(rows: list[FleursTranscriptRow], audio_dir: Path) -> None:
    if len(rows) != EXPECTED_NUM_ROWS:
        msg = f"Expected {EXPECTED_NUM_ROWS} FLEURS rows, found {len(rows)}"
        raise RuntimeError(msg)
    total_frames = sum(row.num_frames for row in rows)
    if total_frames != EXPECTED_TOTAL_FRAMES:
        msg = f"Expected {EXPECTED_TOTAL_FRAMES} FLEURS frames, found {total_frames}"
        raise RuntimeError(msg)

    expected_filenames = {row.filename for row in rows}
    actual_filenames = {path.name for path in audio_dir.iterdir() if path.is_file()}
    if actual_filenames != expected_filenames:
        msg = (
            "FLEURS audio inventory does not match the transcript: "
            f"missing={len(expected_filenames - actual_filenames)}, "
            f"extra={len(actual_filenames - expected_filenames)}"
        )
        raise RuntimeError(msg)
    for row in rows:
        audio_path = audio_dir / row.filename
        info = sf.info(audio_path)
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames != row.num_frames:
            msg = (
                f"Unexpected audio metadata for {audio_path}: "
                f"samplerate={info.samplerate}, channels={info.channels}, frames={info.frames}"
            )
            raise RuntimeError(msg)


def _write_manifest(rows: list[FleursTranscriptRow], manifest_path: Path) -> None:
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(_manifest_row(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _metadata() -> dict[str, object]:
    return {
        "audio_archive_filename": AUDIO_ARCHIVE_FILENAME,
        "audio_archive_sha256": AUDIO_ARCHIVE_SHA256,
        "config": HF_CONFIG,
        "expected_num_rows": EXPECTED_NUM_ROWS,
        "expected_total_frames": EXPECTED_TOTAL_FRAMES,
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "license": "CC-BY-4.0",
        "sample_rate": SAMPLE_RATE,
        "split": HF_SPLIT,
        "transcript_filename": TRANSCRIPT_FILENAME,
        "transcript_sha256": TRANSCRIPT_SHA256,
    }


def verify_dataset(output_path: Path) -> bool:
    """Verify the complete pinned dataset, including every WAV header."""
    try:
        manifest_path = output_path / "manifest.jsonl"
        transcript_path = output_path / "source.tsv"
        metadata_path = output_path / "metadata.json"
        audio_dir = output_path / "audio"
        _require(
            manifest_path.is_file() and transcript_path.is_file() and metadata_path.is_file(),
            f"Required Indic ASR dataset files are missing under {output_path}",
        )
        _require(audio_dir.is_dir(), f"FLEURS audio directory is missing: {audio_dir}")
        _require(
            _sha256_file(transcript_path) == TRANSCRIPT_SHA256,
            "Pinned FLEURS transcript checksum mismatch",
        )
        _require(
            json.loads(metadata_path.read_text(encoding="utf-8")) == _metadata(),
            "Pinned FLEURS metadata does not match the checked-in contract",
        )

        rows = _parse_transcript(transcript_path)
        _validate_rows_and_audio(rows, audio_dir)
        manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        _require(
            manifest_rows == [_manifest_row(row) for row in rows],
            "Pinned FLEURS manifest content or ordering mismatch",
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, sf.SoundFileError) as e:
        logger.error(f"Indic ASR dataset verification failed: {e}")
        return False

    logger.success(
        f"Verified {EXPECTED_NUM_ROWS} Hindi FLEURS clips / "
        f"{EXPECTED_TOTAL_FRAMES / SAMPLE_RATE / 3600:.4f} audio hours at {output_path}"
    )
    return True


def stage_dataset(output_path: Path, cache_dir: str) -> None:
    """Download, verify, and atomically publish the pinned FLEURS cohort."""
    output_path = output_path.resolve()
    if output_path.exists():
        if verify_dataset(output_path):
            logger.info(f"Reusing staged Hindi FLEURS dataset at {output_path}")
            return
        msg = f"Refusing to overwrite incomplete staged data at {output_path}; move or remove it first"
        raise RuntimeError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=TRANSCRIPT_FILENAME,
            cache_dir=cache_dir,
        )
    )
    archive_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=AUDIO_ARCHIVE_FILENAME,
            cache_dir=cache_dir,
        )
    )
    if _sha256_file(transcript_path) != TRANSCRIPT_SHA256:
        msg = f"Unexpected checksum for {TRANSCRIPT_FILENAME} at revision {HF_REVISION}"
        raise RuntimeError(msg)
    if _sha256_file(archive_path) != AUDIO_ARCHIVE_SHA256:
        msg = f"Unexpected checksum for {AUDIO_ARCHIVE_FILENAME} at revision {HF_REVISION}"
        raise RuntimeError(msg)

    staging_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    try:
        extract_archive(str(archive_path), str(staging_path), force_extract=True)
        extracted_audio = staging_path / HF_SPLIT
        _require(
            extracted_audio.is_dir(),
            f"Pinned FLEURS archive did not contain the expected {HF_SPLIT}/ directory",
        )
        audio_dir = staging_path / "audio"
        extracted_audio.replace(audio_dir)
        shutil.copy2(transcript_path, staging_path / "source.tsv")
        rows = _parse_transcript(staging_path / "source.tsv")
        _validate_rows_and_audio(rows, audio_dir)
        _write_manifest(rows, staging_path / "manifest.jsonl")
        (staging_path / "metadata.json").write_text(
            json.dumps(_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_path.replace(output_path)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    if not verify_dataset(output_path):
        msg = f"Published Hindi FLEURS dataset failed verification: {output_path}"
        raise RuntimeError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        return 0 if verify_dataset(args.output_path.resolve()) else 1
    stage_dataset(args.output_path, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
