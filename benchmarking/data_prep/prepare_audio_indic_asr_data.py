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

"""Stage the pinned public 531-hour Hindi ASR cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download
from loguru import logger

from nemo_curator.stages.audio.datasets.file_utils import extract_archive

HF_REPO_ID = "ketav/parakeet-hindi-asr"
HF_REVISION = "35376a112c4b79318eeaba0c0dd1b6f1a9bf0ea0"  # pragma: allowlist secret
HF_SPLIT = "train"
SOURCE_MANIFEST_FILENAME = "data/manifests/train_hi_clean.json"
AUDIO_ARCHIVE_FILENAME = "data/hindi/hindi_audio.tar.gz"
SOURCE_MANIFEST_SHA256 = "407b58ccb9c74c75a5129e882b1fd000970e082e109adf95a1889592c66964a4"  # pragma: allowlist secret
AUDIO_ARCHIVE_SHA256 = "9f481545c1fe183eeab3a80c1a170215299c333f1cd754f4fab221eebf517c20"  # pragma: allowlist secret
EXPECTED_NUM_ROWS = 216_169
EXPECTED_TOTAL_DURATION_MS = 1_914_385_701
SAMPLE_RATE = 16_000
MAX_DURATION_ERROR_S = 0.001
SOURCE_MANIFEST_AUDIO_SUFFIX = ".wav"
ARCHIVE_AUDIO_SUFFIX = ".flac"
ARCHIVE_AUDIO_FORMAT = "FLAC"
DEFAULT_CACHE_DIR = "/tmp/curator/audio_indic_asr_cache"  # noqa: S108


@dataclass(frozen=True)
class HindiASRRow:
    """One validated row from the pinned Hindi source manifest."""

    source_audio_filepath: str
    filename: str
    text: str
    duration_ms: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_source_manifest(manifest_path: Path) -> list[HindiASRRow]:
    rows: list[HindiASRRow] = []
    filenames: set[str] = set()
    total_duration_ms = 0
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                source_row = json.loads(line, parse_float=Decimal)
                source_audio_filepath = source_row["audio_filepath"]
                text = source_row["text"]
                duration = Decimal(str(source_row["duration"]))
            except (json.JSONDecodeError, KeyError, TypeError, InvalidOperation) as e:
                msg = f"Invalid Hindi source manifest row {manifest_path}:{line_number}"
                raise RuntimeError(msg) from e

            source_filename = Path(source_audio_filepath).name if isinstance(source_audio_filepath, str) else ""
            filename = (
                f"{Path(source_filename).stem}{ARCHIVE_AUDIO_SUFFIX}"
                if source_filename.endswith(SOURCE_MANIFEST_AUDIO_SUFFIX)
                else ""
            )
            duration_ms_value = duration * 1000
            if (
                not source_audio_filepath
                or not filename
                or filename in filenames
                or not isinstance(text, str)
                or not text.strip()
                or duration <= 0
                or duration_ms_value != duration_ms_value.to_integral_value()
            ):
                msg = f"Invalid or duplicate Hindi source manifest row {manifest_path}:{line_number}"
                raise RuntimeError(msg)

            duration_ms = int(duration_ms_value)
            filenames.add(filename)
            total_duration_ms += duration_ms
            rows.append(
                HindiASRRow(
                    source_audio_filepath=source_audio_filepath,
                    filename=filename,
                    text=text,
                    duration_ms=duration_ms,
                )
            )

    if len(rows) != EXPECTED_NUM_ROWS:
        msg = f"Expected {EXPECTED_NUM_ROWS} Hindi rows, found {len(rows)}"
        raise RuntimeError(msg)
    if total_duration_ms != EXPECTED_TOTAL_DURATION_MS:
        msg = f"Expected {EXPECTED_TOTAL_DURATION_MS} ms of Hindi audio, found {total_duration_ms}"
        raise RuntimeError(msg)
    return rows


def _manifest_row(row: HindiASRRow) -> dict[str, object]:
    return {
        "audio_filepath": f"audio/{row.filename}",
        "audio_item_id": f"parakeet_hindi_asr_train_{Path(row.filename).stem}",
        "corpus": "Parakeet Hindi ASR",
        "duration": row.duration_ms / 1000,
        "sampling_rate": SAMPLE_RATE,
        "source_audio_filepath": row.source_audio_filepath,
        "source_lang": "hi",
        "text": row.text,
    }


def _validate_rows_and_audio(rows: list[HindiASRRow], audio_dir: Path) -> None:
    expected_filenames = {row.filename for row in rows}
    actual_filenames = {path.name for path in audio_dir.iterdir() if path.is_file()}
    if actual_filenames != expected_filenames:
        msg = (
            "Hindi audio inventory does not match the source manifest: "
            f"missing={len(expected_filenames - actual_filenames)}, "
            f"extra={len(actual_filenames - expected_filenames)}"
        )
        raise RuntimeError(msg)

    for index, row in enumerate(rows, start=1):
        audio_path = audio_dir / row.filename
        info = sf.info(audio_path)
        measured_duration_s = info.frames / info.samplerate
        if (
            info.format != ARCHIVE_AUDIO_FORMAT
            or info.samplerate != SAMPLE_RATE
            or info.channels != 1
            or abs(measured_duration_s - row.duration_ms / 1000) > MAX_DURATION_ERROR_S
        ):
            msg = (
                f"Unexpected audio metadata for {audio_path}: format={info.format}, "
                f"samplerate={info.samplerate}, channels={info.channels}, frames={info.frames}, "
                f"source_duration_ms={row.duration_ms}"
            )
            raise RuntimeError(msg)
        if index % 20_000 == 0:
            logger.info(f"Verified {index}/{len(rows)} Hindi FLAC headers")


def _write_manifest(rows: list[HindiASRRow], manifest_path: Path) -> None:
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(_manifest_row(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _select_duration_prefix(source_manifest_path: Path, target_duration_s: Decimal) -> tuple[list[str], Decimal]:
    selected_lines: list[str] = []
    selected_duration_s = Decimal(0)
    with source_manifest_path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_float=Decimal)
                duration_s = Decimal(str(row["duration"]))
            except (json.JSONDecodeError, KeyError, TypeError, InvalidOperation) as e:
                msg = f"Invalid staged Hindi manifest row {source_manifest_path}:{line_number}"
                raise RuntimeError(msg) from e
            if not duration_s.is_finite() or duration_s <= 0:
                msg = f"Invalid duration in staged Hindi manifest row {source_manifest_path}:{line_number}"
                raise RuntimeError(msg)
            selected_lines.append(line if line.endswith("\n") else f"{line}\n")
            selected_duration_s += duration_s
            if selected_duration_s >= target_duration_s:
                break
    return selected_lines, selected_duration_s


def write_duration_subset(
    dataset_path: Path,
    subset_manifest_path: Path,
    target_hours: Decimal,
) -> None:
    """Write a deterministic duration-prefix manifest from the pinned cohort."""
    if not target_hours.is_finite() or target_hours <= 0:
        msg = f"Subset duration must be finite and positive, found {target_hours}"
        raise ValueError(msg)

    source_manifest_path = (dataset_path / "manifest.jsonl").resolve()
    subset_manifest_path = subset_manifest_path.expanduser().resolve()
    if source_manifest_path == subset_manifest_path:
        msg = "Subset manifest must not replace the canonical full-cohort manifest"
        raise ValueError(msg)

    target_duration_s = target_hours * Decimal(3600)
    selected_lines, selected_duration_s = _select_duration_prefix(source_manifest_path, target_duration_s)

    if selected_duration_s < target_duration_s:
        msg = (
            f"Pinned Hindi cohort contains only {selected_duration_s / Decimal(3600):.4f} hours; "
            f"cannot write a {target_hours}-hour subset"
        )
        raise RuntimeError(msg)

    contents = "".join(selected_lines)
    if subset_manifest_path.is_file():
        if subset_manifest_path.read_text(encoding="utf-8") != contents:
            msg = f"Refusing to overwrite a different subset manifest: {subset_manifest_path}"
            raise RuntimeError(msg)
        logger.info(f"Reusing deterministic Hindi subset manifest at {subset_manifest_path}")
    else:
        subset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{subset_manifest_path.name}.",
            suffix=".tmp",
            dir=subset_manifest_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)
        try:
            temporary_path.replace(subset_manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    logger.success(
        f"Selected {len(selected_lines)} clips / {selected_duration_s / Decimal(3600):.4f} audio hours "
        f"at {subset_manifest_path}"
    )


def _metadata() -> dict[str, object]:
    return {
        "audio_archive_filename": AUDIO_ARCHIVE_FILENAME,
        "audio_archive_sha256": AUDIO_ARCHIVE_SHA256,
        "archive_audio_format": ARCHIVE_AUDIO_FORMAT,
        "archive_audio_suffix": ARCHIVE_AUDIO_SUFFIX,
        "expected_num_rows": EXPECTED_NUM_ROWS,
        "expected_total_duration_ms": EXPECTED_TOTAL_DURATION_MS,
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "license": "Apache-2.0",
        "sample_rate": SAMPLE_RATE,
        "source_manifest_filename": SOURCE_MANIFEST_FILENAME,
        "source_manifest_audio_suffix": SOURCE_MANIFEST_AUDIO_SUFFIX,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "split": HF_SPLIT,
    }


def _verify_manifest(rows: list[HindiASRRow], manifest_path: Path) -> None:
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, row in enumerate(rows, start=1):
            line = manifest_file.readline()
            if not line:
                msg = f"Staged Hindi manifest ended before row {line_number}"
                raise RuntimeError(msg)
            if json.loads(line) != _manifest_row(row):
                msg = f"Staged Hindi manifest differs from its pinned source at row {line_number}"
                raise RuntimeError(msg)
        if any(line.strip() for line in manifest_file):
            msg = "Staged Hindi manifest contains unexpected extra rows"
            raise RuntimeError(msg)


def verify_dataset(output_path: Path) -> bool:
    """Verify the complete pinned dataset and every FLAC header."""
    try:
        manifest_path = output_path / "manifest.jsonl"
        source_manifest_path = output_path / "source_manifest.jsonl"
        metadata_path = output_path / "metadata.json"
        audio_dir = output_path / "audio"
        _require(
            manifest_path.is_file() and source_manifest_path.is_file() and metadata_path.is_file(),
            f"Required Indic ASR dataset files are missing under {output_path}",
        )
        _require(audio_dir.is_dir(), f"Hindi audio directory is missing: {audio_dir}")
        _require(
            _sha256_file(source_manifest_path) == SOURCE_MANIFEST_SHA256,
            "Pinned Hindi source manifest checksum mismatch",
        )
        _require(
            json.loads(metadata_path.read_text(encoding="utf-8")) == _metadata(),
            "Pinned Hindi metadata does not match the checked-in contract",
        )

        rows = _parse_source_manifest(source_manifest_path)
        _validate_rows_and_audio(rows, audio_dir)
        _verify_manifest(rows, manifest_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, sf.SoundFileError) as e:
        logger.error(f"Indic ASR dataset verification failed: {e}")
        return False

    logger.success(
        f"Verified {EXPECTED_NUM_ROWS} unique Hindi clips / "
        f"{EXPECTED_TOTAL_DURATION_MS / 3_600_000:.4f} audio hours at {output_path}"
    )
    return True


def _publish_audio(rows: list[HindiASRRow], extracted_path: Path, audio_dir: Path) -> None:
    expected_filenames = {row.filename for row in rows}
    source_paths: dict[str, Path] = {}
    for candidate in extracted_path.rglob(f"*{ARCHIVE_AUDIO_SUFFIX}"):
        if candidate.name not in expected_filenames:
            continue
        if candidate.name in source_paths:
            msg = f"Duplicate Hindi FLAC filename in source archive: {candidate.name}"
            raise RuntimeError(msg)
        source_paths[candidate.name] = candidate

    missing = expected_filenames - source_paths.keys()
    if missing:
        msg = f"Hindi source archive is missing {len(missing)} manifest-matched FLAC files"
        raise RuntimeError(msg)

    audio_dir.mkdir()
    for index, row in enumerate(rows, start=1):
        source_paths[row.filename].replace(audio_dir / row.filename)
        if index % 20_000 == 0:
            logger.info(f"Published {index}/{len(rows)} Hindi FLAC files")


def stage_dataset(output_path: Path, cache_dir: str) -> None:
    """Download, verify, and atomically publish the pinned Hindi cohort."""
    output_path = output_path.resolve()
    if output_path.exists():
        if verify_dataset(output_path):
            logger.info(f"Reusing staged Hindi ASR dataset at {output_path}")
            return
        msg = f"Refusing to overwrite incomplete staged data at {output_path}; move or remove it first"
        raise RuntimeError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=SOURCE_MANIFEST_FILENAME,
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
    if _sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        msg = f"Unexpected checksum for {SOURCE_MANIFEST_FILENAME} at revision {HF_REVISION}"
        raise RuntimeError(msg)
    if _sha256_file(archive_path) != AUDIO_ARCHIVE_SHA256:
        msg = f"Unexpected checksum for {AUDIO_ARCHIVE_FILENAME} at revision {HF_REVISION}"
        raise RuntimeError(msg)

    staging_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    try:
        extracted_path = staging_path / "source_archive"
        extracted_path.mkdir()
        extract_archive(str(archive_path), str(extracted_path), force_extract=True)
        shutil.copy2(source_manifest_path, staging_path / "source_manifest.jsonl")
        rows = _parse_source_manifest(staging_path / "source_manifest.jsonl")
        _publish_audio(rows, extracted_path, staging_path / "audio")
        shutil.rmtree(extracted_path)
        _validate_rows_and_audio(rows, staging_path / "audio")
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
        msg = f"Published Hindi ASR dataset failed verification: {output_path}"
        raise RuntimeError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--subset-hours",
        type=Decimal,
        default=None,
        help="Also write a deterministic duration-prefix manifest from the staged cohort",
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=None,
        help="Output path paired with --subset-hours",
    )
    args = parser.parse_args()

    if (args.subset_hours is None) != (args.subset_manifest is None):
        parser.error("--subset-hours and --subset-manifest must be provided together")

    output_path = args.output_path.resolve()
    if args.verify_only:
        if not verify_dataset(output_path):
            return 1
    else:
        stage_dataset(output_path, args.cache_dir)
    if args.subset_hours is not None:
        write_duration_subset(output_path, args.subset_manifest, args.subset_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
