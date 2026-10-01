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
import fcntl
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download
from loguru import logger

HF_REPO_ID = "ketav/parakeet-hindi-asr"
HF_REVISION = "35376a112c4b79318eeaba0c0dd1b6f1a9bf0ea0"  # pragma: allowlist secret
HF_SPLIT = "train"
SOURCE_MANIFEST_FILENAME = "data/manifests/train_hi_clean.json"
AUDIO_ARCHIVE_FILENAME = "data/hindi/hindi_audio.tar.gz"
SOURCE_MANIFEST_BYTES = 79_204_142
SOURCE_MANIFEST_SHA256 = "407b58ccb9c74c75a5129e882b1fd000970e082e109adf95a1889592c66964a4"  # pragma: allowlist secret
AUDIO_ARCHIVE_BYTES = 40_543_034_328
AUDIO_ARCHIVE_SHA256 = "9f481545c1fe183eeab3a80c1a170215299c333f1cd754f4fab221eebf517c20"  # pragma: allowlist secret

# The source manifest contract is checked before its five pinned unusable members
# are removed. The retained cohort is shared byte-for-byte by both executors.
EXPECTED_SOURCE_NUM_ROWS = 216_169
EXPECTED_SOURCE_TOTAL_DURATION_MS = 1_914_385_701
EXPECTED_NUM_ROWS = 216_164
EXPECTED_TOTAL_DURATION_MS = 1_914_327_468
EXPECTED_REJECTED_DURATION_MS = 58_233
EXPECTED_CANONICAL_MANIFEST_BYTES = 116_382_162
EXPECTED_CANONICAL_MANIFEST_SHA256 = "0a8ccc0f3ff8d4ad35b3e7e104e5e093b0de14a92542d71fa6911c8045373727"
EXPECTED_AUDIO_INVENTORY_BYTES = 22_136_933
EXPECTED_AUDIO_INVENTORY_SHA256 = "5f96be6dc78711107fa8ad1e1d1c4d05a1b1ef1e02eb4026b42519f1a8de8166"
EXPECTED_AUDIO_BYTES = 36_406_201_076
EXPECTED_DECODED_FRAMES = 30_629_236_133
EXPECTED_REFERENCED_MEMBER_BYTES = 36_406_725_364
EXPECTED_UNREFERENCED_AUDIO_COUNT = 33_246
EXPECTED_UNREFERENCED_AUDIO_NAMES_BYTES = 797_904
EXPECTED_UNREFERENCED_AUDIO_NAMES_SHA256 = "396b930cf1a68f69c4596bcf58bd85805fe6c3f2c2ddfe4c654ab6dc76d82d5b"

SAMPLE_RATE = 16_000
CHANNELS = 1
MAX_DURATION_ERROR_MS = 1
SOURCE_MANIFEST_AUDIO_SUFFIX = ".wav"
ARCHIVE_AUDIO_SUFFIX = ".flac"
ARCHIVE_AUDIO_FORMAT = "FLAC"
AUDIO_INVENTORY_SCHEMA = "relative_path<TAB>payload_bytes<TAB>sha256<TAB>decoded_frames<LF>"
MANIFEST_FILENAME = "manifest.jsonl"
STAGED_SOURCE_MANIFEST_FILENAME = "source_manifest.jsonl"
AUDIO_INVENTORY_FILENAME = "audio_inventory.tsv"
METADATA_FILENAME = "metadata.json"
DATASET_RECEIPT_FILENAME = "dataset_receipt.json"
DATASET_RECEIPT_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = "/tmp/curator/audio_indic_asr_cache"  # noqa: S108


@dataclass(frozen=True)
class HindiASRRow:
    """One validated row from the pinned Hindi source manifest."""

    line_number: int
    source_audio_filepath: str
    filename: str
    text: str
    duration_ms: int


@dataclass(frozen=True)
class RejectedAudioMember:
    """One exact, scan-proven unusable archive member."""

    filename: str
    archive_member: str
    archive_size: int
    sha256: str
    line_number: int
    source_audio_filepath: str
    source_duration_ms: int
    rejection_code: str
    error_type: str
    header_hex: str


REJECTED_AUDIO_MEMBERS = (
    RejectedAudioMember(
        filename="hindi_017643.flac",
        archive_member="audio/hindi_017643.flac",
        archive_size=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        line_number=16_722,
        source_audio_filepath="/data/hindi_data/audio/hindi_017643.wav",
        source_duration_ms=7_457,
        rejection_code="empty_audio_payload",
        error_type="EmptyAudioPayload",
        header_hex="",
    ),
    RejectedAudioMember(
        filename="hindi_017655.flac",
        archive_member="audio/hindi_017655.flac",
        archive_size=262_144,
        sha256="d268fbd944bd439a4fe66efbf70f3764531b932644727feda1b144765cb6898b",
        line_number=16_734,
        source_audio_filepath="/data/hindi_data/audio/hindi_017655.wav",
        source_duration_ms=17_362,
        rejection_code="audio_decode_error",
        error_type="LibsndfileError",
        header_hex="664c6143000000220480048000000000091503e800f000000000000000000000",
    ),
    RejectedAudioMember(
        filename="hindi_017656.flac",
        archive_member="audio/hindi_017656.flac",
        archive_size=262_144,
        sha256="7360b6d905953510ac85c8aa4a2eb5bf947d02f31d4e92a2c782fa553b8faaf7",
        line_number=16_735,
        source_audio_filepath="/data/hindi_data/audio/hindi_017656.wav",
        source_duration_ms=15_329,
        rejection_code="audio_decode_error",
        error_type="LibsndfileError",
        header_hex="664c6143000000220480048000000000091503e800f000000000000000000000",
    ),
    RejectedAudioMember(
        filename="hindi_017666.flac",
        archive_member="audio/hindi_017666.flac",
        archive_size=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        line_number=16_745,
        source_audio_filepath="/data/hindi_data/audio/hindi_017666.wav",
        source_duration_ms=9_242,
        rejection_code="empty_audio_payload",
        error_type="EmptyAudioPayload",
        header_hex="",
    ),
    RejectedAudioMember(
        filename="hindi_018619.flac",
        archive_member="audio/hindi_018619.flac",
        archive_size=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        line_number=17_644,
        source_audio_filepath="/data/hindi_data/audio/hindi_018619.wav",
        source_duration_ms=8_843,
        rejection_code="empty_audio_payload",
        error_type="EmptyAudioPayload",
        header_hex="",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


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
                or not duration.is_finite()
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
                    line_number=line_number,
                    source_audio_filepath=source_audio_filepath,
                    filename=filename,
                    text=text,
                    duration_ms=duration_ms,
                )
            )

    _require(
        len(rows) == EXPECTED_SOURCE_NUM_ROWS,
        f"Expected {EXPECTED_SOURCE_NUM_ROWS} Hindi source rows, found {len(rows)}",
    )
    _require(
        total_duration_ms == EXPECTED_SOURCE_TOTAL_DURATION_MS,
        f"Expected {EXPECTED_SOURCE_TOTAL_DURATION_MS} ms in the Hindi source, found {total_duration_ms}",
    )
    return rows


def _rejected_member_record(member: RejectedAudioMember) -> dict[str, object]:
    return {
        "archive_member": member.archive_member,
        "archive_size": member.archive_size,
        "error_type": member.error_type,
        "filename": member.filename,
        "header_hex": member.header_hex,
        "line_number": member.line_number,
        "rejection_code": member.rejection_code,
        "sha256": member.sha256,
        "source_audio_filepath": member.source_audio_filepath,
        "source_duration_ms": member.source_duration_ms,
    }


def _retained_rows(rows: list[HindiASRRow]) -> list[HindiASRRow]:
    rows_by_filename = {row.filename: row for row in rows}
    rejected_filenames: set[str] = set()
    for rejected in REJECTED_AUDIO_MEMBERS:
        _require(rejected.filename not in rejected_filenames, f"Duplicate rejected filename: {rejected.filename}")
        rejected_filenames.add(rejected.filename)
        row = rows_by_filename.get(rejected.filename)
        _require(row is not None, f"Rejected filename is absent from the pinned source: {rejected.filename}")
        _require(
            row.line_number == rejected.line_number
            and row.source_audio_filepath == rejected.source_audio_filepath
            and row.duration_ms == rejected.source_duration_ms,
            f"Rejected source record changed for {rejected.filename}",
        )
        expected_error_types = {
            "audio_decode_error": "LibsndfileError",
            "empty_audio_payload": "EmptyAudioPayload",
        }
        _require(
            expected_error_types.get(rejected.rejection_code) == rejected.error_type,
            f"Unexpected rejected-audio classification for {rejected.filename}",
        )

    retained = [row for row in rows if row.filename not in rejected_filenames]
    retained_duration_ms = sum(row.duration_ms for row in retained)
    rejected_duration_ms = sum(member.source_duration_ms for member in REJECTED_AUDIO_MEMBERS)
    _require(len(retained) == EXPECTED_NUM_ROWS, f"Expected {EXPECTED_NUM_ROWS} retained rows, found {len(retained)}")
    _require(
        retained_duration_ms == EXPECTED_TOTAL_DURATION_MS,
        f"Expected {EXPECTED_TOTAL_DURATION_MS} retained ms, found {retained_duration_ms}",
    )
    _require(
        rejected_duration_ms == EXPECTED_REJECTED_DURATION_MS,
        f"Expected {EXPECTED_REJECTED_DURATION_MS} rejected ms, found {rejected_duration_ms}",
    )
    _require(
        retained_duration_ms + rejected_duration_ms == EXPECTED_SOURCE_TOTAL_DURATION_MS,
        "Retained and rejected duration do not reconstruct the source duration",
    )
    return retained


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


def _manifest_line(row: HindiASRRow) -> bytes:
    return (json.dumps(_manifest_row(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _write_manifest(rows: list[HindiASRRow], manifest_path: Path) -> None:
    digest = hashlib.sha256()
    total_bytes = 0
    with manifest_path.open("xb") as manifest_file:
        for row in rows:
            line = _manifest_line(row)
            manifest_file.write(line)
            digest.update(line)
            total_bytes += len(line)
    _require(
        total_bytes == EXPECTED_CANONICAL_MANIFEST_BYTES,
        f"Canonical manifest byte count changed: {total_bytes}",
    )
    _require(
        digest.hexdigest() == EXPECTED_CANONICAL_MANIFEST_SHA256,
        f"Canonical manifest checksum changed: {digest.hexdigest()}",
    )


def _decode_retained_audio(payload: bytes, row: HindiASRRow) -> int:
    try:
        with sf.SoundFile(io.BytesIO(payload)) as audio:
            audio_format = audio.format
            samplerate = audio.samplerate
            channels = audio.channels
            declared_frames = audio.frames
            decoded_frames = 0
            while True:
                decoded = audio.buffer_read(65_536, dtype="int16")
                if not decoded:
                    break
                decoded_frames += len(decoded) // (2 * channels)
    except Exception as e:
        msg = f"Could not fully decode retained Hindi audio member {row.filename}"
        raise RuntimeError(msg) from e

    duration_error_numerator = abs(declared_frames * 1000 - row.duration_ms * samplerate)
    duration_error_limit = MAX_DURATION_ERROR_MS * samplerate
    _require(
        audio_format == ARCHIVE_AUDIO_FORMAT
        and samplerate == SAMPLE_RATE
        and channels == CHANNELS
        and decoded_frames == declared_frames
        and duration_error_numerator <= duration_error_limit,
        (
            f"Unexpected retained audio contract for {row.filename}: format={audio_format}, "
            f"samplerate={samplerate}, channels={channels}, declared_frames={declared_frames}, "
            f"decoded_frames={decoded_frames}, source_duration_ms={row.duration_ms}"
        ),
    )
    return decoded_frames


def _validate_rejected_payload(payload: bytes, rejected: RejectedAudioMember) -> None:
    _require(
        payload[:32].hex() == rejected.header_hex,
        f"Pinned rejected audio header changed: {rejected.filename}",
    )
    if rejected.rejection_code == "empty_audio_payload":
        _require(
            not payload and rejected.archive_size == 0 and rejected.error_type == "EmptyAudioPayload",
            f"Pinned empty-audio rejection is no longer empty: {rejected.filename}",
        )
        return

    _require(
        rejected.rejection_code == "audio_decode_error" and rejected.error_type == "LibsndfileError" and bool(payload),
        f"Unexpected non-empty rejection contract: {rejected.filename}",
    )
    try:
        with sf.SoundFile(io.BytesIO(payload)) as audio:
            while audio.buffer_read(65_536, dtype="int16"):
                pass
    except sf.LibsndfileError:
        return
    except Exception as e:
        msg = f"Rejected Hindi audio failed with an unexpected decoder error: {rejected.filename}"
        raise RuntimeError(msg) from e
    msg = f"Pinned Hindi audio unexpectedly completed a full decode: {rejected.filename}"
    raise RuntimeError(msg)


def _validate_unreferenced_members(unreferenced_members: list[str]) -> None:
    unreferenced_names = "".join(f"{name}\n" for name in sorted(unreferenced_members)).encode()
    unreferenced_sha256 = _sha256_bytes(unreferenced_names)
    _require(
        len(unreferenced_members) == EXPECTED_UNREFERENCED_AUDIO_COUNT
        and len(unreferenced_names) == EXPECTED_UNREFERENCED_AUDIO_NAMES_BYTES
        and unreferenced_sha256 == EXPECTED_UNREFERENCED_AUDIO_NAMES_SHA256,
        (
            "Unreferenced Hindi archive member inventory changed: "
            f"count={len(unreferenced_members)}, bytes={len(unreferenced_names)}, sha256={unreferenced_sha256}"
        ),
    )


def _write_audio_inventory(
    retained: list[HindiASRRow],
    inventory: dict[str, tuple[int, str, int]],
    inventory_path: Path,
) -> None:
    inventory_contents = b"".join(
        (
            f"audio/{row.filename}\t{inventory[row.filename][0]}\t{inventory[row.filename][1]}\t"
            f"{inventory[row.filename][2]}\n"
        ).encode("ascii")
        for row in retained
    )
    audio_bytes = sum(entry[0] for entry in inventory.values())
    decoded_frames = sum(entry[2] for entry in inventory.values())
    _require(len(inventory) == EXPECTED_NUM_ROWS, f"Expected {EXPECTED_NUM_ROWS} audio files, found {len(inventory)}")
    _require(audio_bytes == EXPECTED_AUDIO_BYTES, f"Retained Hindi audio bytes changed: {audio_bytes}")
    _require(decoded_frames == EXPECTED_DECODED_FRAMES, f"Retained Hindi decoded frames changed: {decoded_frames}")
    _require(
        len(inventory_contents) == EXPECTED_AUDIO_INVENTORY_BYTES,
        f"Hindi audio inventory byte count changed: {len(inventory_contents)}",
    )
    inventory_sha256 = _sha256_bytes(inventory_contents)
    _require(
        inventory_sha256 == EXPECTED_AUDIO_INVENTORY_SHA256,
        f"Hindi audio inventory checksum changed: {inventory_sha256}",
    )
    inventory_path.write_bytes(inventory_contents)


def _stage_audio_archive(archive_path: Path, rows: list[HindiASRRow], audio_dir: Path, inventory_path: Path) -> None:
    expected = {row.filename: row for row in rows}
    rejected = {member.filename: member for member in REJECTED_AUDIO_MEMBERS}
    retained = _retained_rows(rows)
    seen: set[str] = set()
    unreferenced_members: list[str] = []
    inventory: dict[str, tuple[int, str, int]] = {}
    referenced_member_bytes = 0
    audio_dir.mkdir()

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(ARCHIVE_AUDIO_SUFFIX):
                continue
            filename = Path(member.name).name
            expected_member_name = f"audio/{filename}"
            if member.name != expected_member_name or filename not in expected:
                unreferenced_members.append(member.name)
                continue
            _require(filename not in seen, f"Duplicate referenced Hindi archive member: {member.name}")
            seen.add(filename)

            extracted = archive.extractfile(member)
            _require(extracted is not None, f"Could not read Hindi archive member: {member.name}")
            with extracted:
                payload = extracted.read()
            payload_size = len(payload)
            payload_sha256 = _sha256_bytes(payload)
            referenced_member_bytes += payload_size
            _require(
                payload_size == member.size,
                f"Hindi archive member size mismatch for {member.name}: {payload_size} != {member.size}",
            )

            rejected_member = rejected.get(filename)
            if rejected_member is not None:
                _require(
                    member.name == rejected_member.archive_member
                    and member.size == rejected_member.archive_size
                    and payload_sha256 == rejected_member.sha256,
                    f"Pinned rejected archive member changed: {member.name}",
                )
                _validate_rejected_payload(payload, rejected_member)
            else:
                row = expected[filename]
                decoded_frames = _decode_retained_audio(payload, row)
                with (audio_dir / filename).open("xb") as audio_file:
                    audio_file.write(payload)
                inventory[filename] = (payload_size, payload_sha256, decoded_frames)

            if len(seen) % 20_000 == 0:
                logger.info(f"Decoded {len(seen)}/{len(rows)} referenced Hindi FLAC members")

    missing = set(expected) - seen
    _require(not missing, f"Hindi archive is missing {len(missing)} referenced FLAC members")
    _require(
        referenced_member_bytes == EXPECTED_REFERENCED_MEMBER_BYTES,
        f"Referenced archive payload bytes changed: {referenced_member_bytes}",
    )

    _validate_unreferenced_members(unreferenced_members)
    _write_audio_inventory(retained, inventory, inventory_path)


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

    source_manifest_path = (dataset_path / MANIFEST_FILENAME).resolve()
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
        "audio_archive_bytes": AUDIO_ARCHIVE_BYTES,
        "audio_archive_filename": AUDIO_ARCHIVE_FILENAME,
        "audio_archive_sha256": AUDIO_ARCHIVE_SHA256,
        "audio_bytes": EXPECTED_AUDIO_BYTES,
        "audio_inventory_bytes": EXPECTED_AUDIO_INVENTORY_BYTES,
        "audio_inventory_schema": AUDIO_INVENTORY_SCHEMA,
        "audio_inventory_sha256": EXPECTED_AUDIO_INVENTORY_SHA256,
        "archive_audio_format": ARCHIVE_AUDIO_FORMAT,
        "archive_audio_suffix": ARCHIVE_AUDIO_SUFFIX,
        "canonical_manifest_bytes": EXPECTED_CANONICAL_MANIFEST_BYTES,
        "canonical_manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
        "channels": CHANNELS,
        "decoded_frames": EXPECTED_DECODED_FRAMES,
        "expected_num_rows": EXPECTED_NUM_ROWS,
        "expected_total_duration_ms": EXPECTED_TOTAL_DURATION_MS,
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "license": "Apache-2.0",
        "rejected_audio_members": [_rejected_member_record(member) for member in REJECTED_AUDIO_MEMBERS],
        "rejected_duration_ms": EXPECTED_REJECTED_DURATION_MS,
        "referenced_audio_bytes": EXPECTED_REFERENCED_MEMBER_BYTES,
        "sample_rate": SAMPLE_RATE,
        "source_manifest_audio_suffix": SOURCE_MANIFEST_AUDIO_SUFFIX,
        "source_manifest_bytes": SOURCE_MANIFEST_BYTES,
        "source_manifest_filename": SOURCE_MANIFEST_FILENAME,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_num_rows": EXPECTED_SOURCE_NUM_ROWS,
        "source_total_duration_ms": EXPECTED_SOURCE_TOTAL_DURATION_MS,
        "split": HF_SPLIT,
        "unreferenced_audio_count": EXPECTED_UNREFERENCED_AUDIO_COUNT,
        "unreferenced_audio_names_bytes": EXPECTED_UNREFERENCED_AUDIO_NAMES_BYTES,
        "unreferenced_audio_names_sha256": EXPECTED_UNREFERENCED_AUDIO_NAMES_SHA256,
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _control_file_contract() -> dict[str, dict[str, object]]:
    metadata_contents = _json_bytes(_metadata())
    return {
        AUDIO_INVENTORY_FILENAME: {
            "bytes": EXPECTED_AUDIO_INVENTORY_BYTES,
            "sha256": EXPECTED_AUDIO_INVENTORY_SHA256,
        },
        MANIFEST_FILENAME: {
            "bytes": EXPECTED_CANONICAL_MANIFEST_BYTES,
            "sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
        },
        METADATA_FILENAME: {
            "bytes": len(metadata_contents),
            "sha256": _sha256_bytes(metadata_contents),
        },
        STAGED_SOURCE_MANIFEST_FILENAME: {
            "bytes": SOURCE_MANIFEST_BYTES,
            "sha256": SOURCE_MANIFEST_SHA256,
        },
    }


def _dataset_receipt() -> dict[str, object]:
    return {
        "audio": {
            "channels": CHANNELS,
            "decoded_frames": EXPECTED_DECODED_FRAMES,
            "directory": "audio",
            "files": EXPECTED_NUM_ROWS,
            "format": ARCHIVE_AUDIO_FORMAT,
            "full_decode_during_staging": True,
            "inventory_bytes": EXPECTED_AUDIO_INVENTORY_BYTES,
            "inventory_sha256": EXPECTED_AUDIO_INVENTORY_SHA256,
            "payload_bytes": EXPECTED_AUDIO_BYTES,
            "sample_rate": SAMPLE_RATE,
        },
        "cohort": {
            "canonical_manifest_bytes": EXPECTED_CANONICAL_MANIFEST_BYTES,
            "canonical_manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "duration_ms": EXPECTED_TOTAL_DURATION_MS,
            "rows": EXPECTED_NUM_ROWS,
        },
        "control_files": _control_file_contract(),
        "dataset": "audio_indic_asr",
        "rejected_audio_members": [_rejected_member_record(member) for member in REJECTED_AUDIO_MEMBERS],
        "schema_version": DATASET_RECEIPT_SCHEMA_VERSION,
        "source": {
            "archive_bytes": AUDIO_ARCHIVE_BYTES,
            "archive_filename": AUDIO_ARCHIVE_FILENAME,
            "archive_sha256": AUDIO_ARCHIVE_SHA256,
            "hf_repo_id": HF_REPO_ID,
            "hf_revision": HF_REVISION,
            "license": "Apache-2.0",
            "manifest_bytes": SOURCE_MANIFEST_BYTES,
            "manifest_filename": SOURCE_MANIFEST_FILENAME,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "referenced_audio_bytes": EXPECTED_REFERENCED_MEMBER_BYTES,
            "rows": EXPECTED_SOURCE_NUM_ROWS,
            "split": HF_SPLIT,
            "total_duration_ms": EXPECTED_SOURCE_TOTAL_DURATION_MS,
        },
        "status": "complete",
        "unreferenced_archive_audio": {
            "count": EXPECTED_UNREFERENCED_AUDIO_COUNT,
            "sorted_names_bytes": EXPECTED_UNREFERENCED_AUDIO_NAMES_BYTES,
            "sorted_names_sha256": EXPECTED_UNREFERENCED_AUDIO_NAMES_SHA256,
        },
    }


def _verify_control_file(path: Path, contract: dict[str, object]) -> None:
    _require(path.is_file() and not path.is_symlink(), f"Required Indic ASR control file is missing: {path}")
    actual_bytes = path.stat().st_size
    _require(actual_bytes == contract["bytes"], f"Unexpected byte count for {path}: {actual_bytes}")
    actual_sha256 = _sha256_file(path)
    _require(actual_sha256 == contract["sha256"], f"Unexpected checksum for {path}: {actual_sha256}")


def _verify_audio_file_inventory(audio_dir: Path, inventory_path: Path) -> None:
    expected_sizes: dict[str, int] = {}
    with inventory_path.open(encoding="ascii") as inventory_file:
        for line_number, line in enumerate(inventory_file, start=1):
            try:
                relative_path, payload_bytes, _sha256, _decoded_frames = line.rstrip("\n").split("\t")
                filename = relative_path.removeprefix("audio/")
                payload_size = int(payload_bytes)
            except (TypeError, ValueError) as e:
                msg = f"Invalid Hindi audio inventory row {inventory_path}:{line_number}"
                raise RuntimeError(msg) from e
            _require(
                relative_path == f"audio/{filename}"
                and filename.endswith(ARCHIVE_AUDIO_SUFFIX)
                and "/" not in filename
                and filename not in expected_sizes
                and payload_size >= 0,
                f"Invalid or duplicate Hindi audio inventory row {inventory_path}:{line_number}",
            )
            expected_sizes[filename] = payload_size

    _require(
        len(expected_sizes) == EXPECTED_NUM_ROWS and sum(expected_sizes.values()) == EXPECTED_AUDIO_BYTES,
        "Hindi audio inventory logical totals do not match the checked-in contract",
    )
    with os.scandir(audio_dir) as audio_entries:
        for entry in audio_entries:
            _require(
                entry.is_file(follow_symlinks=False) and not entry.is_symlink(),
                f"Unexpected non-regular Hindi audio entry: {entry.path}",
            )
            expected_size = expected_sizes.pop(entry.name, None)
            _require(expected_size is not None, f"Unexpected Hindi audio file: {entry.path}")
            actual_size = entry.stat(follow_symlinks=False).st_size
            _require(
                actual_size == expected_size,
                f"Unexpected Hindi audio file size for {entry.path}: {actual_size} != {expected_size}",
            )
    _require(not expected_sizes, f"Hindi audio directory is missing {len(expected_sizes)} pinned files")


def verify_dataset(output_path: Path, *, receipt_only: bool = False) -> bool:
    """Verify receipt/controls and, by default, the exact audio name/size inventory."""
    try:
        _require(output_path.is_dir() and not output_path.is_symlink(), f"Hindi dataset is missing: {output_path}")
        required_entries = {
            AUDIO_INVENTORY_FILENAME,
            DATASET_RECEIPT_FILENAME,
            MANIFEST_FILENAME,
            METADATA_FILENAME,
            STAGED_SOURCE_MANIFEST_FILENAME,
            "audio",
        }
        actual_entries = {path.name for path in output_path.iterdir()}
        # Deterministic tutorial subsets may be added after publication; they are
        # derived artifacts and are deliberately outside the dataset receipt.
        missing_entries = required_entries - actual_entries
        _require(not missing_entries, f"Missing top-level Hindi dataset entries: {sorted(missing_entries)}")
        for extra_name in actual_entries - required_entries:
            extra_path = output_path / extra_name
            _require(
                extra_name.startswith("manifest-")
                and extra_name.endswith(".jsonl")
                and extra_path.is_file()
                and not extra_path.is_symlink(),
                f"Unexpected top-level Hindi dataset entry: {extra_path}",
            )

        audio_dir = output_path / "audio"
        _require(audio_dir.is_dir() and not audio_dir.is_symlink(), f"Hindi audio directory is missing: {audio_dir}")
        receipt_path = output_path / DATASET_RECEIPT_FILENAME
        _require(
            receipt_path.is_file() and not receipt_path.is_symlink(),
            f"Hindi dataset receipt is missing: {receipt_path}",
        )
        _require(
            receipt_path.read_bytes() == _json_bytes(_dataset_receipt()),
            "Hindi dataset receipt does not match the checked-in contract",
        )
        for filename, contract in _control_file_contract().items():
            _verify_control_file(output_path / filename, contract)
        if not receipt_only:
            _verify_audio_file_inventory(audio_dir, output_path / AUDIO_INVENTORY_FILENAME)
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Indic ASR dataset verification failed: {e}")
        return False

    verification_scope = "receipt/control files" if receipt_only else "receipt/control files and audio names/sizes"
    logger.success(
        f"Verified {verification_scope} for {EXPECTED_NUM_ROWS} unique Hindi clips / "
        f"{EXPECTED_TOTAL_DURATION_MS / 3_600_000:.4f} audio hours at {output_path}"
    )
    return True


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _publish_staging(staging_path: Path, output_path: Path, backup_path: Path) -> None:
    had_existing_output = _path_exists(output_path)
    if had_existing_output:
        os.replace(output_path, backup_path)
    try:
        os.replace(staging_path, output_path)
    except Exception:
        if had_existing_output and _path_exists(backup_path) and not _path_exists(output_path):
            os.replace(backup_path, output_path)
        raise

    if _path_exists(backup_path):
        try:
            _remove_path(backup_path)
        except OSError as e:
            logger.warning(f"Could not remove stale Hindi dataset backup {backup_path}: {e}")


def _recover_interrupted_publish(output_path: Path, staging_path: Path, backup_path: Path) -> None:
    if _path_exists(backup_path):
        if _path_exists(output_path):
            if verify_dataset(output_path):
                logger.warning(f"Removing stale Hindi dataset backup after completed publication: {backup_path}")
                _remove_path(backup_path)
            elif verify_dataset(backup_path):
                logger.warning(f"Rolling back invalid Hindi dataset publication from backup: {backup_path}")
                _remove_path(output_path)
                os.replace(backup_path, output_path)
            else:
                logger.warning(f"Removing invalid stale Hindi dataset backup: {backup_path}")
                _remove_path(backup_path)
        else:
            logger.warning(f"Restoring Hindi dataset publication interrupted after backup: {backup_path}")
            os.replace(backup_path, output_path)

    if _path_exists(staging_path):
        if not verify_dataset(output_path) and verify_dataset(staging_path):
            logger.warning(f"Publishing complete Hindi dataset staging left by an interrupted process: {staging_path}")
            _publish_staging(staging_path, output_path, backup_path)
        else:
            logger.warning(f"Removing stale or incomplete Hindi dataset staging: {staging_path}")
            _remove_path(staging_path)


def _download_pinned_sources(cache_dir: str) -> tuple[Path, Path]:
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
    _require(
        source_manifest_path.stat().st_size == SOURCE_MANIFEST_BYTES, "Pinned Hindi source manifest size mismatch"
    )
    _require(
        _sha256_file(source_manifest_path) == SOURCE_MANIFEST_SHA256,
        f"Unexpected checksum for {SOURCE_MANIFEST_FILENAME} at revision {HF_REVISION}",
    )
    _require(archive_path.stat().st_size == AUDIO_ARCHIVE_BYTES, "Pinned Hindi audio archive size mismatch")
    _require(
        _sha256_file(archive_path) == AUDIO_ARCHIVE_SHA256,
        f"Unexpected checksum for {AUDIO_ARCHIVE_FILENAME} at revision {HF_REVISION}",
    )
    return source_manifest_path, archive_path


def _prepare_dataset_locked(output_path: Path, cache_dir: str) -> None:
    staging_path = output_path.with_name(f".{output_path.name}.staging")
    backup_path = output_path.with_name(f".{output_path.name}.backup")
    _recover_interrupted_publish(output_path, staging_path, backup_path)
    if verify_dataset(output_path):
        logger.info(f"Reusing staged Hindi ASR dataset at {output_path}")
        return

    source_manifest_path, archive_path = _download_pinned_sources(cache_dir)
    if _path_exists(staging_path):
        _remove_path(staging_path)
    staging_path.mkdir()
    try:
        staged_source_manifest_path = staging_path / STAGED_SOURCE_MANIFEST_FILENAME
        shutil.copyfile(source_manifest_path, staged_source_manifest_path)
        rows = _parse_source_manifest(staged_source_manifest_path)
        retained = _retained_rows(rows)
        _stage_audio_archive(
            archive_path,
            rows,
            staging_path / "audio",
            staging_path / AUDIO_INVENTORY_FILENAME,
        )
        _write_manifest(retained, staging_path / MANIFEST_FILENAME)
        (staging_path / METADATA_FILENAME).write_bytes(_json_bytes(_metadata()))

        # The receipt is the commit marker and must be the final staging write.
        (staging_path / DATASET_RECEIPT_FILENAME).write_bytes(_json_bytes(_dataset_receipt()))
        _require(
            verify_dataset(staging_path), f"Completed Hindi ASR staging failed receipt verification: {staging_path}"
        )
        _publish_staging(staging_path, output_path, backup_path)
    except Exception:
        if _path_exists(staging_path):
            _remove_path(staging_path)
        raise

    _require(verify_dataset(output_path), f"Published Hindi ASR dataset failed verification: {output_path}")


def stage_dataset(output_path: Path, cache_dir: str) -> None:
    """Serialize, fully validate, and atomically publish the pinned Hindi cohort."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with lock_path.open("a") as lock_file:
        logger.info(f"Waiting for exclusive Hindi ASR setup lock: {lock_path}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _prepare_dataset_locked(output_path, cache_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="With --verify-only, skip the FLAC filename/size stat walk after validating receipt and control hashes",
    )
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
    if args.receipt_only and not args.verify_only:
        parser.error("--receipt-only requires --verify-only")

    output_path = args.output_path.resolve()
    if args.verify_only:
        if not verify_dataset(output_path, receipt_only=args.receipt_only):
            return 1
    else:
        stage_dataset(output_path, args.cache_dir)
    if args.subset_hours is not None:
        write_duration_subset(output_path, args.subset_manifest, args.subset_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
