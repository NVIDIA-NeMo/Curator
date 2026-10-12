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

"""Atomic, resumable materialization of one GranaryHiFi SHAR shard.

Completion is tracked by a per-shard marker written *last* (after the three ``os.replace`` calls)::

    <rate_dir>/.complete/shard.NNNNNN.json   {"digest": <planned-entries digest>, "cut_ids": [...], "skipped": [...]}

A shard is reused only when its marker digest equals the digest of the currently planned entries (cuts, context
pairing, annotations, audio addresses) and the cuts/target/context triple is positionally aligned with the marker ids.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import decode_audio_bytes, read_tar_member

if TYPE_CHECKING:
    from nemo_curator.stages.audio.tts.shar import SharEntry

REGISTRY_NAME = "shar_registry.json"
REGISTRY_VERSION = 2
AUDIO_EXTENSIONS = {".flac", ".mp3", ".ogg", ".opus", ".wav"}
MARKER_DIR = ".complete"
DURATION_TOLERANCE_SEC = 0.1
# skip reasons recorded in the shard marker / registry
AUDIO_READ_FAILED = "audio_read_failed"
SHORT_READ = "short_read"
AUDIO_DECODE_FAILED = "audio_decode_failed"
EMPTY_AUDIO = "empty_audio"
DURATION_MISMATCH = "duration_mismatch"
_CUT_ALIGNMENT_TOLERANCE_SEC = 0.01
_HASH_CHUNK = 8 << 20


class EntryAudioError(Exception):
    """Audio of a planned entry is unusable; the entry is skipped and recorded with ``reason``."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# --------------------------------------------------------------------------------------------- small io helpers


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load a shard registry, rejecting unsupported versions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != REGISTRY_VERSION or not isinstance(payload.get("shards"), list):
        msg = f"Invalid or outdated SHAR registry (expected version {REGISTRY_VERSION}): {path}"
        raise ValueError(msg)
    return payload


def read_cut_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line)["id"] for line in stream if line.strip()]


def read_tar_ids(path: Path, payload_extensions: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Return ``(payload ids, .json metadata ids)`` of a SHAR tar in member order."""
    payload_extensions = payload_extensions or AUDIO_EXTENSIONS
    payload, metadata = [], []
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            stem = str(Path(member.name).with_suffix(""))
            if suffix in payload_extensions:
                payload.append(stem)
            elif suffix == ".json":
                metadata.append(stem)
    return payload, metadata


def file_digest(*paths: Path) -> str:
    """sha256 over the bytes of the given files, in order."""
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | None) -> list[Any] | None:
    """``[path, size, mtime_ns]`` of an audio file so a regenerated enhanced copy changes the digest."""
    if path is None:
        return None
    try:
        stat = Path(path).stat()
    except OSError:
        return [path, None, None]
    return [path, stat.st_size, stat.st_mtime_ns]


def entries_digest(entries: list[SharEntry]) -> str:
    """Content digest of planned entries: any change in cuts, annotations, pairing or audio address changes it."""
    payload = [
        [
            e.target_cut,
            e.context_cut,
            e.tar_path,
            e.target_offset,
            e.target_size,
            e.context_offset,
            e.context_size,
            e.audio_source,
            _file_identity(e.target_file),
            _file_identity(e.context_file),
        ]
        for e in entries
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- shard layout


def shard_paths(rate_dir: Path, shard_id: int) -> tuple[Path, Path, Path]:
    return (
        rate_dir / "cuts" / f"cuts.{shard_id:06d}.jsonl.gz",
        rate_dir / "target_audio" / f"recording.{shard_id:06d}.tar",
        rate_dir / "context_audio" / f"recording.{shard_id:06d}.tar",
    )


def marker_path(rate_dir: Path, shard_id: int) -> Path:
    return rate_dir / MARKER_DIR / f"shard.{shard_id:06d}.json"


def triple_is_aligned(rate_dir: Path, shard_id: int, expected_ids: list[str] | None = None) -> bool:
    """Whether a shard's cuts and audio tars contain exactly ``expected_ids`` in order."""
    paths = shard_paths(rate_dir, shard_id)
    if not all(path.is_file() for path in paths):
        return False
    try:
        cut_ids = read_cut_ids(paths[0])
        target_ids, target_meta = read_tar_ids(paths[1])
        context_ids, context_meta = read_tar_ids(paths[2])
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        logger.warning("Shard {} in {} is unreadable: {}", shard_id, rate_dir, exc)
        return False
    if expected_ids is not None and cut_ids != expected_ids:
        return False
    return cut_ids == target_ids == target_meta == context_ids == context_meta


def shard_marker(rate_dir: Path, shard_id: int, digest: str | None = None) -> dict[str, Any] | None:
    """The completion marker if the shard is complete (and matches ``digest`` when given), else ``None``."""
    marker = read_json(marker_path(rate_dir, shard_id))
    if marker is None or (digest is not None and marker.get("digest") != digest):
        return None
    cut_ids = marker.get("cut_ids") or []
    if cut_ids and not triple_is_aligned(rate_dir, shard_id, cut_ids):
        return None
    return marker


def shard_is_aligned(rate_dir: Path, shard_id: int, expected_ids: list[str] | None = None) -> bool:
    """True when the shard has a completion marker and its triple is aligned (with ``expected_ids`` if given)."""
    marker = shard_marker(rate_dir, shard_id)
    if marker is None:
        return False
    return expected_ids is None or marker.get("cut_ids") == expected_ids


# --------------------------------------------------------------------------------------------- materialization


def _load_checked(
    tar_path: str, offset: int, size: int, expected_duration: float, file_path: str | None = None
) -> tuple[Any, int]:
    """Load a tar byte range (``file_path=None``) or a whole audio file, with read-length and duration checks."""
    try:
        payload = Path(file_path).read_bytes() if file_path else read_tar_member(Path(tar_path), offset, size)
    except OSError as exc:
        raise EntryAudioError(AUDIO_READ_FAILED, str(exc)) from exc
    if file_path is None and len(payload) != size:
        raise EntryAudioError(SHORT_READ, f"{tar_path}@{offset}: read {len(payload)} of {size} bytes")
    try:
        audio, sampling_rate = decode_audio_bytes(payload)
    except Exception as exc:
        raise EntryAudioError(AUDIO_DECODE_FAILED, str(exc)) from exc
    if audio.size == 0:
        raise EntryAudioError(EMPTY_AUDIO, f"{tar_path}@{offset}")
    decoded = audio.shape[-1] / float(sampling_rate)
    if abs(decoded - float(expected_duration)) > DURATION_TOLERANCE_SEC:
        raise EntryAudioError(
            DURATION_MISMATCH, f"{tar_path}@{offset}: decoded {decoded:.3f}s vs manifest {expected_duration:.3f}s"
        )
    return audio[None, :], sampling_rate


def _align_cut_to_audio(cut: Any, num_samples: int, sampling_rate: int) -> Any:  # noqa: ANN401
    from lhotse import fastcopy

    duration = num_samples / float(sampling_rate)
    if abs(float(cut.duration) - duration) > _CUT_ALIGNMENT_TOLERANCE_SEC:
        logger.debug("Adjusting cut {} duration {:.4f} -> {:.4f}", cut.id, float(cut.duration), duration)
    recording = fastcopy(cut.recording, sampling_rate=int(sampling_rate), num_samples=num_samples, duration=duration)
    supervisions = [fastcopy(s, duration=min(float(s.duration), duration)) for s in cut.supervisions]
    return fastcopy(cut, duration=duration, recording=recording, supervisions=supervisions)


def _shar_placeholder(cut: Any) -> Any:  # noqa: ANN401
    from lhotse import AudioSource, compute_num_samples, fastcopy

    return fastcopy(
        cut.recording,
        id=cut.id,
        sources=[AudioSource(type="shar", channels=cut.recording.channel_ids, source="")],
        transforms=None,
        duration=cut.duration,
        num_samples=compute_num_samples(cut.duration, cut.recording.sampling_rate),
    )


def _write_triple(entries: list[SharEntry], tmp_dir: Path, shard_id: int, audio_format: str) -> tuple[list, list]:
    from lhotse import MonoCut
    from lhotse.shar.writers import AudioTarWriter, JsonlShardWriter

    shard_size = max(1, len(entries))
    written: list[str] = []
    skipped: list[dict[str, str]] = []
    with (
        JsonlShardWriter(
            str(tmp_dir / "cuts" / "cuts.%06d.jsonl.gz"), shard_size=shard_size, shard_offset=shard_id
        ) as cut_writer,
        AudioTarWriter(
            str(tmp_dir / "target_audio" / "recording.%06d.tar"),
            shard_size=shard_size,
            format=audio_format,
            shard_offset=shard_id,
        ) as target_writer,
        AudioTarWriter(
            str(tmp_dir / "context_audio" / "recording.%06d.tar"),
            shard_size=shard_size,
            format=audio_format,
            shard_offset=shard_id,
        ) as context_writer,
    ):
        for entry in entries:
            target_cut = MonoCut.from_dict(entry.target_cut)
            context_cut = MonoCut.from_dict(entry.context_cut)
            try:
                target_audio, target_sr = _load_checked(
                    entry.tar_path, entry.target_offset, entry.target_size, target_cut.duration, entry.target_file
                )
                context_audio, context_sr = _load_checked(
                    entry.tar_path, entry.context_offset, entry.context_size, context_cut.duration, entry.context_file
                )
            except EntryAudioError as exc:
                logger.warning("Skipping cut {} in shard {}: {}", entry.cut_id, shard_id, exc)
                skipped.append({"cut_id": entry.cut_id, "reason": exc.reason, "detail": str(exc)})
                continue
            target_cut = _align_cut_to_audio(target_cut, target_audio.shape[-1], target_sr)
            context_cut = _align_cut_to_audio(context_cut, context_audio.shape[-1], context_sr)
            target_writer.write(
                key=target_cut.id, value=target_audio, sampling_rate=target_sr, manifest=_shar_placeholder(target_cut)
            )
            context_writer.write(
                key=target_cut.id,
                value=context_audio,
                sampling_rate=context_sr,
                manifest=_shar_placeholder(context_cut),
            )
            cut_writer.write(target_cut)
            written.append(target_cut.id)
    return written, skipped


def materialize_shard(
    entries: list[SharEntry],
    rate_dir: Path,
    shard_id: int,
    audio_format: str = "wav",
    digest: str | None = None,
) -> dict[str, Any]:
    """Write one aligned cuts/target/context triple and its completion marker; returns the marker payload.

    Entries whose audio is unreadable, truncated (read length != ``tar_member_bytes``) or whose decoded duration
    differs from the manifest by more than 0.1 s are skipped and recorded under ``skipped``. A shard with no usable
    entry gets a marker with empty ``cut_ids`` and no stream files.
    """
    digest = digest or entries_digest(entries)
    marker = marker_path(rate_dir, shard_id)
    marker.unlink(missing_ok=True)
    tmp_dir = rate_dir / ".tmp_materialize" / f"shard_{shard_id:06d}_pid_{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for sub in ("cuts", "target_audio", "context_audio"):
        (tmp_dir / sub).mkdir(parents=True, exist_ok=True)
    try:
        written, skipped = _write_triple(entries, tmp_dir, shard_id, audio_format)
        final_paths = shard_paths(rate_dir, shard_id)
        if written:
            if not triple_is_aligned(tmp_dir, shard_id, written):
                msg = f"Materialized shard {shard_id} in {rate_dir} is not aligned"
                raise ValueError(msg)
            for tmp_path, final_path in zip(shard_paths(tmp_dir, shard_id), final_paths, strict=True):
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_path, final_path)
        else:
            for final_path in final_paths:
                final_path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        parent = tmp_dir.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    by_id = {e.cut_id: e for e in entries}
    payload = {
        "digest": digest,
        "cut_ids": written,
        "skipped": skipped,
        "speakers": {cut_id: by_id[cut_id].speaker_id for cut_id in written},
        "context_speakers": {cut_id: by_id[cut_id].context_speaker_id for cut_id in written},
    }
    write_json_atomic(marker, payload)
    return payload
