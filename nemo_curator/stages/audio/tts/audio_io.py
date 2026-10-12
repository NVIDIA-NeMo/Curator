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

"""Segment audio access for TTS Granary manifests.

Granary sample rows hold a ``segments`` list; every segment's audio is one member of a tar archive addressed by
``GranaryHifi.AudioMetadata.{tar_key,tar_member,tar_member_offset,tar_member_bytes}``. Annotation stages read the
member payload in place (seek + read) instead of extracting the tar, as in the TTS Granary reference pipeline.
Segments may alternatively carry a plain ``audio_filepath`` (optionally with ``offset``/``duration`` in seconds).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.fields import get_dotted

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

TAR_KEY_FIELD = "GranaryHifi.AudioMetadata.tar_key"
TAR_MEMBER_FIELD = "GranaryHifi.AudioMetadata.tar_member"
TAR_OFFSET_FIELD = "GranaryHifi.AudioMetadata.tar_member_offset"
TAR_BYTES_FIELD = "GranaryHifi.AudioMetadata.tar_member_bytes"
PNC_TEXT_FIELD = "GranaryV2.pnc_text"
TN_RAW_FIELD = "GranaryV2.tn_raw"

# Annotation error codes (TTS Granary reference error ladder).
MISSING_TAR_MEMBER = "missing_tar_member"
MISSING_TAR_KEY = "missing_tar_key"
MISSING_TAR_OFFSETS = "missing_tar_offsets"
MISSING_AUDIO_TAR = "missing_audio_tar"
MISSING_AUDIO_FILE = "missing_audio_file"
AUDIO_LOAD_FAILED = "audio_load_failed"


class SegmentAudioError(Exception):
    """Audio of a segment could not be loaded; ``code`` is the annotation error string."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


AUDIO_SOURCES = ("original", "enhanced")
ENHANCED_AUDIO_KEY = "enhanced_audio"
MISSING_ENHANCED_AUDIO = "missing_enhanced_audio"


@dataclass(frozen=True)
class TarAudioConfig:
    """Where a segment's audio comes from.

    ``tar_key`` values: absolute keys are used as-is, others are joined to ``tar_root`` after stripping the first
    matching prefix in ``strip_tar_key_prefixes``.

    ``audio_source="enhanced"`` reads the enhanced copy written by an enhancement stage (StudioVoice or ReUse /
    Resemble-Enhance) under ``enhanced_key`` (``{audio_filepath, sampling_rate, duration, method, error}``) instead
    of the original tar member. Segments without a usable enhanced copy raise ``missing_enhanced_audio`` unless
    ``enhanced_fallback`` is set, in which case the original audio is used.
    """

    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = ENHANCED_AUDIO_KEY
    enhanced_fallback: bool = False

    def __post_init__(self) -> None:
        if self.audio_source not in AUDIO_SOURCES:
            msg = f"audio_source must be one of {AUDIO_SOURCES}, got {self.audio_source!r}"
            raise ValueError(msg)


def make_audio_config(stage: object) -> TarAudioConfig:
    """Build the audio config from a stage's ``tar_root`` / ``strip_tar_key_prefixes`` / ``audio_source`` /
    ``enhanced_key`` / ``enhanced_fallback`` fields (missing fields take the defaults)."""
    return TarAudioConfig(
        tar_root=getattr(stage, "tar_root", None),
        strip_tar_key_prefixes=tuple(getattr(stage, "strip_tar_key_prefixes", ()) or ()),
        audio_source=getattr(stage, "audio_source", "original"),
        enhanced_key=getattr(stage, "enhanced_key", ENHANCED_AUDIO_KEY),
        enhanced_fallback=bool(getattr(stage, "enhanced_fallback", False)),
    )


def enhanced_audio_path(segment: dict[str, Any], enhanced_key: str = ENHANCED_AUDIO_KEY) -> str | None:
    """Path of a segment's usable enhanced copy (annotation without error and with an existing file), else None."""
    annotation = get_dotted(segment, enhanced_key)
    if not isinstance(annotation, dict) or annotation.get("error") is not None:
        return None
    path = annotation.get("audio_filepath")
    return str(path) if path and Path(str(path)).is_file() else None


def resolve_tar_path(tar_key: str | None, config: TarAudioConfig | None = None) -> Path | None:
    if not tar_key:
        return None
    tar_path = Path(tar_key)
    if tar_path.is_absolute() or config is None or not config.tar_root:
        return tar_path
    root = Path(config.tar_root).expanduser()
    tar_key_norm = str(tar_key).lstrip("/")
    for prefix in config.strip_tar_key_prefixes:
        prefix_norm = str(prefix).strip("/")
        if tar_key_norm == prefix_norm:
            return root
        if tar_key_norm.startswith(prefix_norm + "/"):
            return root / tar_key_norm[len(prefix_norm) + 1 :]
    return root / tar_key_norm


def resample(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Resample a mono float32 signal (librosa, then scipy polyphase, then linear interpolation)."""
    if sample_rate == target_sample_rate:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.debug("librosa resample unavailable ({}), falling back to scipy", exc)
    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(sample_rate, target_sample_rate)
        return resample_poly(audio, target_sample_rate // gcd, sample_rate // gcd).astype(np.float32)
    except Exception:  # noqa: BLE001
        target_len = max(1, round(audio.shape[-1] * target_sample_rate / sample_rate))
        positions = np.linspace(0, audio.shape[-1] - 1, target_len)
        return np.interp(positions, np.arange(audio.shape[-1]), audio).astype(np.float32)


def decode_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if audio.ndim == 2:  # noqa: PLR2004
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def read_tar_member(tar_path: Path, offset: int, size: int) -> bytes:
    with tar_path.open("rb") as src:
        src.seek(offset)
        return src.read(size)


def tar_member_location(segment: dict[str, Any], config: TarAudioConfig | None = None) -> tuple[Path, int, int, str]:
    """Return ``(tar_path, offset, size, member)`` or raise ``SegmentAudioError`` (reference error ladder)."""
    tar_member = get_dotted(segment, TAR_MEMBER_FIELD)
    if not tar_member:
        raise SegmentAudioError(MISSING_TAR_MEMBER)
    tar_key = get_dotted(segment, TAR_KEY_FIELD)
    tar_path = resolve_tar_path(str(tar_key) if tar_key else None, config)
    if tar_path is None:
        raise SegmentAudioError(MISSING_TAR_KEY)
    try:
        offset = int(get_dotted(segment, TAR_OFFSET_FIELD))
        size = int(get_dotted(segment, TAR_BYTES_FIELD))
    except (TypeError, ValueError) as exc:
        raise SegmentAudioError(MISSING_TAR_OFFSETS) from exc
    if not tar_path.is_file():
        raise SegmentAudioError(MISSING_AUDIO_TAR, str(tar_path))
    return tar_path, offset, size, str(tar_member)


def load_segment_audio(
    segment: dict[str, Any],
    target_sample_rate: int | None = None,
    config: TarAudioConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Load a segment as mono float32, resampled to ``target_sample_rate`` when given.

    Uses the Granary tar-member address when present, otherwise ``audio_filepath`` (+ ``offset``/``duration``).
    With ``config.audio_source == "enhanced"`` the enhanced copy is read instead (see ``TarAudioConfig``).
    """
    if config is not None and config.audio_source == "enhanced":
        path = enhanced_audio_path(segment, config.enhanced_key)
        if path is not None:
            audio, sample_rate = _load_file({"audio_filepath": path})
            if target_sample_rate is not None and sample_rate != target_sample_rate:
                audio = resample(audio, sample_rate, int(target_sample_rate))
                sample_rate = int(target_sample_rate)
            return audio, sample_rate
        if not config.enhanced_fallback:
            raise SegmentAudioError(MISSING_ENHANCED_AUDIO)
    if get_dotted(segment, TAR_MEMBER_FIELD) or get_dotted(segment, TAR_KEY_FIELD):
        tar_path, offset, size, _ = tar_member_location(segment, config)
        payload = read_tar_member(tar_path, offset, size)
        if len(payload) != size:
            raise SegmentAudioError(AUDIO_LOAD_FAILED, f"short read {len(payload)}/{size} bytes from {tar_path}")
        try:
            audio, sample_rate = decode_audio_bytes(payload)
        except Exception as exc:
            raise SegmentAudioError(AUDIO_LOAD_FAILED, str(exc)) from exc
    elif segment.get("audio_filepath"):
        audio, sample_rate = _load_file(segment)
    else:
        raise SegmentAudioError(MISSING_TAR_MEMBER)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        audio = resample(audio, sample_rate, int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return audio, sample_rate


def _load_file(segment: dict[str, Any]) -> tuple[np.ndarray, int]:
    import soundfile as sf

    path = Path(str(segment["audio_filepath"])).expanduser()
    if not path.is_file():
        raise SegmentAudioError(MISSING_AUDIO_FILE, str(path))
    try:
        info = sf.info(str(path))
        start = round(float(segment.get("offset") or 0.0) * info.samplerate)
        duration = segment.get("duration")
        frames = round(float(duration) * info.samplerate) if duration is not None and "offset" in segment else -1
        audio, sample_rate = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=False)
    except Exception as exc:
        raise SegmentAudioError(AUDIO_LOAD_FAILED, str(exc)) from exc
    if audio.ndim == 2:  # noqa: PLR2004
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def iter_segments(row: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, segment)`` for a sample row with ``segments``; a flat row is treated as one segment."""
    segments = row.get("segments")
    if isinstance(segments, list):
        for index, segment in enumerate(segments):
            if isinstance(segment, dict):
                yield index, segment
        return
    yield 0, row


def segment_text(segment: dict[str, Any], key: str, fallbacks: Sequence[str] = ()) -> str | None:
    for candidate in (key, *fallbacks):
        value = get_dotted(segment, candidate)
        if isinstance(value, str) and value.strip():
            return value
    return None


def sample_id(row: dict[str, Any], line_index: int | None = None) -> str:
    """Recording-level id used to prefix speaker ids (reference ``fields.speaker_id_source`` fallbacks)."""
    for key in ("orig_sample_id", "sample_id", "id", "audio_id", "audio_filepath"):
        value = row.get(key)
        if value:
            return str(value)
    return f"sample_{line_index or 0:08d}"
