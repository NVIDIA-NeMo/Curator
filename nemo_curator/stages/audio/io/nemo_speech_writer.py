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

"""Write atomic Opus clips and resumable per-shard NeMo manifests.

Each writer task atomically persists its Opus clip and a small row receipt.
After ``Pipeline.run()`` returns, :func:`finalize_nemo_speech_output` validates
all receipts, atomically materializes one JSONL manifest per input shard, and
only then creates the shard's ``.done`` marker.  Driver-side finalization is
intentional: it prevents a marker from becoming visible while fan-out children
are still in flight and works for executors that do not call stage teardown.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io._nemo_speech_paths import contained_output_path, validate_local_output_dir
from nemo_curator.stages.audio.io._nemo_speech_state import (
    SHARD_REGISTRATION_VERSION,
    register_nemo_speech_shard,
    shard_registration_path,
    shard_registration_root,
)
from nemo_curator.stages.audio.io.shard_key import validate_shard_key
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.atomic_io import (
    fsync_directory,
    write_json_atomically,
    write_json_atomically_if_absent,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_RECEIPT_VERSION = 1
_OWNER_VERSION = 1
_MARKER_VERSION = 1
_RECEIPT_STATUSES = ("success", "placeholder")
_WAVEFORM_2D = 2
_RANDOM_TASK_ID = re.compile(r"r[0-9a-f]{32}", re.IGNORECASE)


def _row_root(output_dir: Path) -> Path:
    return contained_output_path(output_dir, ".nemo_curator/nemo_speech_rows")


def _has_random_task_ancestry(task_id: str) -> bool:
    return bool(_RANDOM_TASK_ID.fullmatch(task_id.split("_", 1)[0]))


def _receipt_dir(output_dir: Path, shard_key: str) -> Path:
    relative = PurePosixPath(".nemo_curator/nemo_speech_rows") / validate_shard_key(shard_key)
    return contained_output_path(output_dir, relative)


def _owner_path(
    output_dir: Path,
    relative_audio_path: PurePosixPath,
    opus_sha256: str | None = None,
) -> Path:
    owner_id = hashlib.sha256(relative_audio_path.as_posix().encode()).hexdigest()
    filename = f"{owner_id}.{opus_sha256}.json" if opus_sha256 else f"{owner_id}.json"
    return contained_output_path(
        output_dir,
        f".nemo_curator/nemo_speech_audio_owners/{owner_id[:2]}/{owner_id[2:4]}/{filename}",
    )


def _safe_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or PurePosixPath(value.replace("\\", "/")).is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = f"{field_name} must be a safe relative path, got {value!r}"
        raise ValueError(msg)
    return path


def _audio_relative_path(value: str, shard_key: str) -> PurePosixPath:
    path = _safe_relative_path(value, field_name="output_audio_filepath")
    shard_root = PurePosixPath(validate_shard_key(shard_key))
    try:
        remainder = path.relative_to(shard_root)
    except ValueError as error:
        msg = f"output_audio_filepath must be below shard directory {shard_root.as_posix()!r}, got {value!r}"
        raise ValueError(msg) from error
    if not remainder.parts or path.suffix.lower() != ".opus":
        msg = f"output_audio_filepath must name an .opus file below {shard_root.as_posix()!r}"
        raise ValueError(msg)
    return path


def _manifest_audio_path(relative_audio_path: PurePosixPath, shard_key: str) -> PurePosixPath:
    """Return the audio path as resolved from the shard manifest's parent."""

    manifest_parent = PurePosixPath(validate_shard_key(shard_key)).parent
    if not manifest_parent.parts:
        return relative_audio_path
    try:
        return relative_audio_path.relative_to(manifest_parent)
    except ValueError as error:
        msg = (
            f"output_audio_filepath must be below manifest parent {manifest_parent.as_posix()!r}, "
            f"got {relative_audio_path.as_posix()!r}"
        )
        raise ValueError(msg) from error


def _clean_source_parts(parts: Iterable[str]) -> list[str]:
    cleaned = []
    for part in parts:
        if part in {"", ".", ".."}:
            continue
        cleaned.append(part.replace("\\", "_").replace(":", "_"))
    return cleaned or ["audio"]


def _source_output_stem(source: str) -> PurePosixPath:
    """Create a bounded, path-safe stem without collapsing distinct sources."""

    if "://" in source:
        parsed = urlsplit(source)
        parts = _clean_source_parts([parsed.netloc, *PurePosixPath(parsed.path.lstrip("/")).parts])
        last = str(PurePosixPath(parts[-1]).with_suffix("")) or "audio"
        return PurePosixPath(*parts[:-1], last)
    if os.path.isabs(source):
        stem = Path(source).stem or "audio"
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:8]
        return PurePosixPath(f"{stem}_{source_hash}")
    parts = _clean_source_parts(PurePosixPath(source.replace("\\", "/")).parts)
    last = str(PurePosixPath(parts[-1]).with_suffix("")) or "audio"
    return PurePosixPath(*parts[:-1], last)


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    if value is None or isinstance(value, (bool, int, float, str)):
        result = value
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, np.generic):
        result = value.item()
    elif isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, dict):
        result = {str(key): _json_safe(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_json_safe(item) for item in value]
    elif hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        result = value.detach().cpu().tolist()
    else:
        msg = f"Value of type {type(value).__name__} is not JSON serializable"
        raise TypeError(msg)
    return result


def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        with contextlib.suppress(OSError):
            fsync_directory(path.parent)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _mono_float32(waveform: Any) -> np.ndarray:  # noqa: ANN401
    if hasattr(waveform, "detach") and hasattr(waveform, "cpu"):
        waveform = waveform.detach().cpu().numpy()
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == _WAVEFORM_2D:
        array = array.mean(axis=0) if array.shape[0] <= array.shape[1] else array.mean(axis=1)
    else:
        array = array.squeeze()
    if array.ndim != 1:
        msg = f"Expected a mono or two-dimensional waveform, got shape {array.shape}"
        raise ValueError(msg)
    return np.ascontiguousarray(array, dtype=np.float32)


def _opus_header_is_valid(path: Path, expected_sample_rate: int) -> bool:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(64 * 1024)
    except OSError:
        return False
    header_index = prefix.find(b"OpusHead")
    if not prefix.startswith(b"OggS") or header_index < 0 or len(prefix) < header_index + 19:
        return False
    channels = prefix[header_index + 9]
    input_sample_rate = int.from_bytes(prefix[header_index + 12 : header_index + 16], "little")
    return channels == 1 and input_sample_rate in {0, expected_sample_rate}


def _sample_count_matches(observed: int, expected: int, sample_rate: int) -> bool:
    # Opus packets can represent up to 60 ms. Container probes may report one
    # packet of encoder padding, but a durable output must never be shorter or
    # longer by more than that bound.
    tolerance = max(1, round(sample_rate * 0.06))
    return expected > 0 and abs(observed - expected) <= tolerance


def _ffprobe_valid_opus(path: Path, expected_sample_rate: int, expected_samples: int) -> bool:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        probe = subprocess.run(  # noqa: S603
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,channels:format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            payload = json.loads(probe.stdout)
            streams = payload.get("streams") or []
            duration = float((payload.get("format") or {}).get("duration") or 0.0)
            metadata_valid = (
                probe.returncode == 0
                and len(streams) == 1
                and streams[0].get("codec_name") == "opus"
                and int(streams[0].get("channels") or 0) == 1
                and duration > 0
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        ffmpeg_path = shutil.which("ffmpeg")
        if not metadata_valid or not ffmpeg_path:
            return False
        with tempfile.TemporaryFile() as decoded:
            result = subprocess.run(  # noqa: S603
                [
                    ffmpeg_path,
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(expected_sample_rate),
                    "-f",
                    "s16le",
                    "-c:a",
                    "pcm_s16le",
                    "pipe:1",
                ],
                check=False,
                stdout=decoded,
                stderr=subprocess.DEVNULL,
            )
            decoded.seek(0, os.SEEK_END)
            decoded_bytes = decoded.tell()
        return (
            result.returncode == 0
            and decoded_bytes % 2 == 0
            and _sample_count_matches(decoded_bytes // 2, expected_samples, expected_sample_rate)
        )
    return False


def _soundfile_decoded_samples(path: Path, expected_sample_rate: int, expected_samples: int) -> int:
    tolerance = max(1, round(expected_sample_rate * 0.06))
    with tempfile.TemporaryFile() as backing:
        buffer = np.memmap(backing, dtype=np.int16, mode="w+", shape=(expected_samples + tolerance + 1,))
        decoded, _ = sf.read(path, out=buffer)
        observed = len(decoded)
        del decoded, buffer
    return observed


def _valid_opus(path: Path, expected_sample_rate: int, expected_samples: int) -> bool:
    """Return whether ``path`` is complete mono Opus with the expected extent."""

    try:
        info = sf.info(path)
        if (
            info.format == "OGG"
            and info.subtype == "OPUS"
            and info.channels == 1
            and info.samplerate == expected_sample_rate
            and info.frames > 0
            and _sample_count_matches(info.frames, expected_samples, expected_sample_rate)
        ):
            decoded_samples = _soundfile_decoded_samples(path, expected_sample_rate, expected_samples)
            if _sample_count_matches(decoded_samples, expected_samples, expected_sample_rate):
                return True
    except (OSError, RuntimeError, sf.SoundFileError):
        pass
    return _opus_header_is_valid(path, expected_sample_rate) and _ffprobe_valid_opus(
        path,
        expected_sample_rate,
        expected_samples,
    )


def _claim_audio_path(output_dir: Path, claim: dict[str, Any], opus_sha256: str | None = None) -> None:
    relative = PurePosixPath(str(claim["audio_filepath"]))
    claim_path = _owner_path(output_dir, relative, opus_sha256)
    created = write_json_atomically_if_absent(claim_path, claim, separators=(",", ":"))
    if created:
        return
    try:
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Invalid NeMo speech audio ownership record {claim_path}: {error}"
        raise ValueError(msg) from error
    if existing != claim:
        msg = f"Conflicting NeMo speech outputs claim {relative.as_posix()!r}"
        raise ValueError(msg)


def _opus_matches_claim(output_dir: Path, path: Path, claim: dict[str, Any]) -> bool:
    try:
        digest = _sha256_file(path)
        relative = PurePosixPath(str(claim["audio_filepath"]))
        persisted = json.loads(_owner_path(output_dir, relative, digest).read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError):
        return False
    return persisted == claim


def _write_receipt_candidate(path: Path, receipt: dict[str, Any]) -> None:
    """Create an immutable receipt candidate or verify an exact replay."""

    created = write_json_atomically_if_absent(path, receipt, separators=(",", ":"))
    if created:
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Invalid NeMo speech row receipt {path}: {error}"
        raise ValueError(msg) from error
    if existing != receipt:
        msg = f"Conflicting NeMo speech row receipt candidate: {path}"
        raise ValueError(msg)


@dataclass
class NeMoSpeechWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Persist one Opus/manifest-row receipt for each ``AudioTask``.

    This is a terminal sink: it persists each task and returns no waveform to
    the driver. The output directory must be an absolute path on a shared
    POSIX-style filesystem visible at the same path to every worker. Call
    :func:`finalize_nemo_speech_output` after a successful pipeline run to
    create manifests and completion markers.

    Args:
        output_dir: Root for Opus clips, manifests, receipts, and markers.
        target_sample_rate: Required sample rate for waveform-bearing rows.
        waveform_key: Task-data key containing the waveform.
        sample_rate_key: Task-data key containing its sample rate.
        writer_concurrency: Number of stateless writer actors.
        save_audio: Encode Opus when true. When false, preserve the original
            audio reference in a manifest-only row.
    """

    output_dir: str
    target_sample_rate: int = 16000
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    writer_concurrency: int = 1
    save_audio: bool = True
    name: str = "nemo_speech_writer"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _encoder: str = field(default="", init=False, repr=False)
    _ffmpeg_path: str = field(default="", init=False, repr=False)
    is_sink_stage = True

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for NeMoSpeechWriterStage"
            raise ValueError(msg)
        validate_local_output_dir(self.output_dir)
        if self.target_sample_rate <= 0:
            msg = "target_sample_rate must be positive"
            raise ValueError(msg)
        if self.writer_concurrency < 1:
            msg = "writer_concurrency must be at least 1"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        # read_error/vad_empty rows intentionally have no waveform.
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return self.writer_concurrency

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._prepare_output()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._prepare_output()

    def _prepare_output(self) -> None:
        output_root = validate_local_output_dir(self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        _row_root(output_root).mkdir(parents=True, exist_ok=True)
        if not self.save_audio:
            self._encoder = "disabled"
        elif sf.check_format("OGG", "OPUS"):
            self._encoder = "soundfile"
        elif (ffmpeg_path := shutil.which("ffmpeg")) and shutil.which("ffprobe"):
            probe = subprocess.run(  # noqa: S603
                [ffmpeg_path, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            if "libopus" not in probe.stdout:
                msg = "ffmpeg is present but its libopus encoder is unavailable"
                raise RuntimeError(msg)
            self._encoder = "ffmpeg"
            self._ffmpeg_path = ffmpeg_path
        else:
            msg = "Opus output requires libsndfile OGG/OPUS support or ffmpeg/libopus plus ffprobe"
            raise RuntimeError(msg)

    def _encode_opus(self, waveform: np.ndarray, sample_rate: int) -> bytes:
        if self._encoder == "soundfile":
            stream = io.BytesIO()
            sf.write(stream, waveform, sample_rate, format="OGG", subtype="OPUS")
            return stream.getvalue()
        if self._encoder == "ffmpeg":
            result = subprocess.run(  # noqa: S603
                [
                    self._ffmpeg_path,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "libopus",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                input=waveform.tobytes(),
                check=False,
                capture_output=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", "replace")[-500:]
                msg = f"ffmpeg Opus encoding failed: {stderr}"
                raise RuntimeError(msg)
            return result.stdout
        msg = "NeMoSpeechWriterStage.setup() must run before audio is encoded"
        raise RuntimeError(msg)

    @staticmethod
    def _input_id(task: AudioTask) -> str:
        value = task._metadata.get("_shard_input_id")
        if value is not None and str(value):
            return str(value)
        msg = "Writer tasks require a non-empty _shard_input_id from the shard reader"
        raise ValueError(msg)

    @staticmethod
    def _output_slot(task: AudioTask) -> str:
        if task.task_id and not _has_random_task_ancestry(task.task_id):
            return task.task_id
        explicit = task._metadata.get("_shard_output_id")
        if explicit is not None and str(explicit):
            return str(explicit)
        if task.task_id:
            msg = "Writer received a non-deterministic framework task_id; set a stable _shard_output_id upstream"
        else:
            msg = "Writer requires a framework task_id or explicit _shard_output_id"
        raise ValueError(msg)

    def _relative_audio_path(self, task: AudioTask, shard_key: str, output_id: str) -> PurePosixPath:
        preset = task.data.get("output_audio_filepath")
        if preset:
            return _audio_relative_path(str(preset), shard_key)

        source = str(
            task.data.get("original_audio_filepath")
            or task.data.get("original_file")
            or task.data.get("audio_filepath")
            or "audio"
        )
        stem = _source_output_stem(source)
        offset = task.data.get("start_ms")
        if offset is None and task.data.get("offset") is not None:
            offset = round(float(task.data["offset"]) * 1000)
        suffix = f"_{int(offset)}ms" if offset else ""
        filename = f"{stem.name}{suffix}_{output_id[:16]}.opus"
        return PurePosixPath(shard_key, "audio", *stem.parts[:-1], filename)

    def _manifest_entry(  # noqa: C901, PLR0912
        self,
        task: AudioTask,
        audio_filepath: str,
        waveform: np.ndarray | None,
        sample_rate: int,
    ) -> dict[str, Any]:
        original_file = str(
            task.data.get("original_audio_filepath")
            or task.data.get("original_file")
            or task.data.get("audio_filepath")
            or ""
        )
        if task.data.get("vad_empty") or task.data.get("read_error"):
            entry: dict[str, Any] = {
                "audio_filepath": "",
                "duration": 0.0 if task.data.get("vad_empty") else float(task.data.get("duration") or 0.0),
                "sample_rate": sample_rate,
                "sampling_rate": sample_rate,
            }
            if task.data.get("vad_empty"):
                entry["vad_empty"] = True
            if task.data.get("read_error"):
                entry["read_error"] = True
            if task.data.get("audio_too_long"):
                entry["audio_too_long"] = True
        else:
            duration = task.data.get("duration_sec", task.data.get("duration"))
            if duration is None:
                duration = len(waveform) / sample_rate if waveform is not None else 0.0
            entry = {
                "audio_filepath": audio_filepath,
                "duration": round(float(duration), 4),
                "sample_rate": sample_rate,
                "sampling_rate": sample_rate,
            }

        if original_file:
            entry["original_audio_filepath"] = original_file
        base_offset = float(task.data.get("offset") or 0.0)
        inherited_offset = task.data.get("original_offset")
        source_base_offset = float(inherited_offset) if inherited_offset is not None else base_offset
        start_ms = task.data.get("start_ms")
        relative_start = float(start_ms) / 1000.0 if start_ms is not None else 0.0
        playback_offset = base_offset + relative_start if start_ms is not None else task.data.get("offset")
        source_offset = source_base_offset + relative_start
        if playback_offset is not None:
            entry["offset"] = 0.0 if self.save_audio and audio_filepath else float(playback_offset)
        if source_offset is not None and (
            inherited_offset is not None or (self.save_audio and audio_filepath and source_offset != 0.0)
        ):
            entry["original_offset"] = float(source_offset)
        if "end_ms" in task.data:
            entry["original_end"] = source_base_offset + float(task.data["end_ms"]) / 1000.0
        if task.data.get("original_sampling_rate") is not None:
            entry["original_sampling_rate"] = int(task.data["original_sampling_rate"])
        if task.data.get("original_channels") is not None:
            entry["original_channels"] = int(task.data["original_channels"])

        internal = {
            self.waveform_key,
            self.sample_rate_key,
            "waveform",
            "sample_rate",
            "sampling_rate",
            "audio_filepath",
            "original_file",
            "output_audio_filepath",
            "duration_sec",
            "start_ms",
            "end_ms",
            "offset",
            "original_sampling_rate",
            "original_channels",
            "num_channels",
            "num_samples",
            "diar_segments",
            "vad_empty",
            "read_error",
            "audio_too_long",
        }
        for key, value in task.data.items():
            if key not in internal and key not in entry:
                entry[key] = _json_safe(value)
        return entry

    def process(self, task: AudioTask) -> None:
        shard_key = validate_shard_key(str(task._metadata.get("_shard_key") or ""))
        shard_total = int(task._metadata.get("_shard_total") or 0)
        if shard_total <= 0:
            msg = f"Task for shard {shard_key!r} has invalid _shard_total={shard_total}"
            raise ValueError(msg)
        register_nemo_speech_shard(Path(self.output_dir), shard_key, shard_total)
        input_id = self._input_id(task)
        output_slot = self._output_slot(task)
        output_id = hashlib.sha256(f"{_RECEIPT_VERSION}|{shard_key}|{input_id}|{output_slot}".encode()).hexdigest()

        placeholder = bool(task.data.get("vad_empty") or task.data.get("read_error"))
        waveform: np.ndarray | None = None
        sample_rate = int(
            task.data.get(self.sample_rate_key) or task.data.get("sampling_rate") or self.target_sample_rate
        )
        relative_audio_path: PurePosixPath | None = None
        audio_claim: dict[str, Any] | None = None
        manifest_audio_path = ""
        if not placeholder:
            if task.data.get(self.waveform_key) is not None:
                waveform = _mono_float32(task.data[self.waveform_key])
            if self.save_audio and waveform is None:
                msg = f"Task for {input_id!r} has no {self.waveform_key!r} waveform"
                raise ValueError(msg)
            if self.save_audio:
                if sample_rate != self.target_sample_rate:
                    msg = (
                        f"Task for {input_id!r} has sample rate {sample_rate}, expected {self.target_sample_rate}; "
                        "resample upstream before writing Opus."
                    )
                    raise ValueError(msg)
                relative_audio_path = self._relative_audio_path(task, shard_key, output_id)
                output_path = contained_output_path(Path(self.output_dir), relative_audio_path)
                audio_claim = {
                    "version": _OWNER_VERSION,
                    "shard_key": shard_key,
                    "input_id": input_id,
                    "output_slot": output_slot,
                    "audio_filepath": relative_audio_path.as_posix(),
                    "sample_rate": sample_rate,
                    "num_samples": len(waveform),
                    "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                }
                _claim_audio_path(Path(self.output_dir), audio_claim)
                if not (
                    _opus_matches_claim(Path(self.output_dir), output_path, audio_claim)
                    and _valid_opus(output_path, sample_rate, len(waveform))
                ):
                    if output_path.exists():
                        logger.warning(f"Replacing unbound or invalid Opus output: {output_path}")
                    encoded = self._encode_opus(waveform, sample_rate)
                    _claim_audio_path(
                        Path(self.output_dir),
                        audio_claim,
                        hashlib.sha256(encoded).hexdigest(),
                    )
                    _write_bytes_atomically(output_path, encoded)
                if not (
                    _opus_matches_claim(Path(self.output_dir), output_path, audio_claim)
                    and _valid_opus(output_path, sample_rate, len(waveform))
                ):
                    msg = f"Encoded Opus output failed validation: {relative_audio_path.as_posix()}"
                    raise RuntimeError(msg)
                manifest_audio_path = _manifest_audio_path(relative_audio_path, shard_key).as_posix()
                task.data["output_audio_filepath"] = relative_audio_path.as_posix()
            else:
                manifest_audio_path = str(task.data.get("original_file") or task.data.get("audio_filepath") or "")

        entry = self._manifest_entry(task, manifest_audio_path, waveform, sample_rate)
        status = "placeholder" if placeholder else "success"
        receipt = {
            "version": _RECEIPT_VERSION,
            "status": status,
            "shard_key": shard_key,
            "expected_inputs": shard_total,
            "input_id": input_id,
            "output_slot": output_slot,
            "output_id": output_id,
            "save_audio": self.save_audio,
            "audio_claim": audio_claim,
            "entry": entry,
        }
        receipt_path = contained_output_path(
            Path(self.output_dir),
            f".nemo_curator/nemo_speech_rows/{shard_key}/{output_id}.{status}.json",
        )
        _write_receipt_candidate(receipt_path, receipt)


@dataclass(frozen=True)
class _ShardFinalizationPlan:
    shard_key: str
    manifest_path: Path
    marker_path: Path
    temp_path: Path
    marker: dict[str, Any]


def _receipt_filename_identity(path: Path) -> tuple[str, str]:
    for status in _RECEIPT_STATUSES:
        suffix = f".{status}.json"
        if path.name.endswith(suffix):
            output_id = path.name[: -len(suffix)]
            if output_id:
                return output_id, status
    msg = f"Unsupported NeMo speech row receipt filename: {path}"
    raise ValueError(msg)


def _receipt_shards(output_dir: Path) -> list[tuple[str, Path]]:
    """Discover receipt directories without loading every receipt into memory."""

    root = _row_root(output_dir)
    if not root.is_dir():
        return []
    discovered: list[tuple[str, Path]] = []
    for directory, child_names, file_names in os.walk(root, followlinks=False):
        child_names.sort()
        directory_path = Path(directory)
        for child_name in child_names:
            child = directory_path / child_name
            if child.is_symlink():
                msg = f"NeMo speech receipt tree contains a symlink: {child}"
                raise ValueError(msg)
        if not any(name.endswith(".json") for name in file_names):
            continue
        relative = directory_path.relative_to(root)
        if not relative.parts:
            msg = f"NeMo speech receipt found outside a shard directory: {directory_path}"
            raise ValueError(msg)
        shard_key = validate_shard_key(PurePosixPath(*relative.parts).as_posix())
        discovered.append((shard_key, _receipt_dir(output_dir, shard_key)))
    return sorted(discovered)


def _registered_shards(output_dir: Path) -> list[tuple[str, int, Path]]:
    root = shard_registration_root(output_dir)
    if not root.is_dir():
        return []
    registered: list[tuple[str, int, Path]] = []
    for directory, child_names, file_names in os.walk(root, followlinks=False):
        child_names.sort()
        directory_path = Path(directory)
        for child_name in child_names:
            child = directory_path / child_name
            if child.is_symlink():
                msg = f"NeMo speech shard-registration tree contains a symlink: {child}"
                raise ValueError(msg)
        for file_name in sorted(name for name in file_names if name.endswith(".json")):
            relative = (directory_path / file_name).relative_to(root)
            key_parts = [*relative.parts[:-1], relative.name.removesuffix(".json")]
            shard_key = validate_shard_key(PurePosixPath(*key_parts).as_posix())
            path = shard_registration_path(output_dir, shard_key)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                expected_inputs = int(payload.get("expected_inputs") or 0)
            except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
                msg = f"Invalid NeMo speech shard registration {path}: {error}"
                raise ValueError(msg) from error
            if (
                payload.get("version") != SHARD_REGISTRATION_VERSION
                or payload.get("shard_key") != shard_key
                or expected_inputs <= 0
            ):
                msg = f"Invalid NeMo speech shard registration: {path}"
                raise ValueError(msg)
            registered.append((shard_key, expected_inputs, _receipt_dir(output_dir, shard_key)))
    return sorted(registered)


def _load_receipt(path: Path, shard_key: str) -> tuple[str, str, dict[str, Any]]:
    output_id, status = _receipt_filename_identity(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Invalid NeMo speech row receipt {path}: {error}"
        raise ValueError(msg) from error
    if payload.get("version") != _RECEIPT_VERSION:
        msg = f"Unsupported NeMo speech row receipt version in {path}"
        raise ValueError(msg)
    if payload.get("status") != status:
        msg = f"NeMo speech row receipt filename does not match status: {path}"
        raise ValueError(msg)
    if validate_shard_key(str(payload.get("shard_key") or "")) != shard_key:
        msg = f"NeMo speech row receipt directory does not match shard_key: {path}"
        raise ValueError(msg)
    if payload.get("output_id") != output_id:
        msg = f"NeMo speech row receipt filename does not match output_id: {path}"
        raise ValueError(msg)
    return output_id, status, payload


def _selected_receipts(output_dir: Path, shard_key: str, directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for discovered_path in sorted(directory.iterdir()):
        if not discovered_path.name.endswith(".json"):
            continue
        relative = PurePosixPath(".nemo_curator/nemo_speech_rows") / shard_key / discovered_path.name
        path = contained_output_path(output_dir, relative)
        output_id, status, payload = _load_receipt(path, shard_key)
        by_status = candidates.setdefault(output_id, {})
        by_status[status] = payload

    # A durable success outranks a transient placeholder in either retry order.
    return [by_status.get("success") or by_status["placeholder"] for by_status in candidates.values()]


def _receipt_sort_key(receipt: dict[str, Any]) -> tuple[Any, ...]:
    input_id = str(receipt.get("input_id") or "")
    try:
        input_order: tuple[int, Any] = (0, int(input_id))
    except ValueError:
        input_order = (1, input_id)
    entry = receipt.get("entry") if isinstance(receipt.get("entry"), dict) else {}
    sort_offset = entry.get("original_offset")
    try:
        offset = float(sort_offset if sort_offset is not None else entry.get("offset") or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    return (*input_order, offset, str(entry.get("audio_filepath") or ""), str(receipt.get("output_id") or ""))


def _claimed_audio_path_for_manifest(
    *,
    shard_key: str,
    manifest_relative: str,
    receipt: dict[str, Any],
) -> tuple[PurePosixPath, dict[str, Any]]:
    safe_manifest_relative = _safe_relative_path(manifest_relative, field_name="audio_filepath")
    claim = receipt.get("audio_claim")
    if not isinstance(claim, dict):
        msg = f"Shard {shard_key!r} has an Opus receipt without an ownership claim"
        raise TypeError(msg)
    safe_relative = _audio_relative_path(str(claim.get("audio_filepath") or ""), shard_key)
    expected_manifest_relative = _manifest_audio_path(safe_relative, shard_key)
    if safe_manifest_relative != expected_manifest_relative:
        msg = (
            f"Shard {shard_key!r} has manifest audio path {manifest_relative!r}, expected "
            f"{expected_manifest_relative.as_posix()!r} from its ownership claim"
        )
        raise ValueError(msg)
    return safe_relative, claim


def _validated_manifest_entry(  # noqa: C901
    *,
    root: Path,
    shard_key: str,
    receipt: dict[str, Any],
    claimed_audio_paths: dict[str, str],
) -> dict[str, Any]:
    output_id = str(receipt.get("output_id") or "")
    output_slot = str(receipt.get("output_slot") or "")
    if not output_id or not output_slot:
        msg = f"Shard {shard_key!r} has a receipt without output identity"
        raise ValueError(msg)
    input_id = str(receipt.get("input_id") or "")
    expected_output_id = hashlib.sha256(
        f"{_RECEIPT_VERSION}|{shard_key}|{input_id}|{output_slot}".encode()
    ).hexdigest()
    if output_id != expected_output_id:
        msg = f"Shard {shard_key!r} has a receipt with a mismatched output_id"
        raise ValueError(msg)
    entry = receipt.get("entry")
    if not isinstance(entry, dict):
        msg = f"Shard {shard_key!r} has a receipt without a manifest object"
        raise TypeError(msg)
    manifest_relative = str(entry.get("audio_filepath") or "")
    if not manifest_relative or not receipt.get("save_audio"):
        return entry

    safe_relative, claim = _claimed_audio_path_for_manifest(
        shard_key=shard_key,
        manifest_relative=manifest_relative,
        receipt=receipt,
    )
    previous_owner = claimed_audio_paths.setdefault(safe_relative.as_posix(), output_id)
    if previous_owner != output_id:
        msg = f"Shard {shard_key!r} has multiple outputs claiming {safe_relative.as_posix()!r}"
        raise ValueError(msg)
    expected_sample_rate = int(entry.get("sample_rate") or 0)
    output_path = contained_output_path(root, safe_relative)
    expected_claim_fields = {
        "version": _OWNER_VERSION,
        "shard_key": shard_key,
        "input_id": input_id,
        "output_slot": output_slot,
        "audio_filepath": safe_relative.as_posix(),
        "sample_rate": expected_sample_rate,
    }
    if any(claim.get(key) != value for key, value in expected_claim_fields.items()):
        msg = f"Shard {shard_key!r} has inconsistent ownership metadata for {safe_relative.as_posix()!r}"
        raise ValueError(msg)
    claim_path = _owner_path(root, safe_relative)
    try:
        persisted_claim = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Invalid NeMo speech audio ownership record {claim_path}: {error}"
        raise ValueError(msg) from error
    if claim != persisted_claim or claim.get("audio_filepath") != safe_relative.as_posix():
        msg = f"Shard {shard_key!r} has a mismatched ownership claim for {safe_relative.as_posix()!r}"
        raise ValueError(msg)
    expected_samples = int(claim.get("num_samples") or 0)
    if expected_samples <= 0:
        msg = f"Shard {shard_key!r} has invalid expected sample count for {safe_relative.as_posix()!r}"
        raise ValueError(msg)
    if not (
        _opus_matches_claim(root, output_path, claim)
        and _valid_opus(output_path, expected_sample_rate, expected_samples)
    ):
        msg = (
            f"Shard {shard_key!r} references missing or invalid Opus file, "
            f"or one without a matching content binding: {safe_relative}"
        )
        raise ValueError(msg)
    return entry


def _stage_shard_manifest(
    root: Path,
    shard_key: str,
    registered_expected: int,
    receipt_directory: Path,
) -> _ShardFinalizationPlan:
    receipts = _selected_receipts(root, shard_key, receipt_directory)
    expected_values = {int(receipt.get("expected_inputs") or 0) for receipt in receipts}
    if expected_values and expected_values != {registered_expected}:
        msg = f"Shard {shard_key!r} has conflicting or invalid expected input counts: {sorted(expected_values)}"
        raise ValueError(msg)
    seen_inputs = {str(receipt.get("input_id")) for receipt in receipts}
    if len(seen_inputs) != registered_expected:
        msg = f"Shard {shard_key!r} is incomplete: saw {len(seen_inputs)} of {registered_expected} inputs"
        raise ValueError(msg)
    save_audio_values = {receipt.get("save_audio") for receipt in receipts}
    if len(save_audio_values) != 1 or not save_audio_values <= {True, False}:
        msg = f"Shard {shard_key!r} mixes invalid audio persistence modes"
        raise ValueError(msg)

    manifest_path = contained_output_path(root, f"{shard_key}.jsonl")
    marker_path = contained_output_path(root, f"{shard_key}.jsonl.done")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    digest = hashlib.sha256()
    row_count = 0
    claimed_audio_paths: dict[str, str] = {}
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            for receipt in sorted(receipts, key=_receipt_sort_key):
                entry = _validated_manifest_entry(
                    root=root,
                    shard_key=shard_key,
                    receipt=receipt,
                    claimed_audio_paths=claimed_audio_paths,
                )
                line = (json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                stream.write(line)
                digest.update(line)
                row_count += 1
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    marker = {
        "version": _MARKER_VERSION,
        "shard_key": shard_key,
        "expected_inputs": registered_expected,
        "completed_inputs": len(seen_inputs),
        "manifest_rows": row_count,
        "manifest_sha256": digest.hexdigest(),
    }
    return _ShardFinalizationPlan(
        shard_key=shard_key,
        manifest_path=manifest_path,
        marker_path=marker_path,
        temp_path=temp_path,
        marker=marker,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_nemo_speech_output(output_dir: str) -> list[str]:
    """Validate row receipts and atomically publish manifests and markers.

    The function performs a validation pass for every pending shard before it
    writes any marker.  It raises ``ValueError`` for missing inputs, conflicting
    totals, unsafe paths, malformed rows, or missing Opus files. Call it once
    on the driver after all writer workers are quiescent; concurrent writers
    or finalizers targeting the same output directory are unsupported.

    Returns:
        Manifest paths finalized during this call.
    """

    root = validate_local_output_dir(output_dir)
    plans: list[_ShardFinalizationPlan] = []
    try:
        registered = _registered_shards(root)
        registered_keys = {shard_key for shard_key, _, _ in registered}
        receipt_keys = {shard_key for shard_key, _ in _receipt_shards(root)}
        if orphaned := sorted(receipt_keys - registered_keys):
            msg = f"NeMo speech receipts have no shard registration: {orphaned}"
            raise ValueError(msg)
        # Stage one fsynced manifest per shard, but publish nothing until every
        # shard has passed completeness, ownership, and Opus validation.
        for shard_key, expected_inputs, receipt_directory in registered:
            plans.append(
                _stage_shard_manifest(
                    root,
                    shard_key,
                    expected_inputs,
                    receipt_directory,
                )
            )

        finalized: list[str] = []
        for plan in plans:
            try:
                existing_marker = json.loads(plan.marker_path.read_text(encoding="utf-8"))
                already_finalized = (
                    existing_marker == plan.marker
                    and plan.manifest_path.is_file()
                    and _sha256_file(plan.manifest_path) == plan.marker["manifest_sha256"]
                )
            except (OSError, json.JSONDecodeError):
                already_finalized = False
            if already_finalized:
                logger.info(f"NeMo speech shard {plan.shard_key} is already finalized")
                continue
            # A marker must never remain visible while its bound manifest is
            # replaced. If publication stops after this unlink, discovery sees
            # an incomplete shard and safely replays/finalizes it.
            if plan.marker_path.exists():
                plan.marker_path.unlink()
                with contextlib.suppress(OSError):
                    fsync_directory(plan.marker_path.parent)
            os.replace(plan.temp_path, plan.manifest_path)
            with contextlib.suppress(OSError):
                fsync_directory(plan.manifest_path.parent)
            write_json_atomically(plan.marker_path, plan.marker, separators=(",", ":"))
            finalized.append(str(plan.manifest_path))
            logger.info(f"Finalized NeMo speech shard {plan.shard_key}: {plan.marker['manifest_rows']} rows")
        return finalized
    finally:
        for plan in plans:
            plan.temp_path.unlink(missing_ok=True)


__all__ = ["NeMoSpeechWriterStage", "finalize_nemo_speech_output"]
