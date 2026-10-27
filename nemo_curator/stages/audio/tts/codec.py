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

"""Codec-code extraction for GranaryHiFi SHAR shards.

Target and context audio of each
shard are resampled to the codec rate, padded to a multiple of ``samples_per_frame``, encoded with
``audio_encoder`` + ``quantize`` (not ``encode``, which pads again) and written as ``uint16 [n_codebooks, T]`` arrays::

    <rate_dir>/<codec_name>/target_codes/codes.NNNNNN.tar    <cut_id>.npy + <cut_id>.json (TemporalArray)
    <rate_dir>/<codec_name>/context_codes/codes.NNNNNN.tar   context codes stored under the target cut id
    <rate_dir>/<codec_name>/.complete/codes.NNNNNN.json      completion marker (sha256 of the two audio tars)

Codes are reused only when the marker's audio digest equals the digest of the shard's current target/context tars,
so annotation-only shard rebuilds (identical audio bytes) keep their codes while changed audio is re-encoded.
"""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.shar_materialize import (
    MARKER_DIR,
    file_digest,
    load_registry,
    read_cut_ids,
    read_json,
    read_tar_ids,
    write_json_atomic,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask

PUBLIC_CODEC_MODEL = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
PUBLIC_CODEC_NAME = "nemoNanoCodec22kHz1.89kbps21.5fps"
# The 21.5 fps NanoCodec is the default everywhere; a local ``.nemo`` can still be passed as ``model_path``
# together with its own ``codec_name``.
DEFAULT_CODEC_NAME = PUBLIC_CODEC_NAME
# Training-only HF discriminators are replaced by an allow-listed no-op module; weights load with strict=False.
_DISCRIMINATOR_STUB = {"_target_": "torch.nn.Identity"}


_IGNORED_CHECKPOINT_PREFIXES = ("discriminator.",)


def checkpoint_keys(model_path: str) -> set[str]:
    """State-dict keys stored in a ``.nemo`` archive (``model_weights.ckpt``)."""
    import torch

    with tarfile.open(model_path, "r:*") as archive:
        member = next((m for m in archive if m.name.endswith("model_weights.ckpt")), None)
        if member is None:
            msg = f"No model_weights.ckpt in {model_path}"
            raise ValueError(msg)
        state = torch.load(archive.extractfile(member), map_location="cpu", weights_only=True)
    return set(state.get("state_dict", state))


def check_codec_keys(model: Any, model_path: str) -> None:  # noqa: ANN401
    """Fail unless the ``strict=False`` load skipped only discriminator weights and left nothing uninitialized."""
    stored = checkpoint_keys(model_path)
    expected = set(model.state_dict())
    unexpected = sorted(k for k in stored - expected if not k.startswith(_IGNORED_CHECKPOINT_PREFIXES))
    missing = sorted(expected - stored)
    if unexpected or missing:
        msg = f"Codec checkpoint {model_path} does not match the model: unexpected={unexpected[:5]} missing={missing[:5]}"
        raise RuntimeError(msg)


def restore_codec_model(model_path: str, map_location: str = "cpu", *, check_keys: bool = True) -> Any:  # noqa: ANN401
    """Restore a NeMo ``AudioCodecModel`` for inference without instantiating its training discriminator.

    The discriminator is replaced by ``torch.nn.Identity`` (no parameters), weights load with ``strict=False``, and
    ``check_keys`` verifies that only ``discriminator.*`` checkpoint keys were ignored.
    """
    from nemo.collections.tts.models import AudioCodecModel
    from omegaconf import OmegaConf, open_dict

    cfg = AudioCodecModel.restore_from(restore_path=model_path, map_location="cpu", return_config=True)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        cfg.discriminator = dict(_DISCRIMINATOR_STUB)
    model = AudioCodecModel.restore_from(
        restore_path=model_path, override_config_path=cfg, map_location=map_location, strict=False
    )
    if check_keys:
        check_codec_keys(model, model_path)
    return model


def load_pretrained_codec_model(model_name: str) -> Any:  # noqa: ANN401
    """Download and load a public NeMo audio codec from Hugging Face."""
    from nemo.collections.tts.models import AudioCodecModel

    return AudioCodecModel.from_pretrained(model_name, map_location="cpu")


def is_local_codec_model(model: str) -> bool:
    """Return whether ``model`` denotes a local checkpoint rather than a Hugging Face model ID."""
    path = Path(model).expanduser()
    return path.is_file() or path.is_absolute() or path.suffix == ".nemo" or model.startswith((".", "~"))


def resample_poly(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resampling exactly as lhotse's ``Resample`` does when torchaudio is absent.

    lhotse silently switches to a sinc kernel when torchaudio is importable, which changes ~20% of codec tokens, so
    resampling is done explicitly with ``scipy.signal.resample_poly`` to be environment independent.
    """
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly as _resample_poly

    gcd = int(np.gcd(source_rate, target_rate))
    return _resample_poly(audio, up=target_rate // gcd, down=source_rate // gcd, axis=-1).astype(np.float32)


def load_resampled(recording: Any, target_rate: int) -> np.ndarray:  # noqa: ANN401
    """Load a lhotse ``Recording`` at its native rate and resample it to ``target_rate`` (mono, 1-D)."""
    audio = recording.load_audio()
    return resample_poly(audio, int(recording.sampling_rate), target_rate).reshape(-1)


def padded_length(num_samples: int, samples_per_frame: int) -> int:
    """Round ``num_samples`` up to a whole number of codec frames."""
    if samples_per_frame <= 0:
        return num_samples
    return ((num_samples + samples_per_frame - 1) // samples_per_frame) * samples_per_frame


def collate(waves: list[np.ndarray], lengths: list[int]) -> np.ndarray:
    """Zero-pad waveforms into one float32 batch array."""
    batch = np.zeros((len(waves), max(lengths)), dtype=np.float32)
    for row, wave in enumerate(waves):
        batch[row, : wave.shape[-1]] = wave
    return batch


def codes_paths(rate_dir: Path, shard_id: int, codec_name: str) -> tuple[Path, Path]:
    root = rate_dir / codec_name
    return (
        root / "target_codes" / f"codes.{shard_id:06d}.tar",
        root / "context_codes" / f"codes.{shard_id:06d}.tar",
    )


def codes_marker_path(rate_dir: Path, shard_id: int, codec_name: str) -> Path:
    return rate_dir / codec_name / MARKER_DIR / f"codes.{shard_id:06d}.json"


def codes_complete(path: Path, expected_ids: list[str]) -> bool:
    """Structural check: payload and metadata ids of a codes tar equal the shard's cut ids, in order."""
    if not path.is_file():
        return False
    try:
        payload, metadata = read_tar_ids(path, {".npy"})
    except (OSError, tarfile.TarError):
        return False
    return payload == expected_ids and metadata == expected_ids


def shard_audio_digest(shard: dict[str, Any]) -> str:
    """sha256 of a shard's target and context audio tars (decides whether codes are stale)."""
    return file_digest(Path(shard["target_audio"]), Path(shard["context_audio"]))


def shard_codes_complete(shard: dict[str, Any], codec_name: str, audio_digest: str | None = None) -> bool:
    """Codes are complete when the marker matches the current audio digest and both tars align with the cuts."""
    rate_dir = Path(shard["dir"])
    shard_id = int(shard["shard_id"])
    marker = read_json(codes_marker_path(rate_dir, shard_id, codec_name))
    audio_digest = audio_digest or shard_audio_digest(shard)
    if marker is None or marker.get("audio_digest") != audio_digest or marker.get("codec") != codec_name:
        return False
    expected = read_cut_ids(Path(shard["cuts"]))
    return all(codes_complete(path, expected) for path in codes_paths(rate_dir, shard_id, codec_name))


class CodecEncoder:
    """Thin batching wrapper around a NeMo audio codec (or any object with the same interface)."""

    def __init__(self, model: Any, device: str = "cuda") -> None:  # noqa: ANN401
        self.model = model
        self.device = device

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    @property
    def samples_per_frame(self) -> int:
        return int(self.model.samples_per_frame)

    def encode(self, waves: list[np.ndarray]) -> list[np.ndarray]:
        """Encode mono waveforms at ``sample_rate``; returns ``uint16 [n_codebooks, T_i]`` per input."""
        import torch

        lengths = [padded_length(int(w.shape[-1]), self.samples_per_frame) for w in waves]
        audio = torch.from_numpy(collate(waves, lengths)).to(self.device)
        audio_len = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        with torch.inference_mode():
            encoded, encoded_len = self.model.audio_encoder(audio=audio, audio_len=audio_len)
            tokens = self.model.quantize(encoded=encoded, encoded_len=encoded_len)
        tokens = tokens.to(dtype=torch.uint16, device="cpu").numpy()
        encoded_len = encoded_len.cpu().numpy()
        return [tokens[i, :, : int(encoded_len[i])] for i in range(len(waves))]


def _temporal_manifest(codes: np.ndarray, frame_rate: float) -> Any:  # noqa: ANN401
    from lhotse.array import Array, TemporalArray

    return TemporalArray(
        array=Array(storage_type="shar", storage_path="", storage_key="", shape=list(codes.shape)),
        temporal_dim=-1,
        frame_shift=1 / frame_rate,
        start=0,
    )


def encode_shard(  # noqa: PLR0913
    encoder: CodecEncoder,
    cuts_path: Path,
    target_audio_path: Path,
    context_audio_path: Path,
    out_target: Path,
    out_context: Path,
    *,
    frame_rate: float = 25.0,
    batch_size: int = 128,
) -> int:
    """Encode one shard's target and context audio into two codes tars (written atomically)."""
    from lhotse import CutSet
    from lhotse.shar.writers.array import ArrayTarWriter

    cuts = CutSet.from_shar(
        fields={
            "cuts": [str(cuts_path)],
            "recording": [str(target_audio_path)],
            "context_recording": [str(context_audio_path)],
        }
    )
    out_target.parent.mkdir(parents=True, exist_ok=True)
    out_context.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = out_target.with_name(f"{out_target.name}.tmp.{os.getpid()}")
    tmp_context = out_context.with_name(f"{out_context.name}.tmp.{os.getpid()}")
    written = 0
    try:
        with (
            ArrayTarWriter(str(tmp_target), shard_size=None, compression="numpy") as target_writer,
            ArrayTarWriter(str(tmp_context), shard_size=None, compression="numpy") as context_writer,
        ):
            batch: list[Any] = []

            def flush() -> None:
                nonlocal written
                if not batch:
                    return
                rate = encoder.sample_rate
                targets = [load_resampled(c.recording, rate) for c in batch]
                contexts = [load_resampled(c.context_recording, rate) for c in batch]
                for cut, t_codes, c_codes in zip(
                    batch, encoder.encode(targets), encoder.encode(contexts), strict=True
                ):
                    target_writer.write(key=cut.id, value=t_codes, manifest=_temporal_manifest(t_codes, frame_rate))
                    context_writer.write(key=cut.id, value=c_codes, manifest=_temporal_manifest(c_codes, frame_rate))
                    written += 1
                batch.clear()

            for cut in cuts:
                if not cut.has_custom("context_recording"):
                    msg = f"Cut {cut.id} has no context_recording in {cuts_path}"
                    raise ValueError(msg)
                batch.append(cut)
                if len(batch) >= batch_size:
                    flush()
            flush()
        os.replace(tmp_target, out_target)
        os.replace(tmp_context, out_context)
    finally:
        for tmp in (tmp_target, tmp_context):
            if tmp.exists():
                tmp.unlink()
    return written


def extract_codes_for_registry(
    registry_path: str | Path,
    encoder: CodecEncoder,
    *,
    codec_name: str = DEFAULT_CODEC_NAME,
    frame_rate: float = 25.0,
    batch_size: int = 128,
) -> dict[str, int]:
    """Extract codes for every shard of a registry, reusing codes whose marker matches the current audio digest."""
    registry = load_registry(registry_path)
    counts = {"encoded_shards": 0, "reused_shards": 0, "cuts": 0}
    for shard in registry["shards"]:
        cuts_path = Path(shard["cuts"])
        rate_dir = Path(shard["dir"])
        shard_id = int(shard["shard_id"])
        out_target, out_context = codes_paths(rate_dir, shard_id, codec_name)
        audio_digest = shard_audio_digest(shard)
        if shard_codes_complete(shard, codec_name, audio_digest):
            counts["reused_shards"] += 1
            continue
        marker = codes_marker_path(rate_dir, shard_id, codec_name)
        marker.unlink(missing_ok=True)
        counts["cuts"] += encode_shard(
            encoder,
            cuts_path,
            Path(shard["target_audio"]),
            Path(shard["context_audio"]),
            out_target,
            out_context,
            frame_rate=frame_rate,
            batch_size=batch_size,
        )
        expected = read_cut_ids(cuts_path)
        if not (codes_complete(out_target, expected) and codes_complete(out_context, expected)):
            msg = f"Codes for shard {shard_id} of {registry_path} are incomplete"
            raise RuntimeError(msg)
        write_json_atomic(marker, {"audio_digest": audio_digest, "codec": codec_name, "num_items": len(expected)})
        counts["encoded_shards"] += 1
    logger.info("Codec extraction {}: {}", registry_path, counts)
    return counts


@dataclass
class CodecCodesStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Extract neural codec codes for SHAR shards listed in ``shar_registry.json`` files.

    ``model_path`` defaults to the public 21.5 fps NeMo NanoCodec on Hugging Face and also accepts any other NeMo
    model ID or a local ``.nemo`` path. A pre-loaded codec model can instead be injected through ``codec_model``
    (tests); exactly one model source is required. Output codes live next to the shard's audio under
    ``<rate>/<codec_name>/{target_codes,context_codes}/``. For local checkpoints, ``check_keys`` verifies that the
    ``strict=False`` load ignored only ``discriminator.*`` weights.
    """

    model_path: str | None = PUBLIC_CODEC_MODEL
    codec_name: str = PUBLIC_CODEC_NAME
    frame_rate: float | None = None
    inference_batch_size: int = 128
    device: str = "cuda"
    codec_model: Any | None = None
    check_keys: bool = True
    name: str = "CodecCodes"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))
    batch_size: int = 1
    _encoder: CodecEncoder | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if self.codec_model is not None and self.model_path == PUBLIC_CODEC_MODEL:
            self.model_path = None  # an injected model wins over the default model id
        sources = sum(source is not None for source in (self.model_path, self.codec_model))
        if sources != 1:
            msg = "exactly one of model_path or codec_model is required"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        import torch

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is not available; codec extraction falls back to CPU (much slower)")
            device = "cpu"
        if self.codec_model is not None:
            model = self.codec_model
        elif self.model_path is not None and not is_local_codec_model(self.model_path):
            model = load_pretrained_codec_model(self.model_path)
        else:
            model = restore_codec_model(str(self.model_path), check_keys=self.check_keys)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        self._encoder = CodecEncoder(model, device)
        actual_frame_rate = self._encoder.sample_rate / self._encoder.samples_per_frame
        if self.frame_rate is None:
            self.frame_rate = actual_frame_rate
        elif not np.isclose(self.frame_rate, actual_frame_rate, rtol=0.0, atol=1e-3):
            msg = (
                f"Configured frame_rate={self.frame_rate} does not match codec frame rate "
                f"{actual_frame_rate:.6g} ({self._encoder.sample_rate}/{self._encoder.samples_per_frame})"
            )
            raise ValueError(msg)
        logger.info(
            "Loaded codec {}: sample_rate={}, samples_per_frame={}, frame_rate={:.6g}",
            self.codec_name,
            self._encoder.sample_rate,
            self._encoder.samples_per_frame,
            self.frame_rate,
        )

    def process(self, task: FileGroupTask) -> FileGroupTask:
        if self._encoder is None:
            self.setup()
        if self.frame_rate is None:
            msg = "Codec frame rate was not initialized"
            raise RuntimeError(msg)
        for registry in task.data:
            extract_codes_for_registry(
                registry,
                self._encoder,
                codec_name=self.codec_name,
                frame_rate=self.frame_rate,
                batch_size=self.inference_batch_size,
            )
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=list(task.data),
            _metadata={**task._metadata, "codec_name": self.codec_name},
            _stage_perf=task._stage_perf,
        )
