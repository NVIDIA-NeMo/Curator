"""Helpers to bridge in-memory waveforms and on-disk audio paths.

Background: in the AIS-streamed pipeline, ``NemoTarShardReaderStage`` decodes
audio in memory from tars and emits ``AudioTask`` objects carrying
``task.data["waveform"]`` (numpy ``float32``) and ``task.data["sample_rate"]``
(int).  Most downstream stages were written for the file-based world and call
``librosa.load(audio_filepath)`` or pass an audio path to NeMo APIs.

Older approach: insert ``AsrBridgeStage`` to materialise every waveform to a
temp WAV under a single shared directory.  This breaks on multi-node Slurm
allocations because ``/tmp`` is per-node-local — an AsrBridge actor on node A
writes a file the SED actor on node B cannot read, and the only shared FS
(Lustre) is too slow for many tiny files.

This module provides two helpers that move the bridge from a separate stage
into the consumer stage itself, so the temp file is always written on the
node that's about to read it (or skipped entirely if the consumer can take
the waveform directly):

* :func:`ensure_waveform` — return the waveform as a numpy array.  Use when
  the stage owns its audio loading (e.g. SED, UTMOSv2).

* :func:`ensure_local_audio_path` — return a local file path for the audio,
  materialising the waveform to a node-local temp WAV if necessary.  Use
  when the stage hands the path off to a third-party API that only accepts
  files (e.g. NeMo's ``diarize()``, ``transcribe()``).
"""
from __future__ import annotations

import os
import tempfile
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.tasks import AudioTask

# Per-process cache so retries don't re-write the same temp WAV.  Cleared
# implicitly when the worker exits.  Keyed by task_id.
_LOCAL_PATH_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def _coerce_mono_float32(wav: "np.ndarray") -> "np.ndarray":
    import numpy as np

    arr = np.asarray(wav)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        # (samples, channels) or (channels, samples) — pick the smaller axis
        # as the channel axis and average down to mono.
        ch_axis = 0 if arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ch_axis)
    return arr


def ensure_waveform(
    task: "AudioTask",
    target_sr: int = 16000,
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
    filepath_key: str = "audio_filepath",
) -> "np.ndarray":
    """Return a mono float32 waveform at ``target_sr`` for the task.

    Prefers ``task.data[waveform_key]`` (with ``task.data[sample_rate_key]``)
    over loading from a file.  Resamples if needed.

    Falls back to ``soundfile.read(task.data[filepath_key])`` if the
    waveform is not in memory.
    """
    if waveform_key in task.data:
        wav = _coerce_mono_float32(task.data[waveform_key])
        sr = int(task.data.get(sample_rate_key, target_sr))
        if sr != target_sr:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav

    import soundfile as sf

    path = task.data.get(filepath_key, "")
    if not path:
        msg = f"Task {task.task_id}: no {waveform_key} and no {filepath_key}"
        raise ValueError(msg)
    audio, sr = sf.read(path, dtype="float32")
    audio = _coerce_mono_float32(audio)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def ensure_local_audio_path(
    task: "AudioTask",
    target_sr: int | None = None,
    temp_dir: str | None = None,
    subtype: str = "PCM_16",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
    filepath_key: str = "audio_filepath",
) -> str:
    """Return a node-local path for the task's audio.

    If ``task.data[filepath_key]`` already points to a readable file on the
    local node, that path is returned unchanged.  Otherwise the in-memory
    ``task.data[waveform_key]`` is written to a temp WAV under
    ``temp_dir`` (default: a per-job subdir of the system temp dir on the
    local node — never a shared FS), and the new path is also written back
    into ``task.data[filepath_key]`` so subsequent reads see the file.

    Designed for stages that pass ``audio_filepath`` to NeMo / external APIs
    that accept only file paths.  Each actor that calls this helper writes
    the temp WAV on its own node, so cross-node access is never required.

    Args:
        task: The AudioTask carrying either a waveform or a filepath.
        target_sr: If set, resample the in-memory waveform to this rate
            before writing.  Has no effect when reusing an existing file.
        temp_dir: Override the parent temp directory.  Defaults to the
            local system temp directory + per-job subdir.
        subtype: ``soundfile`` write subtype.  ``"PCM_16"`` matches what
            most NeMo file readers expect.
    """
    existing = task.data.get(filepath_key, "")
    if existing and os.path.exists(existing):
        return existing

    if waveform_key not in task.data:
        msg = (
            f"Task {task.task_id}: no readable {filepath_key} ({existing!r}) "
            f"and no in-memory {waveform_key}"
        )
        raise ValueError(msg)

    with _CACHE_LOCK:
        cached = _LOCAL_PATH_CACHE.get(task.task_id)
    if cached and os.path.exists(cached):
        task.data[filepath_key] = cached
        return cached

    import soundfile as sf

    wav = _coerce_mono_float32(task.data[waveform_key])
    sr = int(task.data.get(sample_rate_key, target_sr or 16000))
    if target_sr and sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    base = temp_dir or tempfile.gettempdir()
    slurm_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    sub = os.path.join(base, f"audio_io_{slurm_id}")
    os.makedirs(sub, exist_ok=True)
    safe_id = task.task_id.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(sub, f"{safe_id}.wav")
    sf.write(out_path, wav, sr, subtype=subtype)

    with _CACHE_LOCK:
        _LOCAL_PATH_CACHE[task.task_id] = out_path
    task.data[filepath_key] = out_path
    return out_path
