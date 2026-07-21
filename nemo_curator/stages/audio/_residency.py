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

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import soundfile as sf

from nemo_curator.stages.audio._agent_ready import AudioForm, IOSpec
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file

if TYPE_CHECKING:
    from collections.abc import Callable

InputResidency = Literal["file", "waveform", "auto"]


def accepts_for_residency(residency: str) -> list[AudioForm]:
    """Audio forms an instance actually consumes, given its ``input_residency``.

    The single source of truth a stage's ``describe()`` derives ``accepts`` from,
    so a ``file``-mode instance can never advertise ``waveform`` (the drift /
    "lying accepts" bug). ``auto`` accepts either; ``file``/``waveform`` accept
    only that form.
    """
    if residency == "waveform":
        return ["waveform"]
    if residency == "file":
        return ["file"]
    return ["file", "waveform"]  # "auto"


def residency_read_specs(
    input_residency: str,
    *,
    audio_filepath_key: str,
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
) -> list[IOSpec]:
    """The residency-filtered audio read options for a stage's ``reads_one_of``.

    ``file`` -> ``[file spec]``; ``waveform`` -> ``[waveform spec]``; ``auto`` ->
    ``[waveform spec, file spec]``. Keeps ``accepts`` **and** ``data_keys`` in
    lockstep with ``input_residency`` so a stage can never advertise (or require)
    a form it won't consume for its current setting — which lets the deterministic
    role check enforce residency compatibility with no extra check.
    """
    forms = accepts_for_residency(input_residency)
    specs: list[IOSpec] = []
    if "waveform" in forms:
        specs.append(IOSpec(data_keys=[waveform_key, sample_rate_key], accepts=["waveform"]))
    if "file" in forms:
        specs.append(IOSpec(data_keys=[audio_filepath_key], accepts=["file"]))
    return specs


def resolve_audio(  # noqa: PLR0913 (complexity accepted: keyword-only residency/key knobs mirror the stage fields)
    item: dict[str, Any],
    *,
    residency: InputResidency = "auto",
    audio_filepath_key: str = "audio_filepath",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
    mono: bool = True,
    loader: Callable[..., tuple[Any, int]] | None = None,
) -> tuple[Any, int] | None:
    """Return ``(waveform_2d, sample_rate)`` from tensor keys or a file path.

    ``auto`` prefers an existing waveform, then falls back to file loading.
    ``waveform`` never falls back to disk. ``file`` always loads from the
    configured path key.

    ``loader`` overrides the file-loading callable (default
    :func:`~nemo_curator.stages.audio.common.load_audio_file`); stages pass
    their own module-level symbol so callers can patch it at the stage module.
    """
    waveform = item.get(waveform_key)
    sample_rate = item.get(sample_rate_key)
    if residency != "file" and waveform is not None and sample_rate is not None:
        return ensure_waveform_2d(waveform), int(sample_rate)

    if residency == "waveform":
        return None

    path = item.get(audio_filepath_key)
    if path:
        expanded = os.path.expanduser(str(path))
        if os.path.exists(expanded):
            return (loader or load_audio_file)(expanded, mono=mono)
    return None


def _as_soundfile_array(waveform: Any) -> Any:  # noqa: ANN401
    waveform = ensure_waveform_2d(waveform)
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    if getattr(waveform, "ndim", 0) == 2:  # noqa: PLR2004 - 2 == a (channels, samples) 2-D array
        channels, samples = waveform.shape
        if channels == 1:
            return waveform[0]
        if channels < samples:
            return waveform.T
    return waveform


def resolve_audio_path(  # noqa: PLR0913 (complexity accepted: keyword-only residency/key knobs mirror the stage fields)
    item: dict[str, Any],
    *,
    residency: InputResidency = "auto",
    audio_filepath_key: str = "audio_filepath",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
    temp_dir: str | None = None,
    register_temp: list[str] | None = None,
) -> str | None:
    """Return an audio path, writing a temp WAV when only a waveform exists.

    When a temp WAV is materialized from an in-memory waveform and
    ``register_temp`` is provided, the temp path is appended to that list so the
    caller can delete it after use (see :func:`cleanup_temp_files`). Without
    ``register_temp`` the caller is responsible for cleanup itself.
    """
    path = item.get(audio_filepath_key)
    local_path: str | None = None
    if residency != "waveform" and path:
        local_path = os.path.expanduser(str(path))
        if os.path.exists(local_path):
            return local_path
        # Protocol-prefixed paths (file://, http(s)://, s3://, ...) were handled
        # by the stages' own fsspec machinery before the residency layer existed;
        # keep accepting them when the target exists remotely.
        if "://" in str(path):
            try:
                from fsspec.core import url_to_fs

                fs, fspath = url_to_fs(str(path))
                if fs.exists(fspath):
                    return path
            except Exception:  # noqa: BLE001, S110 - unknown protocol/creds -> deliberate fall-through
                pass

    if residency == "file":
        # Pre-residency stages handed unverified paths straight to their own
        # downstream machinery (ffmpeg/NeMo/fsspec) and let it report the
        # failure; keep that contract instead of gating on os.path.exists.
        return local_path

    waveform = item.get(waveform_key)
    sample_rate = item.get(sample_rate_key)
    if waveform is None or sample_rate is None:
        return local_path

    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=temp_dir)
    os.close(fd)
    sf.write(tmp, _as_soundfile_array(waveform), int(sample_rate))
    if register_temp is not None:
        register_temp.append(tmp)
    return tmp


def cleanup_temp_files(paths: list[str] | None) -> None:
    """Best-effort removal of temp files created by :func:`resolve_audio_path`."""
    for path in paths or ():
        with contextlib.suppress(OSError):
            os.remove(path)


def produce_audio_filepath(
    item: dict[str, Any],
    new_path: str,
    *,
    key: str = "audio_filepath",
    original_key: str = "original_audio_filepath",
) -> None:
    """Update a canonical audio path while preserving the first prior value."""
    if key in item and original_key not in item:
        item[original_key] = item[key]
    item[key] = new_path
