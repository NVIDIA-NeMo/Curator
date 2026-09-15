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

"""The agent's eyes: DataProfiler (input data) and EnvProbe (the machine).

Both are deterministic, cheap, and read-only. They feed the confirm gate (scale),
the pre-flight in ``validate`` (deps/GPU/paths/SR), and the PlanningContext. They
introduce no memory/learning — every call re-reads current reality.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

from nemo_curator.audio_agent.contracts import DataProfile, EnvProfile

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac")
_TRANSCRIPT_KEYS = ("text", "text_ref", "reference_text", "transcript", "pred_text")
# Cap how many files we probe so profiling a huge corpus stays fast (sampling).
_MAX_PROBE = 256
# How many per-file inventory entries are worth persisting for delta reuse (see
# ``_keep_inventory``). Deliberately much smaller than the stat cap.
_MAX_INVENTORY = 20_000
# Cap how many files we stat for the content-ish dataset key. Beyond this the key falls back
# to the cheap shape tier (and says so) rather than making profiling slow.
_MAX_STAT = 100_000
# Files a STAGE wrote back into the source folder, not user data. SplitLongAudioStage names
# its chunks "<stem>.<k>_of_<N>.<ext>" beside the input, which would otherwise inflate
# num_files and make a split pipeline invalidate its own dataset key on the next run.
_STAGE_INTERMEDIATE_RE = re.compile(r"\.\d+_of_\d+\.[a-z0-9]+$", re.IGNORECASE)


def _is_stage_intermediate(path: str, siblings: set[str] | None = None) -> bool:
    """True only when a file is CORROBORATED as a chunk a stage wrote beside its input.

    The name pattern alone is not evidence. ``interview.1_of_3.wav`` is an ordinary thing for a
    person to call their own recording, and excluding one hides it from the dataset key -- so
    editing it would not invalidate anything and a stale result would be served as current.

    A real chunk was split FROM a file that is still sitting beside it under the same stem, so
    that sibling is the corroboration. Without it the file counts as user data, which at worst
    re-runs work the key could have matched. The other mistake serves the wrong bytes, and between
    a slow answer and a wrong one this fails towards slow.
    """
    name = os.path.basename(path)
    match = _STAGE_INTERMEDIATE_RE.search(name)
    if not match:
        return False
    if siblings is None:
        return True  # no directory listing to check against; caller has not offered one
    stem = name[: match.start()].lower()
    return any(f"{stem}{ext}" in siblings for ext in _AUDIO_EXTS)


def _stat_entry(path: str, *, root: str) -> tuple[str, bool]:
    """Return the size+mtime identity token and whether its metadata is complete.

    The path is relative to ``root`` so the same corpus copied elsewhere still matches.
    ``?`` remains in the low-trust identity when a stat fails, but the success flag prevents
    that incomplete token from ever being presented as the high-trust ``stat`` tier.
    """
    relpath = os.path.relpath(path, root)
    try:
        st = os.stat(path)
    except OSError:
        return f"{relpath}|?", False
    return f"{relpath}|{st.st_size}|{st.st_mtime_ns}", True


def _split_entry(entry: str) -> tuple[str, str]:
    """An identity token split into ``(relpath, rest)`` -- the inventory's key and value."""
    relpath, _, rest = entry.partition("|")
    return relpath, rest


# --------------------------------------------------------------------------- #
# Data profiling
# --------------------------------------------------------------------------- #
def profile_data(  # noqa: PLR0913
    source: str,
    *,
    audio_filepath_key: str = "audio_filepath",
    max_probe: int = _MAX_PROBE,
    audio_dir: str | None = None,
    audio_path_resolution: str | None = None,
    identity_files: list[str] | tuple[str, ...] | None = None,
    folder_extensions: list[str] | tuple[str, ...] | None = None,
    recursive: bool = True,
    max_files: int | None = None,
    case_sensitive_extensions: bool = False,
    exclude_stage_intermediates: bool = True,
) -> DataProfile:
    """Profile a manifest (JSONL) or a folder of audio files.

    Reads sample rates / channels / durations / codecs from up to ``max_probe``
    files (via ``soundfile``), detects transcript presence, and records any
    unreadable files. Never raises on bad input — problems become ``notes`` /
    ``unreadable`` entries so the agent can triage. ``audio_dir`` plus
    ``audio_path_resolution`` mirrors ``ReadLongFormManifestStage`` when that
    source deliberately re-anchors manifest paths. ``identity_files`` folds
    additional local dataset-definition files (for example a FLEURS transcript
    TSV) into the same identity without changing ordinary profiler defaults.
    The folder-selection options let a source adapter mirror the files its stage
    really emits; their defaults preserve the profiler's historical behavior.
    """
    prof = DataProfile(source=source)
    expanded = os.path.expanduser(str(source))

    if os.path.isdir(expanded):
        prof.kind = "folder"
        paths = _list_audio_files(
            expanded,
            prof,
            extensions=folder_extensions,
            recursive=recursive,
            case_sensitive_extensions=case_sensitive_extensions,
            exclude_stage_intermediates=exclude_stage_intermediates,
        )
        if isinstance(max_files, int) and max_files >= 0:
            paths = paths[:max_files]
        elif max_files is not None:
            prof.notes.append(f"ignored invalid max_files value {max_files!r}")
        prof.num_files = len(paths)
        _probe_files(paths[:max_probe], prof)
        _stat_folder(paths, root=expanded, prof=prof)
    elif expanded.endswith((".jsonl", ".json")) and os.path.isfile(expanded):
        prof.kind = "manifest"
        _profile_manifest(
            expanded,
            prof,
            audio_filepath_key=audio_filepath_key,
            max_probe=max_probe,
            audio_dir=audio_dir,
            audio_path_resolution=audio_path_resolution,
        )
    elif os.path.isfile(expanded) and expanded.endswith(_AUDIO_EXTS):
        prof.kind = "folder"
        prof.num_files = 1
        _probe_files([expanded], prof)
        _stat_folder([expanded], root=os.path.dirname(expanded) or ".", prof=prof)
    else:
        prof.notes.append(f"could not interpret source {source!r} as a manifest or an audio folder")

    if identity_files:
        _fold_identity_files(prof, identity_files)
    if prof.num_files:
        prof.mean_duration_sec = round(prof.total_duration_sec / max(1, _probed_count(prof)), 3)
    return prof


def _list_audio_files(  # noqa: C901, PLR0913
    folder: str,
    prof: DataProfile | None = None,
    *,
    extensions: list[str] | tuple[str, ...] | None = None,
    recursive: bool = True,
    case_sensitive_extensions: bool = False,
    exclude_stage_intermediates: bool = True,
) -> list[str]:
    """Audio files under ``folder``, excluding files a prior stage wrote back into it."""
    raw_extensions = _AUDIO_EXTS if extensions is None else extensions
    try:
        normalized = tuple(extension if extension.startswith(".") else f".{extension}" for extension in raw_extensions)
    except (AttributeError, TypeError):
        if prof is not None:
            prof.notes.append(f"could not interpret folder extensions {raw_extensions!r}")
        return []
    if not case_sensitive_extensions:
        normalized = tuple(extension.lower() for extension in normalized)

    out: list[str] = []
    skipped = 0
    if recursive:
        directories = os.walk(folder)
    else:
        try:
            files = [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
        except OSError:
            files = []
        directories = [(folder, [], files)]
    for root, _dirs, files in directories:
        here = {f.lower() for f in files}  # sibling listing that corroborates a chunk's source
        for f in files:
            candidate = f if case_sensitive_extensions else f.lower()
            if not candidate.endswith(normalized):
                continue
            if exclude_stage_intermediates and _is_stage_intermediate(f, here):
                skipped += 1
                continue
            out.append(os.path.join(root, f))
    if prof is not None and skipped:
        prof.excluded_intermediates = skipped
        prof.notes.append(
            f"excluded {skipped} stage-written intermediate file(s) (e.g. split chunks) from the source scan"
        )
    return sorted(out)


def _stat_folder(paths: list[str], *, root: str, prof: DataProfile) -> None:
    """Fold every file's (relpath, size, mtime) into the content-ish dataset key."""
    if len(paths) > _MAX_STAT:
        identity = hashlib.sha256()
        for p in paths:
            identity.update(os.path.relpath(p, root).encode("utf-8"))
            identity.update(b"\n")
        prof.identity_digest = identity.hexdigest()[:16]
        prof.notes.append(
            f"{len(paths)} files exceeds the stat cap ({_MAX_STAT}); dataset key falls back to the shape tier"
        )
        return
    h = hashlib.sha256()
    failures = 0
    inventory: dict[str, str] = {}
    for p in paths:
        entry, ok = _stat_entry(p, root=root)
        h.update(entry.encode("utf-8"))
        h.update(b"\n")
        if ok:
            relpath, token = _split_entry(entry)
            inventory[relpath] = token
        else:
            failures += 1
    digest = h.hexdigest()[:16]
    if failures:
        prof.identity_digest = digest
        prof.notes.append(f"could not stat {failures} source file(s); dataset key falls back to the shape tier")
    else:
        prof.stat_digest = digest
        _keep_inventory(prof, inventory, root=root)


def _keep_inventory(prof: DataProfile, inventory: dict[str, str], *, root: str, key: str = "") -> None:
    """Attach the per-file inventory, or decline to when it is too large to carry.

    The cap is far below the stat cap on purpose: statting 100k files costs one syscall each
    and produces sixteen hex characters, while remembering them costs a file that has to be
    written, read and kept in step on every run. Past the cap the dataset key still works
    exactly as before and only delta reuse is unavailable, which ``delta`` reports by name.
    """
    if len(inventory) > _MAX_INVENTORY:
        prof.notes.append(
            f"{len(inventory)} files exceeds the inventory cap ({_MAX_INVENTORY}); "
            "reuse still works, but a changed-file delta cannot be computed for this corpus"
        )
        return
    prof.inventory = inventory
    # Recorded rather than re-derived later: the root a relpath is relative to depends on the
    # source kind and, for a manifest, on how it resolves audio paths. Re-deriving that in the
    # delta would be a second copy of a rule that has to agree with this one exactly. Same for
    # the COLUMN these paths were read from: a delta narrows a source by handing it these paths,
    # which only selects the right rows if the source matches them against the same column.
    prof.inventory_root = root
    prof.inventory_key = key


def _profile_manifest(  # noqa: C901, PLR0912, PLR0913, PLR0915
    path: str,
    prof: DataProfile,
    *,
    audio_filepath_key: str,
    max_probe: int,
    audio_dir: str | None = None,
    audio_path_resolution: str | None = None,
) -> None:
    audio_paths: list[str] = []
    keys: set[str] = set()
    has_transcript_value = False
    count = 0
    # The manifest's own bytes ARE the dataset definition, so hash them all; then fold in each
    # referenced audio file's size+mtime so audio edited in place (manifest untouched) is caught.
    content = hashlib.sha256()
    audio_stats = hashlib.sha256()
    root = (
        os.path.abspath(os.path.expanduser(audio_dir))
        if audio_dir and audio_path_resolution in ("basename", "relative")
        else os.path.dirname(os.path.abspath(path)) or "."
    )
    statted = 0
    stat_failures = 0
    relative_refs = 0
    remote_refs = 0
    # relpath -> [stat token, row digest, ...]; folded into one token per file below.
    row_identity: dict[str, list[str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()  # noqa: PLW2901
                if not line:
                    continue
                count += 1
                content.update(line.encode("utf-8"))
                content.update(b"\n")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    location = f"{path}:line{count}"
                    prof.unreadable.append(location)
                    prof.source_errors.append(f"{location}: invalid JSON ({exc.msg})")
                    continue
                if not isinstance(row, dict):
                    prof.source_errors.append(f"{path}:line{count}: manifest row must be a JSON object")
                    continue
                keys.update(row.keys())
                # A transcript COLUMN existing isn't enough -- an all-empty "text" field
                # would falsely imply transcripts (and e.g. that WER is computable). Require
                # at least one row to carry a non-empty transcript value.
                if not has_transcript_value:
                    has_transcript_value = any(str(row.get(k) or "").strip() for k in _TRANSCRIPT_KEYS)
                ap = row.get(audio_filepath_key)
                if ap:
                    expanded_ap = os.path.expanduser(str(ap))
                    if audio_path_resolution == "basename" and audio_dir:
                        expanded_ap = os.path.join(
                            os.path.expanduser(audio_dir),
                            os.path.basename(expanded_ap),
                        )
                    elif audio_path_resolution == "relative" and audio_dir:
                        expanded_ap = os.path.join(os.path.expanduser(audio_dir), expanded_ap)
                    elif audio_path_resolution not in (None, "as_is"):
                        # The source stage itself rejects this mode. Keep the
                        # profiler non-throwing, but never manufacture a trusted
                        # identity for a path interpretation execution will reject.
                        relative_refs += 1
                        audio_stats.update(f"invalid_resolution|{audio_path_resolution}|{expanded_ap}".encode())
                        continue
                    if "://" in expanded_ap:
                        # A URI cannot be statted or probed with the local filesystem APIs.
                        # Its spelling is still folded into the low-trust source identity.
                        remote_refs += 1
                        audio_stats.update(f"remote|{expanded_ap}".encode())
                    elif not os.path.isabs(expanded_ap):
                        # ManifestReader consumes relative paths as authored (cwd-relative).
                        # Preserve that probing behavior, but never claim cwd-dependent
                        # metadata as a portable, high-trust stat identity.
                        relative_refs += 1
                        audio_stats.update(f"relative|{expanded_ap}".encode())
                        if len(audio_paths) < max_probe:
                            audio_paths.append(expanded_ap)
                    else:
                        if len(audio_paths) < max_probe:
                            audio_paths.append(expanded_ap)
                        if statted < _MAX_STAT:
                            entry, ok = _stat_entry(expanded_ap, root=root)
                            audio_stats.update(entry.encode("utf-8"))
                            statted += 1
                            if not ok:
                                stat_failures += 1
                            else:
                                relpath, token = _split_entry(entry)
                                # The row's own bytes belong in this file's identity: a manifest
                                # is edited far more often than the audio it points at, and a
                                # corrected transcript with the wav untouched has to read as a
                                # change to that file or a delta would skip it. Rows accumulate,
                                # since several may reference one file.
                                seen = row_identity.setdefault(relpath, [token])
                                seen.append(hashlib.sha256(line.encode("utf-8")).hexdigest()[:12])
    except (OSError, UnicodeError) as exc:
        prof.unreadable.append(path)
        prof.source_errors.append(f"{path}: could not read complete UTF-8 manifest ({exc})")
    prof.num_files = count
    prof.manifest_keys = sorted(keys)
    prof.has_transcripts = has_transcript_value
    digest = hashlib.sha256(content.hexdigest().encode("utf-8") + audio_stats.hexdigest().encode("utf-8")).hexdigest()[
        :16
    ]
    incomplete = bool(prof.source_errors or stat_failures or relative_refs or remote_refs)
    if statted >= _MAX_STAT:
        incomplete = True
        prof.notes.append(
            f"referenced files exceed the stat cap ({_MAX_STAT}); dataset key falls back to the shape tier"
        )
    if stat_failures or relative_refs or remote_refs:
        reasons = []
        if stat_failures:
            reasons.append(f"{stat_failures} local file stat failure(s)")
        if relative_refs:
            reasons.append(f"{relative_refs} relative reference(s)")
        if remote_refs:
            reasons.append(f"{remote_refs} remote reference(s)")
        prof.notes.append(
            f"incomplete referenced-file metadata ({', '.join(reasons)}); dataset key falls back to the shape tier"
        )
    if incomplete:
        prof.identity_digest = digest
    else:
        prof.stat_digest = digest
        _keep_inventory(
            prof,
            {rel: "|".join(parts) for rel, parts in row_identity.items()},
            root=root,
            key=audio_filepath_key,
        )
    _probe_files(audio_paths, prof)
    # Rows were read but not one audio reference was found: the audio-path column is named
    # something other than ``audio_filepath_key``. Say so, because a silently EMPTY audio
    # profile (no rates, no durations, no unreadable entries) is indistinguishable from a
    # healthy one, and every downstream decision that reads those fields is then reasoning
    # from evidence that was never gathered.
    if count and not audio_paths:
        prof.notes.append(
            f"no rows carried an audio path under {audio_filepath_key!r}; the manifest's columns are "
            f"{prof.manifest_keys} -- sample rates, durations and codecs were NOT read. "
            "Point the reading stage (and this profile) at the real column."
        )


def _fold_identity_files(prof: DataProfile, paths: list[str] | tuple[str, ...]) -> None:
    """Fold supplemental local definition files into a profile's dataset key.

    Their complete bytes are hashed because these are small metadata files, not
    the audio corpus itself. Any inaccessible or remote entry keeps the resulting
    key in the low-trust shape tier.
    """
    h = hashlib.sha256()
    base = prof.stat_digest or prof.identity_digest or prof.fingerprint()
    h.update(f"base|{base}".encode())
    h.update(b"\n")
    complete = bool(prof.stat_digest)
    failures = 0
    for index, raw in enumerate(paths):
        path = os.path.expanduser(str(raw))
        if "://" in path and not path.startswith("file://"):
            h.update(f"{index}|remote|{path}".encode())
            h.update(b"\n")
            complete = False
            failures += 1
            continue
        if path.startswith("file://"):
            from urllib.parse import unquote, urlsplit

            path = unquote(urlsplit(path).path)
        try:
            content = hashlib.sha256()
            with open(path, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    content.update(chunk)
            h.update(f"{index}|{os.path.basename(path)}|{content.hexdigest()}".encode())
            h.update(b"\n")
        except OSError:
            h.update(f"{index}|{os.path.basename(path)}|?".encode())
            h.update(b"\n")
            complete = False
            failures += 1
    digest = h.hexdigest()[:16]
    if complete:
        prof.stat_digest = digest
        prof.identity_digest = ""
    else:
        prof.stat_digest = ""
        prof.identity_digest = digest
        if failures:
            prof.notes.append(
                f"could not fully identify {failures} supplemental dataset file(s); "
                "dataset key falls back to the shape tier"
            )


def _probe_files(paths: list[str], prof: DataProfile) -> None:
    try:
        import soundfile as sf
    except Exception:  # noqa: BLE001 - soundfile is an audio-extra dep
        prof.notes.append("soundfile not installed; cannot read audio headers (install an audio extra)")
        return
    srs: Counter[int] = Counter()
    chans: Counter[int] = Counter()
    codecs: Counter[str] = Counter()
    for p in paths:
        if not os.path.exists(p):
            prof.unreadable.append(p)
            continue
        try:
            info = sf.info(p)
            srs[int(info.samplerate)] += 1
            chans[int(info.channels)] += 1
            codecs[str(info.format)] += 1
            prof.total_duration_sec += float(info.frames) / float(info.samplerate or 1)
        except Exception:  # noqa: BLE001 - corrupt/unsupported file
            prof.unreadable.append(p)
    prof.sample_rates = dict(srs)
    prof.channels = dict(chans)
    prof.codecs = dict(codecs)


def _probed_count(prof: DataProfile) -> int:
    return sum(prof.sample_rates.values()) or 1


# --------------------------------------------------------------------------- #
# Environment probing
# --------------------------------------------------------------------------- #
_AUDIO_PACKAGES = {
    "soundfile": "soundfile",
    "torchaudio": "torchaudio",
    "silero_vad": "silero-vad",
    "librosa": "librosa",
    "onnxruntime": "onnxruntime",
    "nemo": "nemo_toolkit[asr]",
    "whisperx": "whisperx",
    "pyannote.audio": "pyannote-audio",
    "nemo_text_processing": "nemo_text_processing",
}
_KNOWN_SECRET_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NVIDIA_API_KEY", "AWS_ACCESS_KEY_ID")


def probe_env() -> EnvProfile:
    """Probe GPU / ffmpeg / installed audio deps / secrets / Curator version."""
    env = EnvProfile()
    env.has_ffmpeg = shutil.which("ffmpeg") is not None
    if not env.has_ffmpeg:
        env.notes.append("ffmpeg not on PATH; resample/convert stages will fail")

    _probe_gpu(env)
    _probe_cuda_compat(env)
    _probe_packages(env)
    _probe_resources(env)

    env.available_secrets = [k for k in _KNOWN_SECRET_ENVS if os.environ.get(k)]
    try:
        import nemo_curator

        env.curator_version = getattr(nemo_curator, "__version__", "") or ""
    except Exception:  # noqa: BLE001
        env.curator_version = ""
    _probe_python(env)
    return env


def _probe_python(env: EnvProfile) -> None:
    """Record the interpreter version and whether it satisfies the project's requires-python.

    A version mismatch is a common, hard-to-spot cause of import / CUDA / model failures, so
    surface it up front. Just as important: on a *supported* version this confirms the
    interpreter is fine, so it isn't wrongly blamed when a failure is really elsewhere.
    """
    env.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        from importlib.metadata import metadata

        req = metadata("nemo-curator").get("Requires-Python") or ""
    except Exception:  # noqa: BLE001 - dist metadata may be unavailable (odd installs)
        return
    if not req:
        return
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        env.python_supported = SpecifierSet(req).contains(Version(env.python_version), prereleases=True)
    except Exception:  # noqa: BLE001 - packaging missing / unparseable specifier -> don't guess
        return
    if not env.python_supported:
        env.notes.append(
            f"Python {env.python_version} is OUTSIDE the project's requires-python {req!r} -- "
            "imports or GPU/model stages may fail; use a supported interpreter."
        )


def _probe_gpu(env: EnvProfile) -> None:  # noqa: C901, PLR0912, PLR0915 - layered probe retains distinct failure modes
    """Probe the GPU stack in layers instead of collapsing every failure to "no GPU".

    A user can have physical NVIDIA hardware while this process sees no device
    because the scheduler/container did not expose it, ``CUDA_VISIBLE_DEVICES``
    masked it, the driver is unavailable, or torch is a CPU-only build. Those
    cases need different remediation choices, so retain the evidence separately.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        env.cuda_visible_devices = "unset"
    elif not visible.strip():
        env.cuda_visible_devices = "empty"
    elif visible.strip() == "-1":
        env.cuda_visible_devices = "masked"
    else:
        env.cuda_visible_devices = "set"

    try:
        env.nvidia_device_nodes = sum(1 for name in os.listdir("/dev") if re.fullmatch(r"nvidia\d+", name))
    except OSError:
        env.nvidia_device_nodes = 0

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        env.nvidia_smi_status = "missing"
    else:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed, discovered executable; read-only probe
                [nvidia_smi, "-L"],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0:
                env.nvidia_smi_status = "ok"
                env.nvidia_smi_gpu_count = sum(line.lstrip().startswith("GPU ") for line in proc.stdout.splitlines())
            else:
                env.nvidia_smi_status = "driver_or_visibility_error"
        except subprocess.TimeoutExpired:
            env.nvidia_smi_status = "timeout"
        except OSError:
            env.nvidia_smi_status = "unavailable"

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch import itself is an environment fact
        env.gpu_visibility = "torch_unavailable"
        env.cuda_probe_error = f"{type(exc).__name__}: {exc}"[:300]
        env.notes.append("torch could not be imported; GPU execution is unavailable")
        return

    env.torch_version = str(getattr(torch, "__version__", "") or "")
    env.cuda_runtime_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    env.torch_cuda_built = bool(env.cuda_runtime_version)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - preserve the layer that failed
        env.cuda_probe_error = f"{type(exc).__name__}: {exc}"[:300]
        cuda_available = False

    if cuda_available:
        try:
            env.gpu_count = int(torch.cuda.device_count())
        except Exception as exc:  # noqa: BLE001 - keep import success distinct from device probing
            env.cuda_probe_error = f"{type(exc).__name__}: {exc}"[:300]
            env.gpu_count = 0
        if env.gpu_count > 0:
            env.has_gpu = True
            env.gpu_visibility = "available"
            for index in range(env.gpu_count):
                try:
                    env.gpu_names.append(str(torch.cuda.get_device_name(index)))
                except Exception as exc:  # noqa: BLE001 - one metadata lookup must not erase visibility
                    env.gpu_names.append(f"GPU {index}")
                    if not env.cuda_probe_error:
                        env.cuda_probe_error = f"{type(exc).__name__}: {exc}"[:300]
            try:
                env.gpu_mem_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    1,
                )
            except Exception as exc:  # noqa: BLE001 - property lookup can fail on odd drivers
                if not env.cuda_probe_error:
                    env.cuda_probe_error = f"{type(exc).__name__}: {exc}"[:300]
            return

    if not env.torch_cuda_built:
        env.gpu_visibility = "cpu_only_torch"
    elif env.cuda_visible_devices in {"empty", "masked"}:
        env.gpu_visibility = "masked_by_cuda_visible_devices"
    elif env.nvidia_smi_status == "ok" or env.nvidia_device_nodes > 0:
        env.gpu_visibility = "torch_cuda_unavailable"
    elif env.nvidia_smi_status in {"driver_or_visibility_error", "timeout"}:
        env.gpu_visibility = "driver_or_device_exposure_error"
    else:
        env.gpu_visibility = "not_detected"
    env.notes.append(
        f"torch CUDA unavailable (visibility={env.gpu_visibility}, "
        f"nvidia_smi={env.nvidia_smi_status}); execution is CPU-only"
    )
    # A CUDA-capable torch (or visible NVIDIA hardware) that still can't reach a device is
    # usually an ACCESS problem, not absent hardware: a sandbox/container blocking
    # /dev/nvidia* (or CUDA_VISIBLE_DEVICES masking) looks exactly like this. Flag it so a
    # caller never reports "no GPU" as a hardware fact from a possibly-restricted process.
    # A plain CPU-only torch build is a real build fact, not masking, so it is excluded.
    if env.gpu_visibility != "cpu_only_torch" and (
        env.torch_cuda_built
        or env.nvidia_smi_status == "ok"
        or env.nvidia_device_nodes > 0
        or env.cuda_visible_devices in {"empty", "masked"}
    ):
        env.gpu_possibly_masked = True
        env.notes.append(
            "a GPU may be PRESENT but not reachable from THIS process (e.g. a sandbox/"
            "container blocking /dev/nvidia*, or CUDA_VISIBLE_DEVICES masking); re-verify "
            "with full device access before concluding no GPU"
        )


def _cuda_ver_tuple(v: str) -> tuple[int, ...]:
    """'12.9' -> (12, 9); tolerant of junk -> ()."""
    try:
        return tuple(int(p) for p in v.split(".")[:2])
    except (ValueError, AttributeError):
        return ()


def _driver_max_cuda() -> str:
    """The MAX CUDA version the installed GPU driver supports, via ``cuDriverGetVersion``
    (dependency-free; libcuda is the driver). Returns e.g. '12.6', or '' if undeterminable.

    cuDriverGetVersion encodes the version as ``1000*major + 10*minor`` (12060 -> 12.6). This is
    the driver's *ceiling*, independent of the CUDA toolkit torch bundles.
    """
    import ctypes

    for libname in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            lib = ctypes.CDLL(libname)
        except OSError:
            continue
        try:
            ver = ctypes.c_int()
            if lib.cuDriverGetVersion(ctypes.byref(ver)) == 0 and ver.value > 0:
                major, rem = divmod(ver.value, 1000)
                return f"{major}.{rem // 10}"
        except Exception:  # noqa: BLE001 - symbol/ABI oddities -> undeterminable
            return ""
    return ""


def _probe_cuda_compat(env: EnvProfile) -> None:
    """Flag a GPU DRIVER older than the CUDA toolkit torch was built with.

    This is the real, high-signal gate behind the "CUDA-graph decode crashes" class of failure:
    basic (precompiled) kernels run under CUDA minor-version compatibility, but anything that
    JIT-compiles PTX at runtime -- NVRTC / CUDA-graph decoders (e.g. NeMo RNNT/TDT) -- targets
    the toolkit's PTX ISA, which an older driver cannot load (CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
    error 222). It is NOT a Python-version problem. Best-effort: if either version is unknown we
    leave ``cuda_compatible=True`` (don't guess / don't false-alarm).
    """
    if not env.has_gpu:
        # Still retain the driver ceiling when libcuda is visible. It helps
        # distinguish a hidden device from a CPU-only install in ``doctor``.
        env.cuda_driver_max_version = _driver_max_cuda()
        return
    try:
        import torch

        env.cuda_runtime_version = torch.version.cuda or ""
    except Exception:  # noqa: BLE001 - torch missing/odd -> nothing to compare
        return
    env.cuda_driver_max_version = _driver_max_cuda()
    built = _cuda_ver_tuple(env.cuda_runtime_version)
    driver_max = _cuda_ver_tuple(env.cuda_driver_max_version)
    if not built or not driver_max:
        return  # can't compare reliably -> stay silent
    if driver_max < built:
        env.cuda_compatible = False
        env.notes.append(
            f"CUDA driver/toolkit MISMATCH: torch is built for CUDA {env.cuda_runtime_version} but the GPU "
            f"driver supports only CUDA {env.cuda_driver_max_version}. Basic GPU ops work, but runtime-JIT / "
            "CUDA-graph kernels (e.g. NeMo RNNT/TDT ASR decode) can fail with CUDA_ERROR_UNSUPPORTED_PTX_VERSION "
            "(error 222). Fix: upgrade the NVIDIA driver, or install a torch built for CUDA "
            f"<= {env.cuda_driver_max_version}; quick unblock for ASR alignment: decoder_type='ctc'."
        )


def _probe_packages(env: EnvProfile) -> None:
    import importlib.util

    installed: list[str] = []
    missing: list[str] = []
    for module, pkg in _AUDIO_PACKAGES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except Exception:  # noqa: BLE001 - namespace-package edge cases
            found = False
        (installed if found else missing).append(pkg)
    env.installed_extras = sorted(set(installed))
    env.missing_packages = sorted(set(missing))


def _probe_resources(env: EnvProfile) -> None:
    """CPU count, host RAM, and free disk for the resource planner (best-effort)."""
    env.total_cpus = os.cpu_count() or 0
    try:
        import psutil

        env.total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001 - psutil optional; fall back to /proc/meminfo
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        env.total_ram_gb = round(int(line.split()[1]) / (1024**2), 1)  # kB -> GB
                        break
        except Exception:  # noqa: BLE001, S110 - /proc unavailable (non-Linux)
            pass
    try:  # noqa: SIM105
        env.free_disk_gb = round(shutil.disk_usage(os.getcwd()).free / (1024**3), 1)
    except Exception:  # noqa: BLE001, S110 - disk_usage can fail on odd mounts
        pass
