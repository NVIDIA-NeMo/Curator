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

"""Concurrency-safe MFA model handling (adapted from ``CandidateMfaResolverProcessor._run_alignment_locked``).

Many workers share one ``MFA_ROOT_DIR``. MFA extracts pretrained models into ``extracted_models/`` on first use,
and concurrent extraction corrupts them. So per language:

1. the first worker bootstraps the model under an exclusive file lock (MFA downloads/extracts it) and freezes the
   extracted dictionary + acoustic model into an immutable ``model_bundles/<lang>.tar`` with a digest sidecar
   ``<lang>.tar.json``;
2. every later alignment - in any process - finds an up-to-date bundle without the exclusive lock, extracts it once
   per process into a private directory (under a short shared lock) and runs ``mfa align`` against those explicit
   paths, fully in parallel.

The bundle is rebuilt only if it is missing, its sidecar does not match, or the extracted source model changed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.tts.early_cutoff.mfa_runtime import run_alignment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

_MIN_ALIGN_ARGS = 6  # [mfa, align, corpus, dictionary, acoustic_model, output, ...]
BUNDLE_FORMAT = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(dictionary: Path, acoustic: Path) -> str:
    """Cheap fingerprint (relative path, size, mtime) of an extracted dictionary + acoustic model directory."""
    entries = [("dictionary.dict", dictionary.stat().st_size, dictionary.stat().st_mtime_ns)]
    for path in sorted(p for p in acoustic.rglob("*") if p.is_file()):
        stat = path.stat()
        entries.append((f"acoustic/{path.relative_to(acoustic)}", stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()


@contextmanager
def _flock(path: Path, mode: int) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass
class MfaModelCache:
    """Run ``mfa align`` commands against immutable per-language model bundles (see module docstring)."""

    root_dir: str | None = None
    lock_dir: str | None = None
    bundle_dir: str | None = None
    rebuild_bundles: bool = False
    runner: Callable[..., Any] = subprocess.run
    timeout_sec: float | None = None
    env: Mapping[str, str] | None = None
    local_model_dirs: dict[str, Path] = field(default_factory=dict)
    _checked: set[str] = field(default_factory=set, repr=False)

    @property
    def _root(self) -> Path | None:
        return Path(self.root_dir).expanduser() if self.root_dir else None

    def _base(self, work_root: Path) -> Path:
        return self._root or work_root

    def bundle_path(self, language: str, work_root: Path) -> Path:
        base = Path(self.bundle_dir).expanduser() if self.bundle_dir else self._base(work_root) / "model_bundles"
        return base / f"{language}.tar"

    def _lock_path(self, language: str, work_root: Path) -> Path:
        base = Path(self.lock_dir).expanduser() if self.lock_dir else self._base(work_root) / "alignment_locks"
        return base / f"{language}.lock"

    def _run(self, command: Sequence[str]) -> None:
        run_alignment(command, runner=self.runner, timeout_sec=self.timeout_sec, env=self.env)

    # ------------------------------------------------------------------ extracted source model

    def extracted_command(self, command: Sequence[str]) -> list[str] | None:
        """Command rewritten to MFA's extracted model paths, or ``None`` if the extraction is incomplete."""
        root = self._root
        if len(command) < _MIN_ALIGN_ARGS or command[1] != "align":
            return None
        dictionary_name, acoustic_name = str(command[3]), str(command[4])
        if Path(dictionary_name).is_file() and (Path(acoustic_name) / "final.mdl").is_file():
            return list(command)
        if root is None:
            return None
        dictionary = root / "pretrained_models" / "dictionary" / f"{dictionary_name}.dict"
        acoustic = root / "extracted_models" / "acoustic" / f"{acoustic_name}_acoustic"
        if not dictionary.is_file() or not (acoustic / "final.mdl").is_file() or not (acoustic / "tree").is_file():
            return None
        resolved = list(command)
        resolved[3], resolved[4] = str(dictionary), str(acoustic)
        return resolved

    def _remove_corrupt_acoustic_model(self, command: Sequence[str]) -> None:
        root = self._root
        if root is None or len(command) < _MIN_ALIGN_ARGS or Path(str(command[4])).is_absolute():
            return
        acoustic = root / "extracted_models" / "acoustic" / f"{command[4]}_acoustic"
        if acoustic.exists() and self.extracted_command(command) is None:
            shutil.rmtree(acoustic, ignore_errors=True)

    # ------------------------------------------------------------------ bundles

    @staticmethod
    def _sidecar(bundle: Path) -> Path:
        return bundle.with_name(bundle.name + ".json")

    def bundle_is_current(self, command: Sequence[str], bundle: Path) -> bool:
        """Bundle + matching sidecar exist and, if the extracted source is present, it is unchanged."""
        try:
            meta = json.loads(self._sidecar(bundle).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not bundle.is_file() or meta.get("format") != BUNDLE_FORMAT or meta.get("size") != bundle.stat().st_size:
            return False
        if meta.get("dictionary") != str(command[3]) or meta.get("acoustic_model") != str(command[4]):
            return False
        resolved = self.extracted_command(command)
        if resolved is None:  # source not extracted here (e.g. bundle shipped alone): trust the immutable bundle
            return True
        return meta.get("source_fingerprint") == source_fingerprint(Path(resolved[3]), Path(resolved[4]))

    def create_bundle(self, command: Sequence[str], resolved: Sequence[str], bundle: Path) -> None:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        temporary = bundle.with_suffix(".tar.tmp")
        temporary.unlink(missing_ok=True)
        with tarfile.open(temporary, "w") as archive:
            archive.add(Path(str(resolved[3])), arcname="dictionary.dict", recursive=False)
            archive.add(Path(str(resolved[4])), arcname="acoustic", recursive=True)
        meta = {
            "format": BUNDLE_FORMAT,
            "dictionary": str(command[3]),
            "acoustic_model": str(command[4]),
            "source_fingerprint": source_fingerprint(Path(str(resolved[3])), Path(str(resolved[4]))),
            "sha256": _sha256(temporary),
            "size": temporary.stat().st_size,
        }
        sidecar_tmp = self._sidecar(bundle).with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        os.replace(temporary, bundle)
        os.replace(sidecar_tmp, self._sidecar(bundle))
        logger.info("Built MFA model bundle {}", bundle)

    def _bootstrap(self, command: Sequence[str], language: str, bundle: Path) -> bool:
        """Make the extracted model complete and freeze it into the bundle.

        If the model is not extracted yet, ``command`` itself is run (MFA downloads/extracts the model while
        aligning, as in the reference); returns ``True`` in that case so the alignment is not repeated.
        """
        aligned = False
        resolved = self.extracted_command(command)
        if resolved is None:
            self._remove_corrupt_acoustic_model(command)
            self._run(command)
            aligned = True
            resolved = self.extracted_command(command)
            if resolved is None:
                msg = f"MFA model bootstrap incomplete for {language}"
                raise RuntimeError(msg)
        self.create_bundle(command, resolved, bundle)
        return aligned

    def ensure_bundle(self, command: Sequence[str], language: str, work_root: Path) -> tuple[Path, bool]:
        """Return ``(bundle, aligned)``; only a missing/stale bundle takes the exclusive per-language lock.

        ``aligned`` is ``True`` if building the bundle required running ``command`` (the alignment is done).
        """
        bundle = self.bundle_path(language, work_root)
        force = self.rebuild_bundles and language not in self._checked
        aligned = False
        if force or not self.bundle_is_current(command, bundle):
            with _flock(self._lock_path(language, work_root), fcntl.LOCK_EX):
                if force or not self.bundle_is_current(command, bundle):  # double-checked under the lock
                    aligned = self._bootstrap(command, language, bundle)
        self._checked.add(language)
        return bundle, aligned

    def local_command(self, command: Sequence[str], language: str, bundle: Path, work_root: Path) -> list[str]:
        """Command pointing at this process's private extraction of the bundle (extracted once per language)."""
        local = self.local_model_dirs.get(language)
        if local is None:
            local = Path(tempfile.mkdtemp(prefix=f"mfa_model_{language}_"))
            try:
                with _flock(self._lock_path(language, work_root), fcntl.LOCK_SH):
                    expected = json.loads(self._sidecar(bundle).read_text(encoding="utf-8")).get("sha256")
                    if expected and _sha256(bundle) != expected:
                        msg = f"MFA model bundle checksum mismatch: {bundle}"
                        raise RuntimeError(msg)  # noqa: TRY301
                    with tarfile.open(bundle, "r:") as archive:
                        archive.extractall(local, filter="data")
                if not (
                    (local / "dictionary.dict").is_file()
                    and (local / "acoustic/final.mdl").is_file()
                    and (local / "acoustic/tree").is_file()
                ):
                    msg = f"MFA model bundle is incomplete: {bundle}"
                    raise RuntimeError(msg)  # noqa: TRY301
            except Exception:
                shutil.rmtree(local, ignore_errors=True)
                self._checked.discard(language)
                raise
            self.local_model_dirs[language] = local
            logger.info("Extracted MFA model bundle for {} to {}", language, local)
        resolved = list(command)
        resolved[3], resolved[4] = str(local / "dictionary.dict"), str(local / "acoustic")
        return resolved

    def run(self, command: Sequence[str], language: str, work_root: Path) -> None:
        """Run one alignment against the language's bundle, building the bundle first if needed."""
        bundle = self.bundle_path(language, work_root)
        if language not in self.local_model_dirs:
            bundle, aligned = self.ensure_bundle(command, language, work_root)
            if aligned:
                return
        self._run(self.local_command(command, language, bundle, work_root))

    def cleanup(self) -> None:
        for directory in self.local_model_dirs.values():
            shutil.rmtree(directory, ignore_errors=True)
        self.local_model_dirs.clear()
        self._checked.clear()
