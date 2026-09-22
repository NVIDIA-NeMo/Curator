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

"""Which code produced an artifact -- per stage, not per repository.

An artifact records the implementation that produced it so a later run can tell whether the
code that would recompute the step is the code that already did. That stamp used to be
``nemo_curator.__version__``, which ends in the repository's git short SHA, and it was wrong
in both directions at once: committing a README moved the SHA and made every artifact in the
store unreachable, while editing a stage and not committing left the SHA alone and served
results produced by code that no longer exists.

Here the stamp is a digest of the source that actually implements the stage -- the module
defining its class plus the ``nemo_curator`` modules that module transitively imports.
Editing ``nemo_asr_align.py`` invalidates ASR artifacts and everything chained below them;
editing the video pipeline invalidates nothing here.

Imports are read out of the source with :mod:`ast` and resolved against the package
directory rather than through ``importlib``, so computing a stamp imports nothing: a stage
whose optional dependency is missing still gets an answer, and a hash lookup can never
execute module-level code as a side effect.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_PACKAGE = "nemo_curator"
# Marks a stamp that could not be computed from source. Read by ``is_fallback`` so the
# prefix stays knowledge of this module rather than a string other modules match on.
_FALLBACK_PREFIX = "pkg:"
# A stage reaches 20-30 modules today. This is not a tuning knob: a closure that large means
# the walk found something it should not have, and hashing a partial closure would miss the
# very edit the stamp exists to catch. Over the cap, fall back to the value that only ever
# over-invalidates.
_CLOSURE_LIMIT = 400

# Keyed by stage ref. Cached for the life of the process on purpose: the stage's modules are
# already imported, so these bytes describe the code that is actually running even if the
# file on disk is edited underneath a long-lived server.
_version_cache: dict[str, str] = {}
_file_digest_cache: dict[str, str] = {}
# Stages overlap heavily -- twelve of them share about forty files -- so parsing and hashing
# are memoized by file as well as by stage. Without it a full recipe re-parses the same base
# modules a dozen times and the first plan of a process pays hundreds of milliseconds.
_imports_cache: dict[tuple[str, str], list[str]] = {}
_root_cache: list[str | None] = []


def impl_version(stage_ref: str) -> str:
    """A digest of the code implementing ``stage_ref`` (``"impl:..."`` or a fallback)."""
    cached = _version_cache.get(stage_ref)
    if cached is None:
        cached = _version_cache[stage_ref] = _compute(stage_ref)
    return cached


def _compute(stage_ref: str) -> str:
    try:
        from nemo_curator.audio_agent._resolve import resolve_stage_class

        files = _closure(resolve_stage_class(stage_ref).__module__)
        if not files:
            return _fallback()
        h = hashlib.sha256()
        for name in sorted(files):
            h.update(name.encode("utf-8"))
            h.update(b"\x1f")
            h.update(_file_digest(files[name]).encode("ascii"))
            h.update(b"\x1f")
    except Exception:  # noqa: BLE001 - an unreadable stage must degrade, never raise
        return _fallback()
    return f"impl:{h.hexdigest()[:16]}"


def is_fallback(version: str) -> bool:
    """Whether ``version`` is a fallback stamp rather than a real closure digest.

    The distinction is the whole diagnostic: two stamps can differ because the source was
    edited, or because one side could not be read at all. Only the first means the
    implementation changed, and telling a user their code moved when the truth is that this
    environment could not see it sends them looking for an edit nobody made.
    """
    return version.startswith(_FALLBACK_PREFIX)


def unreadable_stages(refs: Iterable[str]) -> list[str]:
    """Which of ``refs`` cannot be stamped from source in this environment.

    A delta that silently declines because every step key was computed from a fallback looks
    exactly like a delta that declines because the corpus moved on. This names the difference
    so the caller can report it.
    """
    return [ref for ref in dict.fromkeys(refs) if is_fallback(impl_version(ref))]


def _fallback() -> str:
    """The package version -- what every stamp used to be.

    Reached when the stage, its module, or its sources cannot be read. It over-invalidates
    (any commit moves it) and never over-reuses, which is the direction an unknown has to
    fail in. Prefixed so a record shows at a glance that the closure was not available.
    """
    from nemo_curator.audio_agent.artifacts import code_version

    return f"{_FALLBACK_PREFIX}{code_version()}"


def _package_root() -> str | None:
    """Directory holding the installed ``nemo_curator`` package, or ``None``."""
    if not _root_cache:
        try:
            import nemo_curator

            path = getattr(nemo_curator, "__file__", "") or ""
            _root_cache.append(os.path.dirname(os.path.abspath(path)) if path else None)
        except Exception:  # noqa: BLE001 - degrade to the fallback stamp
            _root_cache.append(None)
    return _root_cache[0]


def _module_file(name: str, root: str) -> str | None:
    """Source file for a dotted module name, or ``None`` when it names no module.

    ``from x import y`` cannot say whether ``y`` is a submodule or a class, so both are
    tried as modules and the ones that name no file simply drop out here.
    """
    parts = name.split(".")
    if parts[0] != _PACKAGE:
        return None
    base = os.path.join(root, *parts[1:])
    for candidate in (f"{base}.py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _imported_modules(name: str, path: str) -> list[str]:
    """``nemo_curator`` module names imported by ``path``, function-local imports included.

    Read from the source rather than from the module's namespace because stages import their
    heavy dependencies inside ``setup()``, and a name bound in a function body never appears
    in ``module.__dict__``.
    """
    cached = _imports_cache.get((name, path))
    if cached is not None:
        return cached
    try:
        with open(path, "rb") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError, ValueError):
        _imports_cache[(name, path)] = []
        return []
    package = name if os.path.basename(path) == "__init__.py" else name.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package
            if node.level > 1:  # ``from ..x import y`` climbs out of the current package
                parts = package.split(".")
                base = ".".join(parts[: max(1, len(parts) - node.level + 1)])
            full = f"{base}.{node.module}" if node.level and node.module else (node.module or base)
            found.append(full)
            found.extend(f"{full}.{alias.name}" for alias in node.names)
    ours = [m for m in found if m == _PACKAGE or m.startswith(f"{_PACKAGE}.")]
    _imports_cache[(name, path)] = ours
    return ours


def _closure(start: str) -> dict[str, str] | None:
    """``{module name: source file}`` reachable from ``start``, or ``None`` if unbounded."""
    root = _package_root()
    if not root:
        return None
    files: dict[str, str] = {}
    seen: set[str] = set()
    queue = collections.deque([start])
    while queue:
        name = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        if len(seen) > _CLOSURE_LIMIT:
            return None
        path = _module_file(name, root)
        if path is None:
            continue
        files[name] = path
        queue.extend(m for m in _imported_modules(name, path) if m not in seen)
    return files or None


def _file_digest(path: str) -> str:
    """SHA-256 of a source file, cached because stages share most of their closure."""
    cached = _file_digest_cache.get(path)
    if cached is None:
        with open(path, "rb") as fh:
            cached = hashlib.sha256(fh.read()).hexdigest()
        _file_digest_cache[path] = cached
    return cached


def _reset_caches() -> None:
    """Forget every memoized digest (for tests that edit sources mid-process)."""
    _version_cache.clear()
    _file_digest_cache.clear()
    _imports_cache.clear()
    _root_cache.clear()
