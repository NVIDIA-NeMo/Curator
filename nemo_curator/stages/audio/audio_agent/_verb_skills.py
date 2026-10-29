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

"""Installation and discovery of packaged host skills."""

from __future__ import annotations

import filecmp
import os
import shutil
from typing import Any

from nemo_curator.stages.audio.audio_agent import _safety

# Where each host looks for a project- and user-level skill. Codex scans ``.agents/skills``
# from the CWD up to the repo root and Cursor lists it as a project discovery root, so both
# share one directory; Claude Code reads only ``.claude/skills`` and needs its own entry.
_SKILL_HOST_DIRS: dict[str, tuple[str, str]] = {
    "codex": (".agents/skills", "~/.agents/skills"),
    "cursor": (".agents/skills", "~/.cursor/skills"),
    "claude": (".claude/skills", "~/.claude/skills"),
}


def skills_dir() -> str:
    """The packaged skill definitions — the single source of truth every host reads.

    Ships inside the wheel (``MANIFEST.in`` takes ``nemo_curator/**/*.md``), so a
    ``pip install`` user has the same instructions a checkout does, without ``.claude/``
    or ``.cursor/`` which live outside the package.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def available_skills() -> list[str]:
    """Packaged skill names, i.e. the directories under :func:`skills_dir` holding a SKILL.md."""
    root = skills_dir()
    if not os.path.isdir(root):
        return []
    return sorted(name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name, "SKILL.md")))


def _trees_match(source: str, target: str) -> bool:
    """Whether ``target`` already holds a byte-identical copy of everything in ``source``.

    Lets a repeated install report ``unchanged`` instead of rewriting files, and lets a
    genuinely different directory of the same name be reported as a conflict rather than
    silently overwritten.
    """
    for dirpath, _dirnames, filenames in os.walk(source):
        rel = os.path.relpath(dirpath, source)
        mirror = target if rel == "." else os.path.join(target, rel)
        if not os.path.isdir(mirror):
            return False
        for filename in filenames:
            dst_file = os.path.join(mirror, filename)
            if not os.path.isfile(dst_file):
                return False
            try:
                if not filecmp.cmp(os.path.join(dirpath, filename), dst_file, shallow=False):
                    return False
            except OSError:
                return False
    return True


def _is_symlink_farm(source: str, target: str) -> bool:
    """Whether ``target`` is a real directory whose entries are links to ``source``'s.

    This is the shape a symlink install leaves behind, so recognizing it is what makes a
    second install report ``unchanged`` rather than relaying identical links.
    """
    if not os.path.isdir(target) or os.path.islink(target):
        return False
    entries = os.listdir(source)
    if sorted(entries) != sorted(os.listdir(target)):
        return False
    return all(
        os.path.islink(os.path.join(target, entry))
        and os.path.realpath(os.path.join(target, entry)) == os.path.realpath(os.path.join(source, entry))
        for entry in entries
    )


def _names_its_own_target(path: str, source_entry: str) -> bool:
    """A plain file whose whole content is the path it was supposed to link to."""
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    try:
        if os.path.getsize(path) > 4096:  # noqa: PLR2004 - a path, not a document
            return False
        with open(path, encoding="utf-8", errors="strict") as handle:
            text = handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return False
    return bool(text) and "\n" not in text and os.path.basename(text) == os.path.basename(source_entry)


def _is_dead_shim(target: str, source: str) -> bool:
    """A shim git recorded but could not materialize, so it is text where a link belongs.

    On a Windows checkout without ``core.symlinks`` every committed link becomes a file
    holding its target path, and every host silently finds no skill. Replacing that is a
    repair rather than a destructive overwrite, so it does not require ``force``.
    """
    if os.path.isdir(target) and not os.path.islink(target):
        entries = os.listdir(source)
        return bool(entries) and all(
            _names_its_own_target(os.path.join(target, entry), os.path.join(source, entry)) for entry in entries
        )
    return _names_its_own_target(target, source)


def _existing_install(source: str, target: str) -> str | None:
    """What is already sitting at ``target``, in terms this installer can act on.

    ``links`` and ``copy`` both carry exactly the source's content, so switching between
    them is a relayout rather than a loss. Anything else is somebody's own work.
    """
    if not os.path.lexists(target):
        return None
    if _is_symlink_farm(source, target):
        return "links"
    if os.path.islink(target) and os.path.realpath(target) == os.path.realpath(source):
        return "whole_dir_link"  # an older install, or a hand-made shim
    if _is_dead_shim(target, source):
        return "dead"
    if os.path.isdir(target) and not os.path.islink(target) and _trees_match(source, target):
        return "copy"
    return "foreign"


def _install_action(source: str, target: str, *, mode: str, force: bool) -> str:
    """What installing ``source`` at ``target`` would do, without doing any of it."""
    found = _existing_install(source, target)
    if found is None:
        return "created"
    if found == "dead":
        return "repaired"
    if found == "foreign":
        # A hand-written skill, or an install that has since been edited. Overwriting it
        # would delete work the caller never mentioned.
        return "replaced" if force else "conflict"
    already_right = ("links" if mode == "symlink" else "copy") == found
    return "unchanged" if already_right else "replaced"


def _lay_out_skill(source: str, target: str, *, mode: str) -> None:
    """Write the skill at ``target``, replacing whatever is there.

    A symlink install links each top-level entry into a real directory rather than linking
    the directory itself. Both shapes resolve identically when read, but a walker that does
    not follow symlinks -- a common default, and what a ripgrep-backed file search does --
    lists nothing at all inside a symlinked directory, so the skill is silently undiscovered.
    A symlinked file inside a real directory is an ordinary entry to such a walk.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.lexists(target):
        if os.path.islink(target) or os.path.isfile(target):
            os.unlink(target)
        else:
            shutil.rmtree(target)
    if mode != "symlink":
        shutil.copytree(source, target)
        return
    os.makedirs(target)
    for entry in os.listdir(source):
        entry_source = os.path.join(source, entry)
        # Absolute: an installed source sits in site-packages, arbitrarily far from the
        # destination, so a relative link would be long and break if either side moved.
        os.symlink(
            os.path.abspath(entry_source),
            os.path.join(target, entry),
            target_is_directory=os.path.isdir(entry_source),
        )


def _install_one_skill(source: str, target: str, *, mode: str, force: bool, dry_run: bool) -> dict[str, Any]:
    """Place one skill at ``target``, refusing to destroy anything that is not ours."""
    action = _install_action(source, target, mode=mode, force=force)
    entry: dict[str, Any] = {"target": target, "mode": mode, "action": action}
    if action == "conflict":
        entry["reason"] = (
            "already exists with different content; pass force to replace it, or install under a different scope"
        )
        return entry
    if action == "unchanged" or dry_run:
        return entry

    try:
        _lay_out_skill(source, target, mode=mode)
    except OSError as exc:
        entry["action"] = "error"
        entry["reason"] = str(exc)
        if mode == "symlink":
            # The usual cause is Windows without developer mode or admin rights.
            entry["hint"] = "symlinks may be unavailable on this platform; retry with copy mode"
    return entry


def _install_request_error(scope: str, host: str, mode: str, dest: str | None, skills: list[str] | None) -> str | None:
    """The first thing wrong with an install request, or ``None`` if it is coherent."""
    if scope not in {"project", "user"}:
        return f"scope must be 'project' or 'user', got {scope!r}"
    if mode not in {"copy", "symlink"}:
        return f"mode must be 'copy' or 'symlink', got {mode!r}"
    unknown = [h for h in ([] if host == "all" else [host]) if h not in _SKILL_HOST_DIRS]
    if unknown:
        return f"unknown host(s) {unknown!r}; choose from {sorted(_SKILL_HOST_DIRS)} or 'all'"
    if dest and scope == "user":
        return "dest applies to project scope; user scope installs under the home directory"
    packaged = available_skills()
    missing = [name for name in skills or [] if name not in packaged]
    if missing:
        return f"no packaged skill named {missing!r}; available: {packaged}"
    return None


def install_skill(  # noqa: PLR0913 - one verb covering scope, host, mode and the safety flags
    *,
    scope: str = "project",
    host: str = "all",
    mode: str = "copy",
    skills: list[str] | None = None,
    dest: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install the packaged audio skills into the directories the hosts discover.

    The instructions live in the wheel, but no host looks inside site-packages: Codex and
    Cursor read ``.agents/skills``, Claude Code reads ``.claude/skills``. A checkout has
    those wired up already; this is how a ``pip install`` user gets the same workflow.

    ``scope`` is ``project`` (the CWD, or ``dest``) or ``user`` (the home-directory
    equivalents). ``mode`` defaults to ``copy`` because it works everywhere, including a
    Windows checkout without symlink support; ``symlink`` keeps an installed skill in step
    with the package and is what a developer wants. Idempotent, and refuses to replace a
    directory whose content differs unless ``force`` is set.
    """
    bad_request = _install_request_error(scope, host, mode, dest, skills)
    if bad_request:
        return {"status": "error", "error": bad_request}

    source_root = skills_dir()
    wanted = skills or available_skills()
    hosts = sorted(_SKILL_HOST_DIRS) if host == "all" else [host]

    # One directory may serve several hosts (Codex and Cursor share ``.agents/skills``),
    # so collapse them first: installing twice would report the second pass as unchanged
    # and make the summary read as if more was written than actually was.
    roots: dict[str, list[str]] = {}
    for name in hosts:
        project_dir, user_dir = _SKILL_HOST_DIRS[name]
        if scope == "project":
            root = os.path.abspath(os.path.join(dest or os.getcwd(), project_dir))
        else:
            root = os.path.abspath(os.path.expanduser(user_dir))
        roots.setdefault(root, []).append(name)

    targets = [os.path.join(root, name) for root in roots for name in wanted]
    pviol = _safety.path_violations(targets)
    if pviol:
        return {
            "status": "refused",
            "reason": "target path(s) resolve outside the allowed workspace",
            "violations": pviol,
            "hint": (
                "AUDIO_AGENT_WORKSPACE confines writes to that tree; install into the "
                "workspace with project scope, or unset the lock to install for the user"
            ),
        }

    installed: list[dict[str, Any]] = []
    for root, host_names in roots.items():
        for name in wanted:
            entry = _install_one_skill(
                os.path.join(source_root, name),
                os.path.join(root, name),
                mode=mode,
                force=force,
                dry_run=dry_run,
            )
            installed.append({"skill": name, "hosts": host_names, **entry})

    failed = [e for e in installed if e["action"] in {"conflict", "error"}]
    return {
        "status": "error" if failed else "ok",
        "scope": scope,
        "mode": mode,
        "dry_run": dry_run,
        "source": source_root,
        "installed": installed,
        **({"unresolved": failed} if failed else {}),
        "next_step": (
            "restart the host (or reload its skills) so it rescans the directory"
            if not dry_run and not failed
            else "nothing was written"
            if dry_run
            else "resolve the conflicts above, or re-run with force"
        ),
    }
