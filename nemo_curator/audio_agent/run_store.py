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

"""Local run-record store — per-run provenance for tracing + incremental continuation.

**Local history only, NOT shared memory / cross-user learning** (a permanent non-goal).
Records support *deterministic memoization* — content-addressed "has this exact computation
already been done?" (see ``REUSE_ARCHITECTURE.md``) — and provenance; they are never fed back
as learned priors to influence *what* the agent plans.

One JSON per run under a runs directory: ``AUDIO_AGENT_RUNS_DIR`` if set, else
``<AUDIO_AGENT_WORKSPACE>/.audio_agent_runs``, else ``<cwd>/.audio_agent_runs``. Records are
written by ``run`` and read by the ``runs`` / ``continue`` / ``reuse-scan`` verbs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from typing import Any

from nemo_curator.audio_agent.contracts import RunRecord

# A run id becomes a filename and reaches ``load`` from the CLI and MCP, so unvalidated a
# ``../`` would read any ``.json`` on the box back through the record fields. Deliberately
# looser than ``calibration_store``'s ``[A-Za-z0-9_-]``: a run id carries a microsecond
# timestamp, so that pattern would reject every real id. Dots are admitted, separators are not;
# ``.`` and ``..`` alone pass and are harmless, naming a file inside the runs directory.
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9_.-]{1,128}\Z")
# A step key becomes a filename too, but it is a hex digest rather than a timestamp, so it
# can be held to the stricter shape ``_SAFE_RUN_ID`` cannot use.
_SAFE_STEP_KEY = re.compile(r"\A[A-Za-z0-9]{1,64}\Z")


def runs_dir() -> str:
    """The directory run records live in (env > workspace > cwd)."""
    explicit = os.environ.get("AUDIO_AGENT_RUNS_DIR")
    if explicit:
        return os.path.expanduser(explicit)
    from nemo_curator.audio_agent._safety import workspace_root

    root = workspace_root() or os.getcwd()
    return os.path.join(root, ".audio_agent_runs")


def _ensure_private_dir(path: str) -> None:
    """Create a state directory readable only by its owner.

    Run records carry dataset paths, dataset keys, goals and output locations. Created
    under a normal umask (002/022) they are group- and world-readable, which on a shared
    build agent, HPC project space or team NFS share exposes one user's curation history
    to every other. Only a directory this call actually CREATES is tightened -- one that
    already exists was configured deliberately and is left as the deployment set it.

    Underscored but NOT private: imported by ``artifacts`` and ``calibration_store``, which
    write their own state alongside the run records and need the same permissions. It lives
    here rather than in a utility module because the reasoning above is about run records.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        return
    except OSError:
        raise
    with contextlib.suppress(OSError):  # best-effort: a filesystem may not honour chmod
        os.chmod(path, 0o700)


def _write_private_json(path: str, payload: dict) -> None:
    """Write JSON to a file created owner-only (0600); pre-existing files keep their mode.

    Underscored but NOT private: imported by ``artifacts`` and ``calibration_store``, always
    paired with :func:`_ensure_private_dir`.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def scratch_dir() -> str:
    """Where to put a recipe written for one job, created on demand.

    A recipe assembled for a single request is working material, not a contribution, but it has
    to live in a file because every verb takes a path. With nowhere designated, it lands in
    whatever directory the caller happened to start in -- which for anyone working inside a
    checkout is the repository root. Five such files have accumulated there, each looking like
    an untracked change someone forgot to commit.

    Sits under :func:`runs_dir`, which is already git-ignored and already moves with
    ``AUDIO_AGENT_RUNS_DIR``, so a scratch recipe is discardable by the same gesture that
    discards the run records describing what it did.
    """
    path = os.path.join(runs_dir(), "recipes")
    _ensure_private_dir(runs_dir())
    _ensure_private_dir(path)
    return path


def checkpoints_dir() -> str:
    """Where managed reuse checkpoints live, created on demand.

    A checkpoint is recomputable cache, not a deliverable: it exists so that editing the
    tail of a pipeline does not re-run the expensive head. Asking the user where to put one
    made them name a file they have no reason to care about, and the name they were offered
    embedded a recipe hash that inserting the checkpoint had already changed.

    Sits beside the artifact records that index it, under :func:`_ensure_private_dir` for the
    same reason those are: a checkpoint holds per-file scores and source paths, and agent
    state should not be readable by every other account on a shared machine.
    """
    path = os.path.join(runs_dir(), "checkpoints")
    _ensure_private_dir(runs_dir())
    _ensure_private_dir(path)
    return path


def checkpoint_path(step_key: str) -> str | None:
    """The managed location for one step's checkpoint, or ``None`` for an unusable key.

    Named by step key because that is the identity reuse already matches on, and the one
    identifier that is knowable BEFORE the work runs -- a content digest is not, so it could
    never address a cache you want to consult before computing. It also survives the edit a
    recipe hash does not: retuning a downstream threshold leaves every step key above it
    untouched, so the checkpoint stays addressable by the run that wants to reuse it.
    """
    if not step_key or not _SAFE_STEP_KEY.match(str(step_key)):
        return None
    return os.path.join(checkpoints_dir(), f"{step_key}.jsonl")


def workspace_id() -> str:
    """This workspace's stable identity, minted once and kept beside the run records.

    Checkpoints and artifacts are LOCAL work: they name this machine's source paths, and
    "are these bytes trustworthy" is a question only the account that produced them can
    answer. Scoping was previously an accident of layout -- the index lives under
    :func:`runs_dir`, so another workspace simply never saw it -- which held right up until
    a record was copied between trees.

    A minted id rather than the directory path: a workspace that is moved or renamed is
    still the same workspace, and a path comparison would say otherwise. Returns ``""`` when
    the id cannot be read or written, which every caller treats as "cannot prove
    containment" and therefore skips the check rather than refusing everything.
    """
    path = os.path.join(runs_dir(), "workspace.json")
    try:
        with open(path, encoding="utf-8") as f:
            existing = str(json.load(f).get("workspace_id") or "")
        if existing:
            return existing
    except (OSError, ValueError):
        pass  # absent or corrupt: mint below
    minted = uuid.uuid4().hex
    try:
        _ensure_private_dir(runs_dir())
        # O_EXCL so two processes racing to mint cannot each believe they won; the loser
        # re-reads the winner's id rather than overwriting it.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            with open(path, encoding="utf-8") as f:
                return str(json.load(f).get("workspace_id") or "")
        except (OSError, ValueError):
            return ""
    except OSError:
        return ""
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"workspace_id": minted}, f)
    return minted


def origin_recipe_path(config_hash: str) -> str | None:
    """Where the recipe that produced a given ``config_hash`` is kept, keyed by that hash."""
    if not config_hash or not _SAFE_STEP_KEY.match(str(config_hash)):
        return None
    return os.path.join(runs_dir(), "origin_recipes", f"{config_hash}.json")


def save_origin_recipe(config_hash: str, recipe: dict[str, Any]) -> str | None:
    """Keep the recipe an artifact came from, addressed by its config hash. Best-effort.

    An artifact already records the ``run_id`` that wrote it, so the recipe is reachable in
    one hop -- until the run record is pruned, at which point the bytes outlive every
    description of what produced them. Keyed by config hash rather than run id because that
    is what the artifact carries and what a person asking "what made this?" has in hand.

    Distinct from :func:`save_exact_recipe`, which keeps the UNREDACTED copy of a single run
    for re-execution. This is provenance: it answers what a stored artifact came from, and
    it never decides reuse.
    """
    path = origin_recipe_path(config_hash)
    if path is None:
        return None
    if os.path.isfile(path):
        return path  # same hash, same recipe -- rewriting it would only risk losing it
    try:
        _ensure_private_dir(runs_dir())
        _ensure_private_dir(os.path.dirname(path))
        _write_private_json(path, dict(recipe))
    except OSError:
        return None
    return path


def load_origin_recipe(config_hash: str) -> dict[str, Any] | None:
    """The recipe stored for a config hash, or ``None``."""
    path = origin_recipe_path(config_hash)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def exact_recipe_path(run_id: str) -> str | None:
    """Where a run's verbatim recipe copy lives, or ``None`` for an id that cannot name a file."""
    if not run_id or not _SAFE_RUN_ID.match(str(run_id)):
        return None
    return os.path.join(runs_dir(), "exact_recipes", f"{run_id}.json")


def save_exact_recipe(run_id: str, recipe: dict[str, Any]) -> str | None:
    """Keep a verbatim copy of a recipe the run record cannot reproduce. Best-effort.

    The record's own copy has secret-valued params masked, which is right for a payload that
    reaches a host LLM and wrong for the one thing history is asked to do besides tracing:
    re-run this pipeline over what changed. A masked param is part of reuse identity, so a recipe
    rebuilt from the record hashes differently and matches none of that run's own prior work --
    the request "do the same thing again on the new files" fails on a pipeline that needs a
    credential.

    Written only when redaction actually changed something, so the ordinary run adds no second
    file, and only inside the owner-only state directory (0700/0600) the run records already use.
    Returns the path written, or ``None`` when there was nothing to keep.
    """
    from nemo_curator.audio_agent._safety import redact

    if redact(recipe, redact_transcripts=False) == recipe:
        return None  # the record reproduces it exactly; a second copy would be one more place to leak
    path = exact_recipe_path(run_id)
    if path is None:
        return None
    _ensure_private_dir(runs_dir())
    _ensure_private_dir(os.path.dirname(path))
    _write_private_json(path, dict(recipe))
    return path


def load_exact_recipe(run_id: str) -> dict[str, Any] | None:
    """The verbatim recipe for a run, or ``None`` when the record's own copy is already exact."""
    path = exact_recipe_path(run_id)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:  # noqa: BLE001 - a corrupt copy falls back to the record, it does not raise
        return None
    return loaded if isinstance(loaded, dict) else None


def new_run_id(config_hash: str | None = None) -> str:
    """A sortable, collision-resistant run id: ``run-<UTC ts.microseconds>Z-<hash8>-<rand4>``.

    Microsecond precision plus a short random tiebreak keeps ids unique even when two
    identical-config runs start in the same wall-clock second -- a plain second-resolution
    id (with the same config_hash) would collide and silently overwrite the earlier record.
    The zero-padded fixed-width timestamp keeps ids lexicographically time-sortable.
    """
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f".{int((now % 1) * 1_000_000):06d}Z"
    rand = os.urandom(2).hex()  # 4 hex chars: tiebreak within the same microsecond
    suffix = (config_hash or "")[:8]
    return f"run-{ts}-{suffix}-{rand}" if suffix else f"run-{ts}-{rand}"


def record_path(run_id: str) -> str | None:
    """The file a run record occupies, or ``None`` when the id cannot safely name one.

    One place builds this path, so ``load`` and ``save`` cannot disagree about which ids are
    allowed to steer it.
    """
    return os.path.join(runs_dir(), f"{run_id}.json") if run_id and _SAFE_RUN_ID.match(str(run_id)) else None


def save(record: RunRecord) -> str:
    """Persist a run record as JSON and index it; returns the path.

    The JSON is the source of truth; the SQLite index is a rebuildable cache, so a failure
    to index is swallowed rather than losing the record.

    Raises rather than silently relocating a record whose id could steer the path. Nothing
    reaches here with a caller-chosen id today -- ``verbs._record_run`` passes what
    ``new_run_id`` produced -- so this guards the invariant rather than a live route.
    """
    directory = runs_dir()
    path = record_path(record.run_id)
    if path is None:
        msg = (
            f"run_id {record.run_id!r} cannot name a record file; expected the shape "
            f"new_run_id() produces (letters, digits, '.', '_', '-')"
        )
        raise ValueError(msg)
    _ensure_private_dir(directory)
    _write_private_json(path, record.to_dict())
    with contextlib.suppress(Exception):  # the index is a cache; never fail a save over it
        from nemo_curator.audio_agent import run_index

        run_index.index_run(record)
    return path


def load(run_id: str) -> RunRecord | None:
    """Load a run record by id, or None if the id is unusable / it doesn't exist / can't parse.

    An id that could steer the path out of the runs directory reads as "no such record", which
    is what it is: the store holds records under ids it issued, and nothing else is one.
    """
    path = record_path(run_id)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return RunRecord.from_dict(json.load(f))
    except Exception:  # noqa: BLE001 - a corrupt record must not break the caller
        return None


def list_runs() -> list[dict[str, Any]]:
    """Summaries of all stored run records (most recent first)."""
    directory = runs_dir()
    if not os.path.isdir(directory):
        return []
    out: list[dict[str, Any]] = []
    for fn in sorted(os.listdir(directory), reverse=True):
        if not fn.endswith(".json"):
            continue
        rec = load(fn[: -len(".json")])
        if rec is None:
            continue
        out.append(
            {
                "run_id": rec.run_id,
                "config_hash": rec.config_hash,
                "semantic_hash": rec.semantic_hash,
                "dataset_key": rec.dataset_key,
                "parent_run_id": rec.parent_run_id,
                "status": rec.status,
                "accepted": rec.accepted,
                "input_count": rec.input_count,
                "data_source": rec.data_source,
                "elapsed_sec": rec.elapsed_sec,
                "created_at": rec.created_at,
            }
        )
    return out
