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

"""Content-addressed artifacts — the unit of execution reuse.

Instead of asking "was this whole recipe run before?", ask "has **this step**,
with these semantics and the same resolved dataset identity at its recorded
tier, already produced an artifact?". Identity is a Merkle chain over the
pipeline::

    step_key(0) = H(dataset_key,   ref, semantic_params, impl_version, model_version)
    step_key(i) = H(step_key(i-1), ref, semantic_params, impl_version, model_version)

One mechanism then gives whole-pipeline, prefix and single-stage reuse, and invalidation
falls out of the chain instead of needing hand-written rules: change a param at step *i* and
keys ``i..n`` change while ``0..i-1`` stay reusable. ``impl_version`` is per stage (see
:mod:`nemo_curator.audio_agent.code_identity`), so editing one stage invalidates it and
everything chained below it, and editing unrelated code invalidates nothing.

Only steps that PERSIST a resumable result get an artifact -- you can only
resume from disk. Source acquisition caches such as ``raw_data_dir`` are
deliberately excluded: replaying FLEURS/ReadSpeech through a generic audio-folder
source would lose transcript/split semantics. An artifact counts as reusable
only once an atomic ``_COMPLETE`` marker names its ``step_key`` (a crashed run
leaves a partial, appended JSONL that would otherwise look perfectly valid).

See ``REUSE_ARCHITECTURE.md``. This is deterministic memoization, not learning:
nothing here influences *what* the agent plans, only whether work matching the
recorded computation and dataset identities is redone.
"""

from __future__ import annotations

import calendar
import contextlib
import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.audio_agent.code_identity import impl_version

if TYPE_CHECKING:
    from nemo_curator.audio_agent.recipe import Recipe, StageRef

# Bump when the key construction changes, so old artifacts can never be matched by a
# differently-computed key. v2: per-stage ``impl_version`` replaced the repository-wide
# package version, so every v1 key describes a different computation than its digits suggest.
# v3: ``retention_sec`` and ``owner`` left the semantic params, so every v2 key for a
# checkpointed recipe was computed over a policy that does not change the bytes. Old records
# stay on disk and are simply never matched again -- the failure direction is recompute.
STEP_KEY_VERSION = "v3"

ArtifactKind = Literal["manifest", "audio_dir", "rttm_dir", "text_dir", "archive", "unknown"]

# Params naming the model a stage runs; recorded so a model swap is visible on the approval
# card. (They are ordinary semantic params, so they already move the step key.)
_MODEL_PARAMS = ("model_name", "model", "model_path", "pretrained_model", "hf_model", "checkpoint", "model_id")
# Preference order when a stage names more than one output location.
_URI_PREFERENCE = (
    "output_manifest",
    "output_path",
    "output_dir",
    "resampled_audio_dir",
    "separated_audio_dir",
    "rttm_out_dir",
    "output_audio_tar_path",
)
_DOWNLOAD_TTL_SEC = 7 * 24 * 3600  # a re-download can legitimately differ; don't trust it forever
# Card tags meaning "this stage goes to the network". Paired with an ``ingest`` category, that is
# what earns a freshness window; on any other category the download is a model, not the data.
_REMOTE_FETCH_TAGS = frozenset({"needs_internet_first_run", "downloads_dataset", "needs_internet"})
_MARKER = "_COMPLETE"
_CONTENT_DIGEST_VERSION = "sha256-v1"
# What a path's own extension says it is. An archive is deliberately its own kind: nothing can
# re-read one as a pipeline source, and saying so refuses the continuation instead of handing a
# tarball to a stage that expects a folder of audio.
_EXT_KIND: tuple[tuple[tuple[str, ...], ArtifactKind], ...] = (
    ((".jsonl", ".json"), "manifest"),
    ((".txt", ".csv"), "text_dir"),
    ((".rttm",), "rttm_dir"),
    ((".tar", ".tar.gz", ".tgz", ".zip"), "archive"),
    ((".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"), "audio_dir"),
)
_KIND_SAMPLE = 200  # entries sampled when classifying a directory; enough to be sure, cheap to read


@dataclass
class Artifact:
    """One completed step's output: what it is, what it cost, and how far to trust it."""

    # identity
    step_key: str
    input_key: str = ""
    stage_ref: str = ""
    stage_index: int = 0
    semantic_params: dict[str, Any] = field(default_factory=dict)
    contract_hash: str | None = None
    # location
    uri: str = ""
    kind: ArtifactKind = "unknown"
    # evidence
    rows_in: int = 0
    rows_out: int = 0
    bytes: int = 0
    # Full serialized-output digest. The dataset/step keys prove what SHOULD
    # have produced the artifact; this proves the bytes still present are the
    # bytes that were published.
    content_digest: str = ""
    produced_roles: list[str] = field(default_factory=list)
    produced_keys: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    # cost
    started_at: str = ""
    ended_at: str = ""
    duration_sec: float = 0.0
    # Time to get from the SOURCE to here (this step plus every step feeding it). This, not
    # duration_sec, is what reusing the artifact actually saves: a pipeline that only persists
    # at its final writer would otherwise report the writer's milliseconds and quietly skip
    # asking before serving an hour-old ASR result.
    cumulative_sec: float = 0.0
    gpu_seconds: float = 0.0
    device: str = ""
    # trust
    dataset_key: str = ""
    fingerprint_tier: str = ""
    # How many input files this artifact's inventory covers, ``0`` when none was recorded. The
    # inventory itself lives beside the record (``save_coverage``); this is here so a scan can
    # tell whether a delta is even possible without opening the sidecar.
    covers_files: int = 0
    # What produced this: ``impl_version`` is the digest of the stage's own source and is what
    # reuse is checked against. ``code_version`` is the package build, kept because a human
    # reading a record wants to know which Curator wrote it -- it decides nothing.
    impl_version: str = ""
    code_version: str = ""
    model_version: str = ""
    deterministic: bool = True
    ttl_sec: int = 0
    status: str = "complete"
    # provenance
    run_id: str = ""
    created_at: str = ""
    # What produced this, kept so the answer survives the run record being pruned. Provenance
    # only: reuse matches on ``step_key``, never on these. Binding reuse to the config hash
    # would defeat the checkpoint's whole purpose -- retuning a downstream threshold changes
    # that hash, and the checkpoint above it is exactly what the retuned run wants to reuse.
    origin_config_hash: str = ""
    origin_recipe_uri: str = ""
    # Which workspace produced it. Empty on records written before this field existed, which
    # is why every check below is guarded rather than fail-closed.
    workspace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# --------------------------------------------------------------------------- step keys
def _semantic_blob(ref: str, params: dict[str, Any]) -> str:
    return json.dumps({"ref": ref, "params": params}, sort_keys=True, ensure_ascii=False, default=str)


def _h(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def code_version() -> str:
    """The Curator build an artifact was written by -- provenance only.

    Reuse is decided by :func:`~nemo_curator.audio_agent.code_identity.impl_version`. This
    string ends in the repository's git SHA, so using it as the test meant one commit
    anywhere emptied the store; it is recorded because a human reading a record still wants
    to know which build produced it.
    """
    try:
        import nemo_curator

        return str(getattr(nemo_curator, "__version__", "") or "unknown")
    except Exception:  # noqa: BLE001 - never fail key computation over a version lookup
        return "unknown"


def model_version(params: dict[str, Any]) -> str:
    """The model identifier a stage runs, if it names one ("" when it runs no model)."""
    for key in _MODEL_PARAMS:
        v = params.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _card(stage_ref: str) -> dict[str, Any]:
    try:
        from nemo_curator.audio_agent.index import get_index

        return get_index().card(stage_ref) or {}
    except Exception:  # noqa: BLE001 - knowledge is optional context, never a hard dependency
        return {}


def stage_trust(stage_ref: str) -> tuple[bool, int]:
    """``(deterministic, ttl_sec)`` for a stage, from its capability card.

    A card sets ``deterministic: false`` when the same inputs can legitimately produce different
    output -- randomized internal chunking, unseeded sampling, a decode whose ties break by
    thread order. That does not make a stored result *wrong*, so it does not make the artifact
    invalid; it means reusing it needs the user to say yes (:func:`caution_reasons`).

    A TTL says the same inputs could legitimately yield different bytes later, which is true of a
    stage that FETCHES ITS DATA from the internet and not of one that downloads a pinned model --
    a changed model shows up in ``model_version`` instead. So the window follows the card's
    category, not merely the presence of a download: this used to look for a tag containing
    "download" that no card carries, making the whole freshness window unreachable.
    """
    card = _card(stage_ref)
    declared = card.get("deterministic")
    ttl = int(card.get("artifact_ttl_sec") or 0)
    if not ttl and card.get("category") == "ingest" and _fetches_remotely(card):
        ttl = _DOWNLOAD_TTL_SEC
    return (True if declared is None else bool(declared)), ttl


def stage_is_costly(stage_ref: str) -> bool:
    """Whether this stage's runtime is too large to assume away when nobody measured it.

    Reuse under :data:`reuse.AUTO_REUSE_SEC` is taken without asking, which is only sound when
    the saving was actually *measured*. An unmeasured step contributes zero seconds, so a prefix
    nobody timed read exactly like a cheap one and an hour of transcription was served silently
    under a rule written for milliseconds. This decides which unmeasured steps still deserve the
    question, and it asks the card rather than a list of stage names, so a new GPU stage is
    covered the day its card lands.

    A card that says ``bound: cpu``/``io`` and names no model is taken at its word. A stage with
    no card at all is not something we can call cheap, so it counts.
    """
    card = _card(stage_ref)
    if not card:
        return True
    bound = str((card.get("resource") or {}).get("bound") or "").lower()
    if bound == "gpu":
        return True
    if card.get("model_id") or card.get("model_version"):
        return True
    if not bound:
        # The card exists but does not SAY ``cpu``/``io``, so there is nothing to take at its
        # word. That is the same state as having no card, which is refused as cheap two lines
        # up; reading an unstated bound as "not gpu" instead let a config-dependent composite
        # under the auto-take threshold. No shipped card relies on this -- it keeps the next
        # ``bound: null`` placeholder from quietly buying a pass.
        return True
    return _fetches_remotely(card)


def _fetches_remotely(card: dict[str, Any]) -> bool:
    """Whether the card says this stage reaches the network for its content."""
    return any(t in _REMOTE_FETCH_TAGS for t in (str(t) for t in (card.get("tags") or [])))


def output_uri(stage: StageRef) -> tuple[str, ArtifactKind]:
    """Where this stage persists its output, and what shape that output has.

    ``("", "unknown")`` when the stage writes nothing -- such a step has no artifact and can
    never be a reuse point, which is exactly right: reuse resumes from disk.
    """
    for key in _URI_PREFERENCE:
        v = stage.params.get(key)
        if isinstance(v, str) and v:
            return v, _kind_of(v)
    return "", "unknown"


def _kind_of(value: str) -> ArtifactKind:
    """What shape the output at ``value`` has -- decided by the output, not by a parameter's name.

    The kind chooses which source stage can re-read the artifact, so a wrong answer routes a
    continuation into a stage that cannot read it and "succeeds" over zero rows. The parameter name
    used to decide: ``output_audio_tar_path`` made a ``.tar`` an ``audio_dir`` because the word
    "audio" appears in it, and any ``*_dir`` param was audio regardless of what a stage put there.
    Hence no ``param`` argument here -- there is no name whose presence should change the answer.

    A path with a known extension answers for itself. A directory that exists is sampled. A
    directory that does not exist yet cannot be classified honestly, so it stays ``unknown`` and
    is re-classified at publish time, when there is something to look at.
    """
    low = value.lower()
    for exts, kind in _EXT_KIND:
        if low.endswith(exts):
            return kind
    path = os.path.expanduser(value)
    if os.path.isdir(path):
        return _kind_of_dir(path)
    return "unknown"


def classify_output(uri: str) -> ArtifactKind | None:
    """What the output at ``uri`` actually is, or ``None`` if it cannot be told from the bytes.

    For use once a stage has run, when there is something on disk to look at, so a plan-time guess
    about a not-yet-created directory does not become the recorded truth.
    """
    if not uri:
        return None
    kind = _kind_of(uri)
    return kind if kind != "unknown" else None


def _kind_of_dir(path: str) -> ArtifactKind:
    """Classify a directory by what is in it; ``unknown`` when its contents disagree."""
    found: set[ArtifactKind] = set()
    with contextlib.suppress(OSError):
        for entry in itertools.islice(os.scandir(path), _KIND_SAMPLE):
            name = entry.name.lower()
            if name == _MARKER:
                continue
            for exts, kind in _EXT_KIND:
                if name.endswith(exts):
                    found.add("text_dir" if kind == "manifest" else kind)
                    break
    return found.pop() if len(found) == 1 else "unknown"


@dataclass
class StepPlan:
    """A recipe step's reuse identity: its key, and the artifact it would publish."""

    index: int
    stage_ref: str
    step_key: str
    input_key: str
    semantic_params: dict[str, Any]
    uri: str
    kind: ArtifactKind
    deterministic: bool
    ttl_sec: int
    model_version: str
    impl_version: str = ""

    def persists(self) -> bool:
        return bool(self.uri)


def plan_steps(recipe: Recipe, dataset_key: str) -> list[StepPlan]:
    """Compute the Merkle step-key chain for a recipe over a given source dataset."""
    plans: list[StepPlan] = []
    prev = f"{STEP_KEY_VERSION}:{dataset_key}"
    for i, stage in enumerate(recipe.stages):
        sp = stage.semantic_params()
        mv = model_version(stage.params)
        iv = impl_version(stage.ref)
        key = _h(STEP_KEY_VERSION, prev, stage.ref, _semantic_blob(stage.ref, sp), iv, mv)
        det, ttl = stage_trust(stage.ref)
        uri, kind = output_uri(stage)
        plans.append(
            StepPlan(
                index=i,
                stage_ref=stage.ref,
                step_key=key,
                input_key=prev,
                semantic_params=sp,
                uri=uri,
                kind=kind,
                deterministic=det,
                ttl_sec=ttl,
                model_version=mv,
                impl_version=iv,
            )
        )
        prev = key
    return plans


def step_keys(recipe: Recipe, dataset_key: str) -> list[str]:
    """Just the step-key chain (see :func:`plan_steps` for the full per-step detail)."""
    return [p.step_key for p in plan_steps(recipe, dataset_key)]


# --------------------------------------------------------------------------- storage
def artifacts_dir() -> str:
    """Where artifact records live (beside the run records, under the runs dir)."""
    from nemo_curator.audio_agent.run_store import runs_dir

    return os.path.join(runs_dir(), "artifacts")


def _record_path(step_key: str) -> str:
    return os.path.join(artifacts_dir(), f"{step_key}.json")


def save(artifact: Artifact) -> str:
    """Persist an artifact record as JSON and index it. Returns the record path.

    Written owner-only for the same reason as a run record: it names dataset keys and
    output locations, and agent state should not be readable by every other account on a
    shared machine. (The ``_COMPLETE`` marker beside the user's OUTPUT is deliberately
    left alone -- that directory belongs to the user, not to the agent.)
    """
    from nemo_curator.audio_agent.run_store import _ensure_private_dir, _write_private_json

    directory = artifacts_dir()
    _ensure_private_dir(directory)
    path = _record_path(artifact.step_key)
    _write_private_json(path, artifact.to_dict())
    with contextlib.suppress(Exception):  # the index is a rebuildable cache; JSON is the truth
        from nemo_curator.audio_agent import run_index

        run_index.index_artifact(artifact)
    return path


def coverage_path(step_key: str) -> str:
    return os.path.join(artifacts_dir(), "coverage", f"{step_key}.json")


def save_coverage(step_key: str, inventory: dict[str, str]) -> str:
    """Persist which input files an artifact covers, as ``{relpath: identity token}``.

    Beside the record rather than inside it: a reuse scan loads one record per step to compare
    keys, and a corpus-sized dict on each would make the cheap probe expensive. Only a delta
    decision reads this, and only for the one artifact it is about to resume from.
    """
    from nemo_curator.audio_agent.run_store import _ensure_private_dir, _write_private_json

    directory = os.path.join(artifacts_dir(), "coverage")
    _ensure_private_dir(directory)
    path = coverage_path(step_key)
    _write_private_json(path, {"step_key": step_key, "files": inventory})
    return path


def load_coverage(step_key: str) -> dict[str, str] | None:
    """The inventory an artifact covers, or ``None`` when it was never recorded.

    ``None`` and ``{}`` differ and the distinction decides a delta: nothing recorded means the
    comparison cannot be made, while an empty corpus is a fact about the data.
    """
    path = coverage_path(step_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            files = json.load(f).get("files")
    except Exception:  # noqa: BLE001 - a corrupt sidecar means no delta, never a crash
        return None
    return files if isinstance(files, dict) else None


def load(step_key: str) -> Artifact | None:
    """Load an artifact record by step key (None if absent or corrupt)."""
    path = _record_path(step_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return Artifact.from_dict(json.load(f))
    except Exception:  # noqa: BLE001 - a corrupt record must not break the caller
        return None


def list_artifacts() -> list[Artifact]:
    """Every stored artifact record (newest first)."""
    directory = artifacts_dir()
    if not os.path.isdir(directory):
        return []
    out: list[Artifact] = []
    for fn in sorted(os.listdir(directory), reverse=True):
        if fn.endswith(".json"):
            art = load(fn[: -len(".json")])
            if art is not None:
                out.append(art)
    return sorted(out, key=lambda a: a.created_at, reverse=True)


# --------------------------------------------------------------------------- atomic publish
def marker_path(uri: str) -> str:
    """``<uri>/_COMPLETE`` for a directory, ``<uri>._COMPLETE`` for a file."""
    expanded = os.path.expanduser(uri)
    return os.path.join(expanded, _MARKER) if os.path.isdir(expanded) else f"{expanded}.{_MARKER}"


def write_marker(
    uri: str,
    *,
    step_key: str,
    rows: int = 0,
    size: int = 0,
    content_digest: str = "",
) -> str | None:
    """Mark an output complete. Written LAST, so a crashed run leaves no marker and its
    partial output can never be mistaken for a reusable artifact."""
    path = marker_path(uri)
    payload = {
        "step_key": step_key,
        "rows": rows,
        "bytes": size,
        "content_digest": content_digest,
        "content_digest_version": _CONTENT_DIGEST_VERSION,
        "completed_at": _now(),
        "version": STEP_KEY_VERSION,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        return None
    return path


def read_marker(uri: str) -> dict[str, Any] | None:
    """The completion marker beside ``uri``, or None when the output is not complete."""
    path = marker_path(uri)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001 - an unreadable marker means "not proven complete"
        return None


def publish(artifact: Artifact) -> Artifact:
    """Atomically publish a completed step: measure it, mark it complete, then register it.

    Order matters. The marker is what makes the output reusable, and it is written before the
    record so a registry entry never points at an unmarked (possibly partial) output.
    """
    rows, size = measure(artifact.uri, artifact.kind)
    digest = content_digest(artifact.uri)
    artifact.rows_out = artifact.rows_out or rows
    artifact.bytes = size
    artifact.content_digest = digest or ""
    artifact.created_at = artifact.created_at or _now()
    marker = write_marker(
        artifact.uri,
        step_key=artifact.step_key,
        rows=artifact.rows_out,
        size=artifact.bytes,
        content_digest=artifact.content_digest,
    )
    if marker is None:
        msg = f"artifact publication failed: could not write the completion marker for {artifact.uri!r}"
        raise OSError(msg)
    save(artifact)
    return artifact


def content_digest(uri: str) -> str | None:
    """Hash the complete serialized output currently present at ``uri``.

    Relative paths, not the artifact's absolute location, enter a directory
    digest. Agent completion markers are excluded because they attest to the
    payload and are not payload themselves. ``None`` fails reuse closed.
    """
    expanded = os.path.expanduser(uri or "")
    if not expanded or not os.path.exists(expanded):
        return None
    digest = hashlib.sha256()
    try:
        if os.path.isfile(expanded):
            digest.update(b"file\0")
            _hash_file(expanded, digest)
        elif os.path.isdir(expanded):
            digest.update(b"directory\0")
            for root, dirs, files in os.walk(expanded):
                dirs.sort()
                rel_root = os.path.relpath(root, expanded)
                digest.update(f"dir\0{rel_root}".encode())
                digest.update(b"\0")
                for name in sorted(files):
                    if name == _MARKER or name.endswith(f".{_MARKER}"):
                        continue
                    full = os.path.join(root, name)
                    relative = os.path.relpath(full, expanded)
                    digest.update(f"file\0{relative}".encode())
                    digest.update(b"\0")
                    _hash_file(full, digest)
        else:
            return None
    except OSError:
        return None
    return f"sha256:{digest.hexdigest()}"


def _hash_file(path: str, digest: Any) -> None:  # noqa: ANN401 - hashlib protocol is private
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)


def measure(uri: str, kind: ArtifactKind = "unknown") -> tuple[int, int]:
    """``(rows, bytes)`` at a location: JSONL lines for a manifest, file count for a dir."""
    expanded = os.path.expanduser(uri or "")
    if not expanded or not os.path.exists(expanded):
        return 0, 0
    if os.path.isfile(expanded):
        size = _size(expanded)
        if expanded.endswith((".jsonl", ".json")):
            return _count_lines(expanded), size
        return 1, size
    rows = 0
    size = 0
    for root, _dirs, files in os.walk(expanded):
        for fn in files:
            if fn == _MARKER:
                continue
            full = os.path.join(root, fn)
            size += _size(full)
            rows += _count_lines(full) if fn.endswith((".jsonl", ".json")) and kind == "manifest" else 1
    return rows, size


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _count_lines(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- validity
def invalid_reasons(  # noqa: C901, PLR0912
    artifact: Artifact,
    *,
    dataset_key: str | None = None,
    require_high_trust: bool = False,
) -> list[str]:
    """Why this artifact must NOT be reused (empty list = safe to reuse).

    Every condition here is a way a reused result could silently differ from a fresh run, so the
    check is deliberately conservative: an unknown fails closed.

    What is NOT here is anything that merely wants a human's agreement -- see
    :func:`caution_reasons`. Those used to sit in this list, which meant a stage declaring itself
    non-deterministic did not become a warned-about candidate, it *vanished*: the user was never
    told prior work existed, and the whole "pre-select fresh" path around it was unreachable.
    Pass ``require_high_trust`` to fold the cautions back in and refuse outright.
    """
    reasons: list[str] = []
    if artifact.status != "complete":
        reasons.append(f"artifact status is {artifact.status!r}, not 'complete'")
    from nemo_curator.audio_agent import _safety

    path_issues = _safety.path_violations([artifact.uri])
    if path_issues:
        reasons.append("artifact output is outside the allowed workspace: " + "; ".join(path_issues))
    elif not artifact.uri or not os.path.exists(os.path.expanduser(artifact.uri)):
        reasons.append(f"output no longer exists at {artifact.uri!r}")
    else:
        marker = read_marker(artifact.uri)
        if marker is None:
            reasons.append("no _COMPLETE marker; the output may be partial (a crashed run leaves one behind)")
        elif marker.get("step_key") != artifact.step_key:
            reasons.append("the _COMPLETE marker belongs to a different step; the location was overwritten")
        elif not artifact.content_digest or not marker.get("content_digest"):
            reasons.append("artifact predates serialized-content binding; republish it before reuse")
        elif marker.get("content_digest") != artifact.content_digest:
            reasons.append("the _COMPLETE marker and artifact record bind different serialized content")
        else:
            current_digest = content_digest(artifact.uri)
            if current_digest is None:
                reasons.append("serialized output could not be hashed; its content is unverified")
            elif current_digest != artifact.content_digest:
                reasons.append("serialized output changed after the artifact was published")
    reasons.extend(_foreign_workspace_reasons(artifact))
    if dataset_key and artifact.dataset_key and dataset_key != artifact.dataset_key:
        reasons.append("source data changed since this artifact was produced")
    reasons.extend(_impl_reasons(artifact))
    if artifact.ttl_sec and _age_sec(artifact) > artifact.ttl_sec:
        reasons.append(f"older than its {artifact.ttl_sec}s freshness window (re-fetch may differ)")
    if require_high_trust:
        reasons.extend(caution_reasons(artifact))
    return reasons


def _impl_reasons(artifact: Artifact) -> list[str]:
    """Why this artifact's code stamp does not match the code here, or ``[]``.

    Two stamps differ for two very different reasons, and saying "the implementation changed"
    for both is how an environment problem gets read as a code edit. A fallback stamp on
    either side means the closure could not be read -- by the run that published this, or by
    this process -- so the stage's identity is simply unproven. Both refuse reuse; only one
    of them is a fact about the source.
    """
    from nemo_curator.audio_agent.code_identity import is_fallback

    if not artifact.impl_version:
        return []
    current = impl_version(artifact.stage_ref)
    if artifact.impl_version == current:
        return []
    stage = artifact.stage_ref or "the stage"
    if is_fallback(current):
        return [
            f"{stage}'s source cannot be read here, so its implementation is unproven "
            "(this is an environment problem, not a code change); reuse is refused rather "
            "than risked"
        ]
    if is_fallback(artifact.impl_version):
        return [
            f"{stage} was produced by a run that could not read its own source, so the code "
            "behind this artifact is unproven; reuse is refused rather than risked"
        ]
    return [f"{stage}'s implementation changed since this artifact was produced"]


def _foreign_workspace_reasons(artifact: Artifact) -> list[str]:
    """Why this artifact belongs to a different workspace, or ``[]``.

    Guarded at both ends. An artifact written before workspace ids existed carries none, and
    a workspace whose own id cannot be read has nothing to compare against; in either case
    containment is simply unproven, and refusing every artifact over a missing id would
    break reuse for a property that used to be implicit in the directory layout.
    """
    if not artifact.workspace_id:
        return []
    from nemo_curator.audio_agent.run_store import workspace_id

    current = workspace_id()
    if not current or current == artifact.workspace_id:
        return []
    return ["produced in a different workspace; local work is not shared across workspaces"]


def caution_reasons(artifact: Artifact) -> list[str]:
    """Why reusing this needs an explicit yes -- true of the artifact, but not disqualifying.

    The difference from :func:`invalid_reasons` is who gets to decide. An artifact whose
    ``_COMPLETE`` marker is missing is not a result and no one may serve it. An artifact from a
    stage that does not reproduce exactly IS a result -- just not one we can promise a rerun
    would match -- so the honest move is to show it, say why it is weaker, and let the person
    who wants the hour back choose. Recomputing behind their back is the default, not the rule.
    """
    reasons: list[str] = []
    if not artifact.deterministic:
        reasons.append(f"{artifact.stage_ref} is declared non-deterministic; a rerun may differ")
    if artifact.fingerprint_tier != "stat":
        reasons.append("source identified only by its shape, so a file edited in place would be invisible")
    return reasons


def _age_sec(artifact: Artifact) -> float:
    """How long ago the artifact was produced, in seconds.

    Underscored but NOT private: the ``checkpoints`` verb asks the same expiry question this
    module does, and re-deriving it there would duplicate the UTC subtlety below.

    ``created_at`` is written in UTC (``_now`` uses ``gmtime``), so it must be read as UTC.
    ``time.mktime`` interprets its argument as LOCAL time, which shifted every age by the machine's
    offset -- harmless while no TTL was reachable, and a wrong expiry the moment one is.
    """
    try:
        created = calendar.timegm(time.strptime(artifact.created_at, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, time.time() - created)


def lookup(step_key: str, *, dataset_key: str | None = None) -> tuple[Artifact | None, list[str]]:
    """Find a reusable artifact for a step key: ``(artifact, reasons_it_is_unusable)``.

    An artifact is returned even when it is unusable, so callers can EXPLAIN the miss ("we
    found prior work but the data changed") rather than silently rerunning.
    """
    art = load(step_key)
    if art is None:
        return None, ["no prior artifact for this step"]
    return art, invalid_reasons(art, dataset_key=dataset_key)
