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

"""Deterministic safety guardrails enforced INSIDE the verbs (not just the skill).

A weaker host, or a direct verb/CLI call, cannot bypass these:

* Workspace path lock  - file paths must resolve under an allowed root (opt-in via
  ``AUDIO_AGENT_WORKSPACE``); blocks traversal / reads-writes outside it. It governs the
  dataset and the outputs. Shared dependency locations (``cache_dir``, ``config_path``) are
  exempt, so a locked deployment can still share one model cache across projects.
* Secrets/transcript redaction - secret-looking keys and transcript text are
  stripped from verb return values before they reach the host LLM.
* Require-smoke evidence - a ``run`` can be made to refuse unless handed a valid
  ``smoke_token`` (opt-in via ``AUDIO_AGENT_REQUIRE_SMOKE``), proving a smoke ran
  for this exact recipe.

Semantic misuse refusal (e.g. "isolate a named person's voice") is NOT here: it
needs judgment a deterministic tool can't make, so it stays a skill/policy concern.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import itertools
import json
import os
import re
import secrets
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

_SMOKE_SECRET_ENV = "AUDIO_AGENT_SMOKE_SECRET"  # noqa: S105
# Same variable ``run_store`` reads; the smoke secret prefers the agent's own state dir
# so a deployment that already redirects run records keeps its secret beside them.
_RUNS_DIR_ENV = "AUDIO_AGENT_RUNS_DIR"
_WORKSPACE_ENV = "AUDIO_AGENT_WORKSPACE"
_LOCAL_URI_SCHEMES = frozenset({"file", "local"})

# Key names that hold a secret (redacted from returns). Matched on the key's WORDS, not as
# bare substrings: a substring match reads ``tokenizer_path`` -- a real, card-advertised
# parameter -- as a credential and destroys it, and a destroyed value is then persisted into
# run records and compared during reuse, so the corruption outlives the display.
_SECRET_WORDS = frozenset(
    {
        "token",
        "tokens",
        "secret",
        "secrets",
        "password",
        "passwords",
        "passwd",
        "pwd",
        "apikey",
        "credential",
        "credentials",
    }
)
# Secret names that span a separator, so they survive the word split as adjacent pairs
# (``api_key`` -> ``api`` + ``key``). ``key`` alone is deliberately NOT a secret word: it
# ends most semantic field names in this codebase (``audio_filepath_key``, ``score_key``).
_SECRET_WORD_PAIRS = frozenset(
    {("api", "key"), ("access", "key"), ("secret", "key"), ("private", "key"), ("auth", "key")}
)
# The same pairs written without a separator. ``secretkey`` has no boundary of any kind to
# split on, so the pair rule cannot see it and it would otherwise read as one unknown word.
_SECRET_GLUED_PAIRS = frozenset(first + second for first, second in _SECRET_WORD_PAIRS)
_KEY_WORD_SPLIT = re.compile(r"[^a-z0-9]+")
# Field names the agent itself issues, which the word rule above would otherwise destroy.
# ``smoke_token`` is evidence a smoke ran for this recipe, not a credential: redacting it left
# ``AUDIO_AGENT_REQUIRE_SMOKE`` with no value the caller could pass back to ``run``, so the gate
# refused every time. An HMAC over a config hash reveals nothing.
_OWN_TOKEN_FIELDS = frozenset({"smoke_token"})
# camelCase carries its word boundaries in the case changes alone, so they have to become
# separators BEFORE the key is lowercased -- lowercasing first collapses ``apiToken`` into one
# unknown word and the credential survives redaction. Two boundaries: lower/digit followed by
# upper (``apiToken``), and an acronym running into a word (``APIToken`` -> ``API`` + ``Token``).
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?
        [a-z0-9_-]*
        (?:token|api[_-]?key|access[_-]?key|password|secret|credential)
        [a-z0-9_-]*
        ["']?
        \s*[:=]\s*
    )
    (?P<value>
        "(?:\\.|[^"\\])*"
        |
        '(?:\\.|[^'\\])*'
        |
        [^\s,;}\]]+
    )
    """
)
_HF_TOKEN_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_VALUE = re.compile(r"(?i)(?P<prefix>\bbasic\s+)(?P<value>[A-Za-z0-9+/]{4,}={0,2})(?=$|[^A-Za-z0-9+/=])")
_URL_USERINFO = re.compile(r"(?i)(?P<scheme>\b[a-z][a-z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@")
_JWT_VALUE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"(?![A-Za-z0-9_-])"
)
# Strong, well-known prefixes plus a minimum opaque suffix length keep this
# conservative: ordinary strings such as ``sk-learn`` are left untouched.
_PREFIXED_TOKEN_VALUE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|nvapi-[A-Za-z0-9_-]{16,}"
    r"|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|AIza[A-Za-z0-9_-]{20,}"
    r")(?![A-Za-z0-9_-])"
)
# Keys holding transcript text (stripped from returns so transcripts don't reach the LLM).
_TRANSCRIPT_KEYS = frozenset({"text", "pred_text", "reference_text", "transcript", "text_ref"})
# Dataset and output parameters that the agent itself reads or writes. This is a
# closed list: substring matching mistakes semantic fields such as
# ``audio_filepath_key`` and ``audio_path_resolution`` for filesystem paths, and
# mistakes model IDs such as ``nvidia/...`` for workspace-relative files.
_PATH_PARAM_NAMES = frozenset(
    {
        "audio_dir",
        "data_dir",
        "file_paths",
        "input_manifest",
        "manifest",
        "manifest_path",
        "output_audio_tar_path",
        "output_dir",
        "output_manifest",
        "output_path",
        "raw_data_dir",
        "resampled_audio_dir",
        "rttm_out_dir",
        "separated_audio_dir",
    }
)
# Deliberately NOT locked, though they resemble the paths above. The lock governs the dataset
# and outputs; a model cache and a packaged config are shared dependencies, closer to
# site-packages than user data. Locking them refused ``cache_dir: ~/.cache/huggingface``
# outright on validate, smoke and run, so a workspace could no longer share a model cache and
# re-downloaded checkpoints per recipe. Pointing them inside the workspace still works.
_UNLOCKED_LOCATION_PARAM_NAMES = frozenset({"cache_dir", "config_path"})
# Parameters that hold EITHER a local path or a hub id, so membership alone cannot decide.
# ``SpeakerSeparationStage`` DEFAULTS ``model_path`` to ``nvidia/diar_sortformer_4spk-v1``
# and ``tokenizer_path`` is documented as an HF tokenizer: locking those by name would make
# the agent refuse its own default recipe, while ignoring them would miss a real local path.
_HUB_OR_PATH_PARAM_NAMES = frozenset({"model_path", "tokenizer_path"})
# Suffixes that mark a value as a concrete local artifact even when written relatively.
_LOCAL_ARTIFACT_SUFFIXES = (
    ".nemo",
    ".joblib",
    ".onnx",
    ".model",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".yaml",
    ".yml",
    ".json",
)
# A hub id is a bare name (``bert-base-uncased``) or one ``org/name`` pair. A second slash
# means a directory tree, which no hub id has, so it can only be a path.
_HUB_ID_SHAPE = re.compile(r"^[\w.-]+(?:/[\w.-]+)?$")


def names_local_path(value: str) -> bool:
    """Whether a dual-purpose value names a filesystem path rather than a hub id.

    Only the hub-id SHAPE is treated as not-a-path: a bare ``org/name`` with no anchor and no
    artifact suffix (``nvidia/diar_sortformer_4spk-v1``). Everything else is checked.

    Stated that way round on purpose. Asking "does this look like a path?" and defaulting to
    no let a bare relative value such as ``out/mymodel`` fall through unchecked, after which the
    stage resolves it against the process working directory -- which may sit anywhere. The
    narrow question has one ambiguous answer (``out/mymodel`` is shaped exactly like a hub id)
    and the containment check below settles it: an ``org/name`` that exists under the workspace
    is a path, and one that does not is the hub id it looks like.

    Existence is probed against the WORKSPACE ROOT, never the working directory. The old
    ``os.path.exists`` probe was CWD-relative, so merely running the agent next to a directory
    named ``nvidia/`` reclassified the default model id as a local file and refused the agent's
    own default recipe -- the same command passing or failing depending on where it was typed.
    """
    text = value.strip()
    if not text:
        return False
    if text.startswith(("~", "./", "../", os.sep)):
        return True
    if os.path.isabs(os.path.expanduser(text)):
        return True
    if text.lower().endswith(_LOCAL_ARTIFACT_SUFFIXES):
        return True
    if not _HUB_ID_SHAPE.match(text):
        return True
    root = workspace_root()
    return bool(root) and os.path.exists(os.path.join(root, text))


def workspace_config_error() -> str | None:
    """Why ``AUDIO_AGENT_WORKSPACE`` is unusable, or None when it is unset or valid.

    A MISCONFIGURED lock must never be indistinguishable from an ABSENT one: reading it
    as "no lock" silently removes the containment a deployment deliberately opted into.
    Callers fail closed on a non-None result.

    Rejected: a relative value (the allowed root would then change with the working
    directory, so the same configuration means different boundaries depending on where
    the process was launched) and anything that is not an existing directory.
    """
    raw = os.environ.get(_WORKSPACE_ENV)
    if not raw:
        return None  # unset: the lock is deliberately disabled
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        return (
            f"{_WORKSPACE_ENV}={raw!r} is relative, so the allowed root would depend on the "
            "working directory; set an absolute path"
        )
    if not os.path.isdir(expanded):
        return f"{_WORKSPACE_ENV}={raw!r} is not an existing directory"
    return None


def workspace_root() -> str | None:
    """The allowed workspace root, or None when the lock is unset OR misconfigured.

    Returning None for a misconfigured value is safe only because every enforcement path
    consults :func:`workspace_config_error` first and refuses; see :func:`path_violations`.
    """
    raw = os.environ.get(_WORKSPACE_ENV)
    if not raw or workspace_config_error():
        return None
    return os.path.realpath(os.path.expanduser(raw))


def path_violations(paths: list[str | None]) -> list[str]:
    """Human-readable violations for any path outside the workspace.

    No-op (returns ``[]``) unless ``AUDIO_AGENT_WORKSPACE`` is set, so normal use
    with data outside the CWD is not blocked; locked-down deployments opt in.
    """
    # A broken lock blocks the operation rather than silently permitting everything.
    misconfigured = workspace_config_error()
    if misconfigured:
        return [misconfigured]
    root = workspace_root()
    if not root:
        return []
    out: list[str] = []
    for p in paths:
        if not p or not isinstance(p, str):
            continue
        parsed = urlsplit(p)
        if parsed.scheme and parsed.scheme not in _LOCAL_URI_SCHEMES:
            continue  # a local workspace root cannot constrain a remote namespace
        local = unquote(parsed.path) if parsed.scheme in _LOCAL_URI_SCHEMES else p
        if parsed.scheme in _LOCAL_URI_SCHEMES and parsed.netloc not in ("", "localhost"):
            local = f"//{parsed.netloc}{local}"
        rp = os.path.realpath(os.path.expanduser(local))
        if rp != root and not rp.startswith(root + os.sep):
            out.append(f"{p!r} resolves outside the allowed workspace {root!r}")
    return out


def recipe_path_params(recipe: Any) -> list[str]:  # noqa: ANN401
    """Path-like string params across a recipe's stages (for the workspace lock)."""
    paths: list[str] = []
    for s in getattr(recipe, "stages", []) or []:
        for k, v in (getattr(s, "params", {}) or {}).items():
            key = str(k).lower()
            if key in _UNLOCKED_LOCATION_PARAM_NAMES:
                # Skipped deliberately and visibly, rather than by silent absence from the
                # set above, so the next person to add a path-looking param sees the choice.
                continue
            dual_purpose = key in _HUB_OR_PATH_PARAM_NAMES
            if key not in _PATH_PARAM_NAMES and not dual_purpose:
                continue
            values = v if isinstance(v, (list, tuple)) else [v]
            for item in values:
                if not isinstance(item, str) or not item:
                    continue
                if dual_purpose and not names_local_path(item):
                    continue  # a hub id names no filesystem location to contain
                paths.append(item)
    return paths


def redact_secret_text(value: str) -> str:
    """Redact credential values embedded in otherwise ordinary error/log text."""

    def replace_assignment(match: re.Match[str]) -> str:
        raw_value = match.group("value")
        quote = (
            raw_value[0]
            if len(raw_value) >= 2 and raw_value[0] in {'"', "'"} and raw_value[-1] == raw_value[0]  # noqa: PLR2004
            else ""
        )
        replacement = f"{quote}<redacted-secret>{quote}"
        return match.group("prefix") + replacement

    def replace_basic(match: re.Match[str]) -> str:
        """Redact only syntactically valid Basic user:password credentials."""
        token = match.group("value")
        unpadded = token.rstrip("=")
        padded = unpadded + ("=" * (-len(unpadded) % 4))
        try:
            decoded = base64.b64decode(padded, validate=True)
        except (ValueError, TypeError):
            return match.group(0)
        if b":" not in decoded:
            return match.group(0)
        return match.group("prefix") + "<redacted-secret>"

    text = _SECRET_ASSIGNMENT.sub(replace_assignment, value)
    text = _URL_USERINFO.sub(r"\g<scheme><redacted-secret>@", text)
    text = _BASIC_VALUE.sub(replace_basic, text)
    text = _HF_TOKEN_VALUE.sub("<redacted-secret>", text)
    text = _BEARER_VALUE.sub("Bearer <redacted-secret>", text)
    text = _JWT_VALUE.sub("<redacted-secret>", text)
    return _PREFIXED_TOKEN_VALUE.sub("<redacted-secret>", text)


def is_secret_key(name: Any) -> bool:  # noqa: ANN401 - any key a caller wants to test
    """Whether a key name designates a secret whose VALUE must never be shown.

    Extracted from :func:`redact` so a caller holding a value under a neutral key -- a recipe
    diff filing ``hf_token``'s old value under ``from`` -- can mask it before it reaches a
    payload ``redact`` would only scan by key. One definition of "secret" for both, so a name
    redaction trusts and a name it does not cannot diverge.
    """
    if str(name).lower() in _OWN_TOKEN_FIELDS:
        return False
    split_case = _CAMEL_BOUNDARY.sub("_", str(name))
    words = [w for w in _KEY_WORD_SPLIT.split(split_case.lower()) if w]
    if any(w in _SECRET_WORDS or w in _SECRET_GLUED_PAIRS for w in words):
        return True
    return any(pair in _SECRET_WORD_PAIRS for pair in itertools.pairwise(words))


def redact(obj: Any, *, redact_transcripts: bool = True) -> Any:  # noqa: ANN401, C901
    """Recursively strip secret-keyed values and (optionally) transcript text.

    Applied to verb return values so tokens never leak and transcripts don't enter
    the host LLM's context. Full transcripts remain in the output files on disk.
    """

    def _is_secret(k: Any) -> bool:  # noqa: ANN401
        return is_secret_key(k)

    def _redacted_transcript(value: Any) -> Any:  # noqa: ANN401
        """Transcript text under a transcript key, whatever shape it arrives in.

        Only bare strings were handled, so a transcript key holding a LIST -- per-segment
        text, per-word text, the shape every segmenting stage produces -- fell through to the
        generic walk and reached the host LLM in full. The key had already been identified as
        transcript-bearing; it was the container that hid it.
        """
        if isinstance(value, str):
            return f"<redacted-transcript:{len(value)}chars>"
        # Tuples and sets alongside lists: the list case was fixed once, and a tuple carrying
        # the same per-segment text walked straight past the fix. ``verbs._jsonable`` admits
        # tuples by name, so a row value shaped that way is a real path, not a hypothetical
        # one. Normalised to a list on the way out, exactly as ``contracts._clean`` does -- the
        # payload is about to be serialized as JSON, where neither type survives.
        if isinstance(value, (list, tuple)):
            return [_redacted_transcript(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted((_redacted_transcript(item) for item in value), key=repr)
        if isinstance(value, dict):
            return {k: _redacted_transcript(v) for k, v in value.items()}
        return value

    def _r(o: Any) -> Any:  # noqa: ANN401
        if isinstance(o, dict):
            out: dict[str, Any] = {}
            for k, v in o.items():
                if _is_secret(k):
                    out[k] = "<redacted-secret>"
                elif redact_transcripts and str(k).lower() in _TRANSCRIPT_KEYS:
                    out[k] = _redacted_transcript(v)
                else:
                    out[k] = _r(v)
            return out
        # Every container the payload can actually hold, not just the two that were noticed
        # first. A secret nested inside a TUPLE used to be returned verbatim: the walk fell
        # through to ``return o`` and handed back the original object untouched.
        # ``contracts._clean`` has always flattened tuples and sets, so this was the outlier.
        if isinstance(o, (list, tuple)):
            return [_r(v) for v in o]
        if isinstance(o, (set, frozenset)):
            return sorted((_r(v) for v in o), key=repr)
        if isinstance(o, str):
            return redact_secret_text(o)
        return o

    return _r(obj)


@lru_cache(maxsize=1)
def _process_smoke_secret() -> bytes:
    """Random per-process secret (fallback when no stable secret is available)."""
    return secrets.token_bytes(32)


def _secret_dir_candidates() -> list[str]:
    """Directories that may hold the shared smoke secret, most-preferred first.

    A single hardcoded ``~/.cache`` breaks the officially recommended container path: on
    a read-only rootfs, or a UID with no passwd entry (``expanduser`` then yields a
    literal ``"~"``), every write raised and the agent silently fell back to a
    per-process secret -- so a smoke in one process could never satisfy the ``run`` in
    another and ``AUDIO_AGENT_REQUIRE_SMOKE`` refused forever. Falling through to the
    agent's own state dir, the XDG dirs, then the temp dir keeps the token verifiable
    across processes wherever *something* is writable.
    """
    out: list[str] = []

    def add(base: str | None, *parts: str) -> None:
        if not base:
            return
        # Expanded here so every source is treated the same way ``run_store.runs_dir``
        # treats AUDIO_AGENT_RUNS_DIR. Without it that one variable meant two different
        # directories: the run records went to the real home while the secret beside them
        # went to a literal "~" folder created under the working directory -- so the secret
        # moved with the CWD and stopped being shared, which is the one thing it is for.
        base = os.path.expanduser(base)
        if base.startswith("~"):  # no HOME / no passwd entry -> never create a "~" dir
            return
        path = os.path.join(base, *parts)
        if path not in out:
            out.append(path)

    add(os.environ.get(_RUNS_DIR_ENV), "secrets")
    add(workspace_root(), ".audio_agent_state")
    add(os.environ.get("XDG_STATE_HOME"), "nemo_curator")
    add(os.environ.get("XDG_CACHE_HOME"), "nemo_curator")
    add("~", ".cache", "nemo_curator")
    add(tempfile.gettempdir(), "nemo_curator")
    return out


def _is_private_file(file: Path) -> bool:
    """Whether a secret file is ours alone -- owned by this user, closed to everyone else.

    The load-bearing check on the fallback chain. Its last candidate lives under the temp dir,
    whose parent is world-writable, so any local user can pre-create ``<tmp>/nemo_curator`` and
    leave a secret of their choosing in it. Creation here is deliberately if-absent, so the
    agent would ADOPT that file and sign every smoke token with a key its author already knows
    -- ``AUDIO_AGENT_REQUIRE_SMOKE`` still reporting as enforced while anyone could mint a
    token for any config hash.

    Ownership is what settles it, and it settles it cheaply: a planted file belongs to whoever
    planted it, and nobody can create a file owned by us. A group member with write access to
    the directory can still delete ours and leave their own, but that costs them the shared
    secret (we refuse it and fall through to a per-process key) rather than winning them a
    forgeable one -- denial, not forgery.

    Deliberately NOT paired with a mode check on the DIRECTORY. The directories this agent
    already created in the field are group-writable under an ordinary 002 umask (both
    ``~/.cache/nemo_curator`` and ``/tmp/nemo_curator`` are 0775 on the machine this was
    written on), so refusing those would send every one of them to a per-process secret --
    reinstating the exact cross-process failure the fallback chain was added to fix.
    """
    geteuid = getattr(os, "geteuid", None)  # POSIX-only; elsewhere the mode check stands alone
    try:
        info = file.stat()
    except OSError:
        return False
    if geteuid is not None and info.st_uid != geteuid():
        return False
    return not info.st_mode & 0o077


def _stored_secret(file: Path) -> bytes | None:
    """The secret held in an existing file, or None when it is absent or unusable."""
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("secret") if isinstance(payload, dict) else None
    return str(value).encode() if value else None


def _read_or_create_secret(path: str) -> bytes | None:
    """Read the shared smoke secret at ``path``, creating it once if absent.

    The create-or-lose-the-race step is the package's own
    :func:`~nemo_curator.utils.atomic_io.write_json_atomically_if_absent` -- the same
    fsynced-temp-then-``os.link`` algorithm this used to hand-roll, already relied on by
    ``backends/slurm_array.py``, and it leaves the file owner-only (0600) because the temp
    file it links in is created that way.
    """
    from nemo_curator.utils.atomic_io import write_json_atomically_if_absent

    file = Path(path)
    try:
        # Owner-only for a directory we create here. An existing one is left as the deployment
        # set it -- ``run_store._ensure_private_dir`` reasons the same way -- because the file
        # check below is what decides whether a secret may be trusted, and tightening a
        # directory somebody configured on purpose is not this function's call to make.
        file.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        for _attempt in (0, 1):
            secret = _stored_secret(file)
            if secret:
                # Only a file that is OURS may hand us a key to sign with. One holding a secret
                # we cannot vouch for is refused rather than healed -- deleting somebody else's
                # file is not this function's business, and adopting it is the whole attack.
                # Falling through to the next candidate (ultimately a per-process key) costs
                # cross-process reuse, which is the safe direction to fail in.
                return secret if _is_private_file(file) else None
            # Present but empty or unparseable is NOT a usable secret, and leaving it in
            # place is what made this unrecoverable before: ``os.link`` kept failing against
            # the dead file, so every process silently fell back to its own per-process key
            # and cross-process smoke evidence never worked again. Clear it, then create.
            with contextlib.suppress(OSError):
                file.unlink(missing_ok=True)
            write_json_atomically_if_absent(file, {"secret": secrets.token_hex(32)})
    except OSError:
        return None
    return None


@lru_cache(maxsize=1)
def _smoke_secret() -> bytes:
    """HMAC key for smoke tokens.

    Precedence: ``AUDIO_AGENT_SMOKE_SECRET`` env (pin across machines / CI) > a secret
    persisted in the first writable candidate directory (so a smoke in one process and
    the run in another share it) > a random per-process secret (still unforgeable within
    this process, e.g. a long-lived MCP server, but not shareable across them).
    """
    env = os.environ.get(_SMOKE_SECRET_ENV)
    if env:
        return env.encode("utf-8")
    for directory in _secret_dir_candidates():
        secret = _read_or_create_secret(os.path.join(directory, "audio_agent_smoke.secret"))
        if secret:
            return secret
    return _process_smoke_secret()


def smoke_token(config_hash: str | None) -> str:
    """Unforgeable proof that a smoke ran for this exact (frozen) recipe.

    HMAC over the config_hash keyed by a deployment/process secret (see
    :func:`_smoke_secret`), so — unlike a plain hash of the public config_hash — it
    cannot be minted by anything that merely knows the config_hash.
    """
    mac = hmac.new(_smoke_secret(), f"audio_agent_smoke|{config_hash}".encode(), hashlib.sha256)
    return mac.hexdigest()[:24]


def verify_smoke_token(token: str | None, config_hash: str | None) -> bool:
    """True iff ``token`` is the smoke token for ``config_hash`` (constant-time).

    A wrong token has to come back as ``False``, never as an exception. ``compare_digest``
    rejects str arguments outside ASCII, so a token relayed with a smart quote or an en-dash --
    what a chat UI does to a hex string on its way through -- raised ``TypeError`` out of
    ``run``, where the design calls for a refusal the host can read and act on. Comparing the
    encoded bytes treats every string uniformly and stays constant-time; anything that is not a
    string at all cannot be the token.
    """
    if not isinstance(token, str) or not isinstance(config_hash, str) or not token or not config_hash:
        return False
    try:
        candidate = token.encode("utf-8", "surrogatepass")
    except UnicodeEncodeError:  # pragma: no cover - defensive, surrogatepass covers lone surrogates
        return False
    return hmac.compare_digest(candidate, smoke_token(config_hash).encode("utf-8"))


def require_smoke() -> bool:
    """Whether ``run`` must refuse without a valid smoke token (opt-in)."""
    return os.environ.get("AUDIO_AGENT_REQUIRE_SMOKE", "").strip().lower() in ("1", "true", "yes", "on")
