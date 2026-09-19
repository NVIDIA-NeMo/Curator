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

"""Recipe IR — the single artifact the host LLM emits and the core consumes.

A ``Recipe`` is a typed, hashable, serializable description of an ordered audio
pipeline: a list of ``{ref, params}`` stages plus inputs and an optional preset.
It round-trips to the ``stages:`` YAML that ``nemo_curator.config.run`` already
understands, and freezes to a ``recipe_id`` + ``config_hash`` for reproducibility
and plan-execution integrity ("what was approved is what runs").

The Recipe IR is the anti-hallucination boundary: the host proposes a Recipe,
never raw Python, and the core validates its mechanically provable composition
(``verbs.validate``) before anything runs. Open-ended intent fit remains a
separate host-LLM critique over the returned semantic-review evidence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


# Constructor keys that configure the framework, not stage semantics; peeled out
# and re-applied via .with_() rather than passed to the dataclass constructor.
EXECUTION_KNOB_PARAMS = frozenset({"resources", "batch_size", "runtime_env", "num_workers"})
# Hashed policy carried by a stage but excluded from reuse identity because it
# changes approval requirements, not the bytes the stage produces.
#
# ``retention_sec`` / ``owner`` are ManifestCheckpointStage's lifecycle policy: how long the
# checkpoint may be kept and who may delete it. They decide nothing about its contents, but
# while they were semantic they moved the step key -- and since a checkpoint is now ADDRESSED
# by that key, asking to keep the same scores for a week wrote them to a different file and
# missed the cache that already held them. Same computation, three addresses.
NON_SEMANTIC_POLICY_PARAMS = frozenset({"planning_provenance", "retention_sec", "owner"})
REUSABLE_CHECKPOINT_PROVENANCE = "reusable_pipeline_v1"
_REUSABLE_CHECKPOINT_REF = "ManifestCheckpointStage"
PLANNING_PREFERENCE_SCHEMA_VERSION = 1
CURATION_MODES = frozenset({"refine_later", "fast_first"})
PLANNING_PREFERENCE_SOURCES = frozenset({"explicit_user_choice", "inferred_from_request"})

# Params that name WHERE a stage writes, not WHAT it computes. Changing one moves the
# bytes; it does not change them. Excluded from ``semantic_hash`` (so a re-run into a new
# directory can reuse prior work) and scanned by output discovery (so resampled/per-speaker/
# RTTM directories stop being invisible side effects). See REUSE_ARCHITECTURE.md.
OUTPUT_LOCATION_PARAMS = frozenset(
    {
        "output_path",  # ManifestWriterStage, Snippet*/PretrainMetrics* writers
        "output_manifest",  # recipe-level alias mapped onto output_path
        "output_dir",  # SegmentExtraction / MonoConversion / SegmentConcatenation / SnippetExtraction
        "output_audio_tar_path",  # SnippetExtractionStage
        "resampled_audio_dir",  # ResampleAudioStage
        "separated_audio_dir",  # SpeakerSeparationStage
        "rttm_out_dir",  # InferenceSortformerStage
    }
)


def _criteria(raw: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - shape-checking untrusted input IS the job
    """The success contract, or a loud error -- never a silently mangled one.

    ``list()`` over a mapping yields its KEYS, so a plausible-looking
    ``acceptance_criteria: {must: [...]}`` used to become ``["must"]``: a contract that
    cannot be verified, which ``run`` then skipped without a word while reporting success.
    The shape is checked here, at the door, so the CLI, the SDK and a recipe file all
    reject the same mistake with the same message. ``ValueError`` (not ``TypeError``) to
    match ``from_dict`` and ``acceptance.parse_criteria``, which callers already catch.
    """
    if raw is None:
        return []
    shape = "{id, type, check: {field, op, value}, severity}"
    if not isinstance(raw, list):
        detail = f" with keys {sorted(raw, key=repr)!r}" if isinstance(raw, dict) else ""
        msg = (
            f"acceptance_criteria must be a LIST of criterion mappings, got {type(raw).__name__}{detail}; "
            f"each criterion is {shape}"
        )
        raise ValueError(msg)  # noqa: TRY004 - ValueError, per this module's convention
    bad = [f"#{i} ({type(c).__name__})" for i, c in enumerate(raw) if not isinstance(c, dict)]
    if bad:
        msg = f"acceptance_criteria entries must be mappings, got {', '.join(bad)}; each criterion is {shape}"
        raise ValueError(msg)
    copied = [dict(c) for c in raw]

    # Validate through the same boundary used by the SDK ``verify`` verb, but retain
    # the exact host-authored mappings here. Adding normalized defaults to Recipe would
    # change config_hash/contract_hash and break already-frozen tutorial recipes.
    from nemo_curator.audio_agent.acceptance import parse_criteria

    parse_criteria(copied)
    return copied


def _criteria_hash_payload(raw: list[Any]) -> list[dict[str, Any]]:
    """Canonical criterion mappings for hashing without rewriting raw recipes.

    ``Recipe.from_dict`` deliberately preserves the exact host-authored mappings
    because adding normalized defaults would change existing confirmation hashes.
    The SDK also permits a directly constructed ``Recipe`` to contain
    ``AcceptanceCriterion`` objects. Convert only those objects to their mapping
    form so ``recipe -> to_dict -> recipe`` keeps the same integrity hashes.
    """
    from nemo_curator.audio_agent.acceptance import parse_criteria

    parsed = parse_criteria(raw)
    return [
        dict(original) if isinstance(original, dict) else criterion.to_dict()
        for original, criterion in zip(raw, parsed, strict=True)
    ]


def parse_planning_preference(raw: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Validate optional, non-semantic host planning metadata.

    This is deliberately a small closed shape. A typo here must not silently
    change how the host breaks ties, while an omitted field keeps every existing
    recipe valid.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        msg = "planning_preference must be a mapping with schema_version, curation_mode, and source"
        raise ValueError(msg)  # noqa: TRY004 - one public recipe schema error type
    value = dict(raw)
    expected = {"schema_version", "curation_mode", "source"}
    missing = sorted(expected - set(value))
    unknown = sorted((key for key in value if key not in expected), key=repr)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing required field(s) {missing}")
        if unknown:
            details.append(f"unknown field(s) {unknown}")
        msg = "planning_preference has " + " and ".join(details)
        raise ValueError(msg)
    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PLANNING_PREFERENCE_SCHEMA_VERSION
    ):
        msg = (
            f"planning_preference.schema_version must be {PLANNING_PREFERENCE_SCHEMA_VERSION}, got {schema_version!r}"
        )
        raise ValueError(msg)
    mode = value["curation_mode"]
    if not isinstance(mode, str) or mode not in CURATION_MODES:
        msg = f"planning_preference.curation_mode must be one of {sorted(CURATION_MODES)}, got {mode!r}"
        raise ValueError(msg)
    source = value["source"]
    if not isinstance(source, str) or source not in PLANNING_PREFERENCE_SOURCES:
        msg = f"planning_preference.source must be one of {sorted(PLANNING_PREFERENCE_SOURCES)}, got {source!r}"
        raise ValueError(msg)
    return {
        "schema_version": PLANNING_PREFERENCE_SCHEMA_VERSION,
        "curation_mode": str(mode),
        "source": str(source),
    }


@dataclass
class StageRef:
    """One stage in a recipe: a registered stage class name + its params."""

    ref: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "params": dict(self.params)}

    def semantic_params(self) -> dict[str, Any]:
        """Params that change the stage's OUTPUT BYTES — execution knobs and output
        locations removed. This is the reuse identity of the stage's configuration."""
        skip = EXECUTION_KNOB_PARAMS | OUTPUT_LOCATION_PARAMS | NON_SEMANTIC_POLICY_PARAMS
        return {k: v for k, v in self.params.items() if k not in skip}

    def semantic_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "params": self.semantic_params()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageRef:
        if not isinstance(d, dict) or "ref" not in d:
            msg = f"stage entry must be a dict with a 'ref' key, got {d!r}"
            raise ValueError(msg)
        raw_params = d.get("params") or {}
        try:
            params = dict(raw_params)
        except (TypeError, ValueError) as exc:
            # ``params`` written as a string is an ordinary host-LLM slip, and it used to fall
            # through to whatever ``dict()`` raises -- "dictionary update sequence element #0
            # has length 1", or a TypeError the rest of this parser never raises. ``validate``
            # should hand back something the host can act on. Conversion is unchanged: anything
            # that parsed before, including a list of key/value pairs, still parses.
            msg = f"stage {str(d['ref'])!r}: 'params' must be a mapping of parameter name to value, got {raw_params!r}"
            raise ValueError(msg) from exc
        return cls(ref=str(d["ref"]), params=params)


@dataclass
class Recipe:
    """An ordered, configured audio pipeline the agent builds and validates."""

    stages: list[StageRef] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    preset: str | None = None
    # The confirmed success contract (1A.3). PORTABLE + HASHED: it is part of the
    # user-confirmed intent, so config_hash covers it -- a 'must' bar cannot be
    # silently relaxed after confirmation without changing the hash (integrity).
    acceptance_criteria: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    name: str = "audio_agent_recipe"
    recipe_id: str | None = None
    config_hash: str | None = None
    # Reuse identity, split out of config_hash (REUSE_ARCHITECTURE.md §2). config_hash stays
    # the confirm-gate integrity anchor and covers EVERYTHING the user approved; these two
    # answer the narrower questions reuse actually asks.
    semantic_hash: str | None = None  # "would this produce the same bytes?"
    contract_hash: str | None = None  # "is the success bar the same?" (re-verify, don't recompute)
    # Layered save: recomputable annotations kept OUT of the hash so the recipe
    # stays portable. Re-run on a different machine/dataset recomputes these rather
    # than reusing stale, machine-/data-specific numbers.
    machine_plan: dict[str, Any] | None = None  # mode + per-stage resources (per machine)
    data_derived: dict[str, Any] | None = None  # data-derived values, e.g. relative thresholds (per dataset)
    config_strategy: list[dict[str, Any]] | None = None  # how each param was chosen (1A.2 audit trail)
    knowledge_version: str | None = None  # knowledge/cards version the plan was approved against
    parent_run_id: str | None = None  # provenance chain for incremental continuation
    # Host policy attestation, bound to config_hash but excluded from computation semantics.
    # Present only when the user explicitly chose the baseline after the core exposed a
    # recommended reusable checkpoint.
    checkpoint_decision: dict[str, Any] | None = None
    # Optional host planning tie-breaker. It records how this workflow was
    # authored, not what the stages compute, so every recipe hash excludes it.
    planning_preference: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # (de)serialization
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        if not isinstance(d, dict):
            msg = f"recipe must be a dict, got {type(d).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        acceptance_typos = sorted(
            str(key) for key in d if str(key).startswith("acceptance_") and key != "acceptance_criteria"
        )
        if acceptance_typos:
            msg = (
                f"unknown acceptance contract field(s) {acceptance_typos}; "
                "use 'acceptance_criteria' so the success contract is validated "
                "and included in the confirmation hash"
            )
            raise ValueError(msg)
        stages = [StageRef.from_dict(s) for s in (d.get("stages") or [])]
        raw_inputs = d.get("inputs") or {}
        try:
            inputs = dict(raw_inputs)
        except (TypeError, ValueError) as exc:
            # Same shape of slip as a stage's ``params``, and it landed the same way: a raw
            # ``dict()`` error about update sequences, from the verb whose job is to say what
            # is wrong with the recipe. Conversion is unchanged.
            msg = f"'inputs' must be a mapping of input name to value, got {raw_inputs!r}"
            raise ValueError(msg) from exc
        return cls(
            stages=stages,
            inputs=inputs,
            preset=d.get("preset"),
            acceptance_criteria=_criteria(d.get("acceptance_criteria")),
            rationale=str(d.get("rationale") or ""),
            name=str(d.get("name") or "audio_agent_recipe"),
            recipe_id=d.get("recipe_id"),
            config_hash=d.get("config_hash"),
            semantic_hash=d.get("semantic_hash"),
            contract_hash=d.get("contract_hash"),
            machine_plan=d.get("machine_plan"),
            data_derived=d.get("data_derived"),
            config_strategy=d.get("config_strategy"),
            knowledge_version=d.get("knowledge_version"),
            parent_run_id=d.get("parent_run_id"),
            checkpoint_decision=d.get("checkpoint_decision"),
            planning_preference=parse_planning_preference(d.get("planning_preference")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["stages"] = [s.to_dict() for s in self.stages]
        if self.checkpoint_decision is None:
            out.pop("checkpoint_decision", None)
        if self.planning_preference is None:
            out.pop("planning_preference", None)
        return out

    def _canonical(self) -> str:
        """Stable JSON of the PORTABLE semantic content only (stages + inputs + preset
        + acceptance_criteria — the confirmed success contract).

        Deliberately excludes id/hash/rationale AND the recomputable layered-save
        annotations (``machine_plan`` / ``data_derived`` / ``config_strategy`` /
        ``knowledge_version`` / ``parent_run_id`` / ``checkpoint_decision`` /
        ``planning_preference``), so
        ``config_hash`` stays portable:
        the same intent on a different machine or dataset hashes identically. (The
        *resolved* param values live in ``stages`` and so are hashed; the
        ``config_strategy`` audit trail explaining them is not.)
        """
        payload = {
            "stages": [s.to_dict() for s in self.stages],
            "inputs": self.inputs,
            "preset": self.preset,
            "acceptance_criteria": _criteria_hash_payload(self.acceptance_criteria),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def compute_hash(self) -> str:
        return hashlib.sha256(self._canonical().encode("utf-8")).hexdigest()[:16]

    def compute_semantic_hash(self) -> str:
        """Identity of what this recipe COMPUTES, ignoring how and where it runs.

        Drops the execution knobs (``resources`` / ``batch_size`` / ``num_workers`` /
        ``runtime_env``) and the output *locations* from every stage, and the acceptance
        criteria entirely -- none of those change a single output byte. Two recipes with the
        same ``semantic_hash`` on the same data produce the same result, so the second one
        can reuse the first's artifacts. (``config_hash`` still covers all of it, so the
        confirm gate is unaffected.)
        """
        payload = {
            "stages": [s.semantic_dict() for s in self.stages],
            "inputs": {k: v for k, v in self.inputs.items() if k not in OUTPUT_LOCATION_PARAMS},
            "preset": self.preset,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def compute_contract_hash(self) -> str:
        """Identity of the success bar alone. A changed bar means re-VERIFY the reused
        data against the new criteria, never recompute it."""
        blob = json.dumps(
            _criteria_hash_payload(self.acceptance_criteria),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def freeze(self) -> Recipe:
        """Stamp ``config_hash`` (integrity anchor) + a stable ``recipe_id``, plus the
        ``semantic_hash`` / ``contract_hash`` reuse keys.

        Only the portable layer (see :meth:`_canonical`) is hashed into ``config_hash``; the
        layered-save annotations are attached separately and never change it.
        """
        self.config_hash = self.compute_hash()
        self.semantic_hash = self.compute_semantic_hash()
        self.contract_hash = self.compute_contract_hash()
        if not self.recipe_id:
            self.recipe_id = f"{self.name}-{self.config_hash[:8]}"
        return self

    # ------------------------------------------------------------------ #
    # layered save: recomputable annotations (never affect config_hash)
    # ------------------------------------------------------------------ #
    def with_machine_plan(self, plan: dict[str, Any], *, machine_fingerprint: str) -> Recipe:
        """Attach the resolved machine plan (mode + per-stage resources), stamped with
        the machine it was computed for. Does not change ``config_hash``."""
        self.machine_plan = {**plan, "machine_fingerprint": machine_fingerprint}
        return self

    def with_data_derived(self, values: dict[str, Any], *, data_fingerprint: str) -> Recipe:
        """Attach data-derived values (e.g. relative thresholds), stamped with the
        dataset they were computed from. Does not change ``config_hash``."""
        self.data_derived = {**values, "data_fingerprint": data_fingerprint}
        return self

    def with_config_strategy(self, entries: list[dict[str, Any]]) -> Recipe:
        """Attach the config-strategy audit trail (how each param was chosen, 1A.2).

        The *resolved values* already live in ``stages`` (and so are hashed); this
        is the explanation + provenance (source/kind/recompute_on) that rides
        alongside as a recomputable annotation. Does not change ``config_hash``."""
        self.config_strategy = list(entries)
        return self

    def stale_layers(
        self, *, machine_fingerprint: str | None = None, data_fingerprint: str | None = None
    ) -> list[str]:
        """Recomputable layers that must be (re)built for the given machine/data.

        A layer is stale when it is absent or was stamped for a different
        fingerprint, so a re-run recomputes it instead of reusing stale numbers.
        """
        stale: list[str] = []
        if (
            machine_fingerprint is not None
            and (self.machine_plan or {}).get("machine_fingerprint") != machine_fingerprint
        ):
            stale.append("machine_plan")
        if data_fingerprint is not None and (self.data_derived or {}).get("data_fingerprint") != data_fingerprint:
            stale.append("data_derived")
        return stale

    # ------------------------------------------------------------------ #
    # pipeline-config bridge (round-trips to config.run's `stages:` format)
    # ------------------------------------------------------------------ #
    def to_pipeline_config(self) -> dict[str, Any]:
        """Return the ``{"stages": [...]}`` dict ``create_pipeline_from_yaml`` reads."""
        from nemo_curator.audio_agent._resolve import resolve_target

        stages_cfg: list[dict[str, Any]] = []
        for s in self.stages:
            entry: dict[str, Any] = {"_target_": resolve_target(s.ref)}
            entry.update(s.params)
            stages_cfg.append(entry)
        return {"stages": stages_cfg}


def _accepted_params(cls: type) -> list[str]:
    """Constructor param names a stage accepts, for a helpful ``bad_params`` error."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    return [n for n, p in sig.parameters.items() if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def build_stages(recipe: Recipe) -> tuple[list[ProcessingStage] | None, list[dict[str, Any]]]:
    """Instantiate a recipe's stages, returning ``(stages, issues)``.

    Instantiation doubles as a pre-flight check: a ``ref`` whose module failed to
    import (missing optional dep) or whose required params are absent yields an
    actionable issue instead of a stage. ``resources`` is applied via ``.with_()``.
    Returns ``(None, issues)`` if any stage could not be built.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class

    issues: list[dict[str, Any]] = []
    stages: list[ProcessingStage] = []

    for idx, s in enumerate(recipe.stages):
        requested_workers = s.params.get("num_workers", 1)
        if s.ref == _REUSABLE_CHECKPOINT_REF and (
            isinstance(requested_workers, bool) or not isinstance(requested_workers, int) or requested_workers != 1
        ):
            issues.append(
                {
                    "code": "checkpoint_single_worker_required",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": (
                        "ManifestCheckpointStage requires num_workers=1 so its exclusive "
                        "output reservation and JSONL appends remain authoritative"
                    ),
                    "fix": "remove the num_workers override or set it to 1",
                }
            )
            continue
        try:
            cls = resolve_stage_class(s.ref)
        except KeyError:
            issues.append(
                {
                    "code": "unknown_stage",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"{s.ref!r} is not a registered agent-ready audio stage in this environment",
                    "fix": "check the name via discover(), or install the extra that provides it (audio_cpu/audio_cuda12)",
                }
            )
            continue
        except Exception as e:  # noqa: BLE001 - import-time failure of an optional dep
            issues.append(
                {
                    "code": "stage_import_error",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"could not load {s.ref!r}: {type(e).__name__}: {e}",
                    "fix": "install the audio extra that provides this stage's dependency",
                }
            )
            continue

        params = dict(s.params)
        with_kwargs = {k: params.pop(k) for k in list(params) if k in EXECUTION_KNOB_PARAMS}
        try:
            inst = cls(**params)
            if with_kwargs:
                inst = _apply_with(inst, with_kwargs)
        except TypeError as e:
            accepted = _accepted_params(cls)
            fix = (
                f"accepted params for {s.ref}: {accepted}"
                if accepted
                else "check required/allowed params via describe() or cards()"
            )
            issues.append(
                {
                    "code": "bad_params",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"could not construct {s.ref!r} with params {sorted(params)}: {e}",
                    "fix": fix,
                }
            )
            continue
        except Exception as e:  # noqa: BLE001 - stage __post_init__ validation, etc.
            issues.append(
                {
                    "code": "construct_error",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"{s.ref!r} rejected its configuration: {type(e).__name__}: {e}",
                    "fix": "see the stage's card for valid parameter ranges",
                }
            )
            continue
        stages.append(inst)

    if any(i["severity"] == "error" for i in issues):
        return None, issues
    return stages, issues


def _apply_with(stage: ProcessingStage, with_kwargs: dict[str, Any]) -> ProcessingStage:
    """Apply framework knobs via ``.with_()``, coercing a resources dict."""
    resources = with_kwargs.get("resources")
    if isinstance(resources, dict):
        from nemo_curator.stages.resources import Resources

        with_kwargs = dict(with_kwargs)
        with_kwargs["resources"] = Resources(**resources)
    return stage.with_(**with_kwargs)
