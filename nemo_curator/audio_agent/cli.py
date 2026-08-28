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

"""CLI adapter over the audio-agent verbs — the shell/notebook surface.

Every subcommand maps 1:1 to a verb and prints a JSON result, so a host agent
(or a human) can drive the ``discover -> route -> validate -> smoke -> confirm ->
run -> report`` loop from a terminal. The same core also backs the MCP adapter.

    python -m nemo_curator.audio_agent discover
    python -m nemo_curator.audio_agent catalog-tree
    python -m nemo_curator.audio_agent cards --category quality
    python -m nemo_curator.audio_agent validate --recipe recipe.yaml --data data.jsonl
    python -m nemo_curator.audio_agent smoke   --recipe recipe.yaml --sample 10
    python -m nemo_curator.audio_agent run     --recipe recipe.yaml --confirm <hash>
    python -m nemo_curator.audio_agent report  --output out/ --data data.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_RECIPE_DATA_HELP = (
    "optional assertion that must canonically match the recipe's first source stage; never overrides stage parameters"
)


def _load_recipe(path: str) -> dict[str, Any]:
    import yaml

    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _load_doc(path: str | None) -> Any:  # noqa: ANN401
    """Load a YAML/JSON doc from a file path (or ``-`` for stdin); ``None`` -> None."""
    if not path:
        return None
    import yaml

    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _criteria_list(doc: Any) -> list[dict[str, Any]] | None:  # noqa: ANN401
    """Accept a bare list, or a mapping carrying ``acceptance_criteria`` / ``criteria``.

    Fail LOUD (ValueError) when a non-empty doc doesn't yield a criteria list -- a common
    mistake is a top-level mapping like ``{output_completeness: ..., yield: ...}`` -- so a
    malformed criteria file is reported instead of being silently ignored (which used to
    make ``validate``/``verify`` skip the contract without any warning).
    """
    if doc is None:
        return None
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        crit = doc.get("acceptance_criteria")
        if crit is None:
            crit = doc.get("criteria")
        if crit is None:
            if not doc:  # genuinely empty mapping -> no criteria
                return None
            msg = (
                "acceptance criteria not recognized: expected a YAML/JSON LIST of criteria, "
                "or a mapping with an 'acceptance_criteria' (or 'criteria') key holding that "
                f"list; got a mapping with top-level keys {sorted(doc)!r}. Each criterion is "
                "{id, type, check:{field,op,value}, severity}; e.g.\n"
                "  acceptance_criteria:\n"
                "    - {id: dur, type: output_completeness, compiles_to: duration, severity: must}\n"
                "    - {id: keep, type: yield, kind: absolute, check: {op: '==', value: 4}, severity: must}"
            )
            raise ValueError(msg)
        if not isinstance(crit, list):
            msg = f"'acceptance_criteria' must be a list of criterion mappings, got {type(crit).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        return crit
    msg = f"acceptance criteria must be a list or a mapping with 'acceptance_criteria', got {type(doc).__name__}"
    raise ValueError(msg)


def _calibration_arg(path: str | None) -> dict[str, Any] | None:
    """Load a bare calibration or wrapper without discarding wrapper metadata."""
    return _load_doc(path)


def _emit(obj: Any) -> None:  # noqa: ANN401
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _result_exit_code(cmd: str, obj: Any) -> int:  # noqa: ANN401, PLR0911
    """Stable shell semantics for structured verb outcomes.

    ``validate`` intentionally returns zero even for a non-runnable verdict:
    validation answered successfully and callers inspect its JSON. Execution,
    evidence, and lookup failures return non-zero so shell automation cannot
    accidentally continue after a structured refusal.
    """
    if not isinstance(obj, dict):
        return 0
    acceptance = obj.get("acceptance")
    if isinstance(acceptance, dict) and acceptance.get("overall") is not None and acceptance.get("overall") != "met":
        return 1
    if cmd == "validate":
        return 0
    if cmd == "diagnose" and str(obj.get("status") or "").lower() == "unknown":
        return 1
    if "error" in obj:
        return 1
    if str(obj.get("status") or "").lower() in {
        "action_required",
        "blocked",
        "error",
        "fail",
        "failed",
        "refused",
    }:
        return 1
    if cmd == "smoke" and obj.get("goals_met") is False:
        return 1
    if cmd == "verify" and obj.get("overall") not in {None, "met"}:
        return 1
    return 0


def _finish(cmd: str, obj: Any) -> int:  # noqa: ANN401
    _emit(obj)
    return _result_exit_code(cmd, obj)


def _parse_goal(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        goal = json.loads(raw)
    except json.JSONDecodeError:
        return {"task": raw}
    if not isinstance(goal, dict):
        msg = "goal JSON must be an object mapping; pass unquoted free text or a JSON object"
        raise ValueError(msg)  # noqa: TRY004
    return goal


def _parse_params(raw: str | None) -> dict[str, Any] | None:
    """Stage params as a JSON object, or ``None`` when none were given.

    Refuses a non-object rather than passing it on: ``describe`` would report it as a stage that
    "could not be configured", blaming the stage for the caller's quoting.
    """
    if not raw:
        return None
    params = json.loads(raw)
    if not isinstance(params, dict):
        msg = f"--params must be a JSON object mapping param names to values, got {type(params).__name__}"
        raise ValueError(msg)  # noqa: TRY004
    return params


def _parse_scalar(raw: str | None) -> Any:  # noqa: ANN401 - JSON scalar or plain string
    """A finite JSON scalar when possible, otherwise literal string text."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if value is None or isinstance(value, (list, dict)):
        msg = "--decision-value must be a JSON scalar (boolean, number, or string); omit it for no change"
        raise ValueError(msg)
    if isinstance(value, float) and not math.isfinite(value):
        msg = "--decision-value must be a finite JSON number"
        raise ValueError(msg)
    return value


def _parse_conditions(raw: str | None) -> list[Any] | dict[str, Any] | None:
    """A complete compound decision condition list or mapping."""
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, (list, dict)):
        msg = "--decision-conditions must be a non-empty JSON list or object mapping"
        raise ValueError(msg)  # noqa: TRY004 - public CLI input errors use ValueError
    return value


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - one flat block per subcommand
    p = argparse.ArgumentParser(prog="nemo_curator.audio_agent", description="Audio Agent (P1) tool surface")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover", help="list agent-ready audio stages with category + one-liner")
    sub.add_parser("catalog-tree", help="L0 category tree for coarse-to-fine routing")

    d = sub.add_parser("describe", help="contract (+ card) for one stage, resolved against --params")
    d.add_argument("name")
    d.add_argument(
        "--params",
        help=(
            "JSON object of the params the recipe will use; reads/writes follow from them "
            '(e.g. \'{"segments_key": "diar_segments"}\'). Omitting them describes the defaults'
        ),
    )

    pr = sub.add_parser("producers", help="which stages write a role or key (answers 'who makes segments?')")
    pr.add_argument("role", help="a semantic role (segments, pred_text) or a literal key name")

    c = sub.add_parser("cards", help="L1 (one-liners for a category) or L2 (full cards for names)")
    c.add_argument("--category")
    c.add_argument("--names", nargs="*")

    ctx = sub.add_parser("context", help="assemble a PlanningContext for the router/planner")
    ctx.add_argument("--goal", help="JSON goal spec, or a free-text task string")
    ctx.add_argument("--data", help="dataset path to profile directly for pre-recipe planning")
    ctx.add_argument("--stages", nargs="*")
    ctx.add_argument("--roles", nargs="*")
    ctx.add_argument(
        "--planning-mode",
        choices=("refine_later", "fast_first"),
        help="optional soft planning tie-breaker chosen for this workflow",
    )
    ctx.add_argument(
        "--planning-source",
        choices=("explicit_user_choice", "inferred_from_request"),
        default="explicit_user_choice",
        help="how --planning-mode was selected",
    )

    v = sub.add_parser(
        "validate",
        help="validate mechanical runnability and emit host semantic-review context",
    )
    v.add_argument("--recipe", required=True, help="path to a recipe YAML/JSON (or - for stdin)")
    v.add_argument("--data", help=_RECIPE_DATA_HELP)
    v.add_argument(
        "--expected-outputs", nargs="*", help="semantic output roles the user asked for (output-completeness)"
    )
    v.add_argument(
        "--acceptance-criteria",
        help=(
            "optional cross-check file (list or {acceptance_criteria: [...]}); "
            "the same criteria must be embedded in the recipe so run hashes them"
        ),
    )
    v.add_argument("--request-type", help="goal/request kind (e.g. filter, transcribe) for request-type sanity")

    s = sub.add_parser("smoke", help="bounded run for evidence")
    s.add_argument("--recipe", required=True)
    s.add_argument("--sample", type=int, default=10)
    s.add_argument("--data", help=_RECIPE_DATA_HELP)
    s.add_argument(
        "--output-dir",
        help=(
            "legacy no-op retained for compatibility; smoke always redirects "
            "stage-declared outputs to an ephemeral sandbox"
        ),
    )
    s.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    s.add_argument("--calibration", help="path to a calibration JSON from a prior smoke (1C.2)")

    r = sub.add_parser("run", help="confirm-gated full run (0 silent runs)")
    r.add_argument("--recipe", required=True)
    r.add_argument(
        "--confirm",
        nargs="?",
        const=True,
        default=False,
        help="pass the recipe config_hash (integrity) or bare --confirm",
    )
    r.add_argument("--data", help=_RECIPE_DATA_HELP)
    r.add_argument(
        "--output-dir",
        help=("legacy no-op retained for compatibility; configure output paths on recipe stages"),
    )
    r.add_argument("--checkpoint-path")
    r.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    r.add_argument(
        "--smoke-token", help="smoke-evidence token from a prior smoke (required if AUDIO_AGENT_REQUIRE_SMOKE is set)"
    )
    r.add_argument(
        "--calibration",
        help="path to a calibration JSON from a prior smoke; omit to apply the measurements the last smoke of this recipe stored",
    )
    r.add_argument("--goal", help="what this run is FOR (JSON or free text); recorded so prior work stays legible")

    rp = sub.add_parser("report", help="post-hoc report from an output manifest/dir")
    rp.add_argument("--output", required=True)
    rp.add_argument("--recipe")
    rp.add_argument(
        "--data",
        help=(
            "without --recipe, input used for counts; with --recipe, an optional "
            "assertion that must match its first source stage"
        ),
    )

    vf = sub.add_parser("verify", help="verify acceptance criteria against evidence -> AcceptanceReport")
    vf.add_argument(
        "--criteria",
        required=True,
        help="acceptance criteria YAML/JSON (list or {acceptance_criteria: [...]}; - for stdin)",
    )
    vf.add_argument("--evidence", help="evidence YAML/JSON (produced_roles/metrics/retained/...); - for stdin")
    vf.add_argument("--frozen-criteria", help="the confirmed contract, for the honesty guard (YAML/JSON)")
    vf.add_argument(
        "--recipe", dest="verify_recipe", help="recipe carrying acceptance_criteria (alt source for the honesty guard)"
    )

    rs = sub.add_parser("resolve", help="resolve an outcome (label/use_case/explicit) to concrete stage config (1A.2)")
    rs.add_argument("--stage", required=True)
    rs.add_argument("--label", help="outcome label, e.g. studio / transcription_grade")
    rs.add_argument("--use-case", help="named card preset, e.g. tts_reference")
    rs.add_argument("--explicit", help="JSON object of {param: value}")
    rs.add_argument("--data", help="profile this dataset and bind observed values (Path B)")

    dg = sub.add_parser("diagnose", help="analyze a captured failure and return grounded user choices")
    dg.add_argument("--error", required=True, help="captured error text, or - to read it from stdin")
    dg.add_argument("--recipe", help="optional recipe YAML/JSON for stage-aware applicability")
    dg.add_argument("--operation", default="run", choices=["validate", "smoke", "run"])
    dg.add_argument("--phase", default="runtime", help="where the failure occurred")
    dg.add_argument("--attempted-actions", nargs="*", help="actions already tried, so they are not repeated")
    dg.add_argument(
        "--execution-target",
        choices=["local", "external_ray", "custom_executor"],
        help="where stages execute (default: infer local vs RAY_ADDRESS)",
    )

    ru = sub.add_parser("runs", help="list local run records (provenance) or show one by id")
    ru.add_argument("--run-id", help="show a single run record")
    ru.add_argument("--data", help="filter to runs/artifacts for this source dataset (a path, or a dataset_key)")
    ru.add_argument(
        "--goal",
        help=(
            "with --data as a folder path: rank priors by how much of this current request is "
            "covered by each prior's recorded prompt + pipeline_summary (compare before inventing a recipe)"
        ),
    )
    ru.add_argument("--stage", help="filter artifacts to one stage")
    ru.add_argument("--since", help="only records created at/after this ISO timestamp")
    ru.add_argument("--limit", type=int, default=50, help="max runs/artifacts to return")

    sc = sub.add_parser("reuse-scan", help="find prior work this recipe could reuse (read-only)")
    sc.add_argument("--recipe", required=True, help="the recipe to scan for (or - for stdin)")
    sc.add_argument("--data", help=_RECIPE_DATA_HELP)
    sc.add_argument("--limit", type=int, default=5)

    dr = sub.add_parser(
        "delta-run",
        help="run only the files that changed since a prior run and merge them into its result",
    )
    dr.add_argument("--recipe", help="the same recipe as the prior run (or - for stdin); omit with --from-run")
    dr.add_argument(
        "--from-run",
        help=(
            "adopt this prior run's own recipe instead of passing one (a run_id from runs / "
            "reuse-scan's prior_on_same_path), so the delta matches the pipeline that produced it"
        ),
    )
    dr.add_argument("--data", help=_RECIPE_DATA_HELP)
    dr.add_argument(
        "--confirm",
        nargs="?",
        const=True,
        default=False,
        help="pass the recipe config_hash (integrity) or bare --confirm; omit to see the card",
    )
    dr.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    dr.add_argument(
        "--smoke-token", help="smoke-evidence token from a prior smoke (required if AUDIO_AGENT_REQUIRE_SMOKE is set)"
    )
    dr.add_argument("--calibration", help="path to a calibration JSON from a prior smoke")
    dr.add_argument("--goal", help="what this run is FOR (JSON or free text); recorded so prior work stays legible")

    ck = sub.add_parser(
        "add-checkpoint",
        help="where a mid-pipeline manifest would make the expensive stages reusable (read-only)",
    )
    ck.add_argument("--recipe", required=True, help="the recipe to place a checkpoint in (or - for stdin)")
    ck.add_argument("--data", help="dataset path; lets the checkpoint location be derived instead of named")
    ck.add_argument(
        "--output-path",
        help="override the managed checkpoint location; omit to let --data derive it",
    )
    ck.add_argument("--after", help="place it after this stage instead of where the agent would put it")

    pc = sub.add_parser(
        "plan-checkpoint",
        help="build validated same-dataset checkpoint candidates before authoritative smoke",
    )
    pc.add_argument("--recipe", help="initial recipe to analyze (or - for stdin); omit with --from-run")
    pc.add_argument("--from-run", help="adopt an exact completed recipe for threshold feedback")
    pc.add_argument("--data", help="current dataset path; with --from-run it must match the prior dataset")
    pc.add_argument(
        "--output-path",
        help="override the managed checkpoint location; omit to let --data derive it",
    )
    pc.add_argument("--decision-stage", help="producer stage whose downstream decision is being tuned")
    decision = pc.add_mutually_exclusive_group()
    decision.add_argument("--decision-value", help="new JSON scalar decision value for scalar feedback")
    decision.add_argument(
        "--decision-conditions",
        help=(
            "complete JSON list/object of card-declared compound ge conditions; replaces the selector condition set"
        ),
    )
    pc.add_argument(
        "--choice",
        choices=["checkpoint", "baseline"],
        help="select the checkpoint candidate or explicitly decline it for the baseline",
    )
    pc.add_argument("--retention-sec", type=int, default=0, help="0 means user-managed/no expiry")
    pc.add_argument("--owner", default="user", help="who owns checkpoint retention/deletion")

    ck2 = sub.add_parser("checkpoints", help="list the managed checkpoint cache, or collect what nothing can reuse")
    ck2.add_argument(
        "--gc",
        action="store_true",
        help="delete orphaned and expired checkpoints (never a reusable one, never a path you chose)",
    )

    sub.add_parser("reindex", help="rebuild the run/artifact index from the JSON records")

    cont = sub.add_parser("continue", help="plan (and optionally execute) a follow-up run that reuses prior work")
    cont.add_argument("--recipe", required=True, help="the follow-up recipe (or - for stdin)")
    cont.add_argument(
        "--parent-run-id", help="a prior run to diff against (optional; the artifact scan works without one)"
    )
    cont.add_argument("--data", help=_RECIPE_DATA_HELP)
    cont.add_argument("--execute", action="store_true", help="carry the plan out instead of only printing it")
    cont.add_argument(
        "--choice",
        choices=["as_is", "extend", "fresh"],
        help="which option to take (default: what the plan concluded)",
    )
    cont.add_argument(
        "--confirm",
        nargs="?",
        const=True,
        default=False,
        help="config_hash of the recipe that will run (integrity), or bare --confirm",
    )
    cont.add_argument(
        "--output-dir",
        help=("legacy no-op retained for compatibility; configure output paths on recipe stages"),
    )
    cont.add_argument("--checkpoint-path")
    cont.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    cont.add_argument("--smoke-token", help="smoke token for the exact recipe branch that will execute")
    cont.add_argument(
        "--calibration",
        help="path to a calibration JSON from a prior smoke; omit to apply the measurements the last smoke of this recipe stored",
    )
    cont.add_argument("--goal", help="what this run is FOR (JSON or free text)")

    cal = sub.add_parser("calibrate", help="extract measured per-stage resources from a smoke report (1C.2)")
    cal.add_argument("--smoke", required=True, help="path to a smoke-result JSON (or - for stdin)")

    dr = sub.add_parser("doctor", help="check environment health (driver/CUDA, ffmpeg, deps, ...) + get fix steps")
    dr.add_argument("--json", action="store_true", help="emit the JSON report instead of human-readable text")

    ins = sub.add_parser("install-skill", help="install the packaged skills where Codex/Cursor/Claude Code find them")
    ins.add_argument(
        "--scope",
        choices=["project", "user"],
        default="project",
        help="project: the current directory; user: the home-directory equivalents (default: project)",
    )
    ins.add_argument(
        "--host",
        choices=["all", "claude", "codex", "cursor"],
        default="all",
        help="which host's discovery directory to write (default: all)",
    )
    ins.add_argument(
        "--skill", dest="skills", nargs="*", help="install only these packaged skills (default: all of them)"
    )
    ins.add_argument("--dest", help="project-scope root to install into instead of the current directory")
    mode = ins.add_mutually_exclusive_group()
    mode.add_argument(
        "--copy",
        dest="mode",
        action="store_const",
        const="copy",
        help="copy the files (default; works without symlink support)",
    )
    mode.add_argument(
        "--symlink",
        dest="mode",
        action="store_const",
        const="symlink",
        help="link to the installed package so the skill tracks upgrades",
    )
    ins.set_defaults(mode="copy")
    ins.add_argument("--force", action="store_true", help="replace a target whose content differs")
    ins.add_argument("--dry-run", action="store_true", help="report what would be written, and write nothing")
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0911 - a flat verb dispatch table
    from nemo_curator import audio_agent as aa

    args = build_parser().parse_args(argv)
    cmd = args.cmd

    try:
        if cmd == "discover":
            return _finish(cmd, aa.discover())
        elif cmd == "catalog-tree":
            return _finish(cmd, aa.catalog_tree())
        elif cmd == "describe":
            return _finish(cmd, aa.describe(args.name, _parse_params(args.params)))
        elif cmd == "producers":
            return _finish(cmd, aa.producers(args.role))
        elif cmd == "cards":
            return _finish(cmd, aa.cards(category=args.category, names=args.names))
        elif cmd == "context":
            return _finish(
                cmd,
                aa.context(
                    _parse_goal(args.goal),
                    data=args.data,
                    stages=args.stages,
                    roles=args.roles,
                    planning_preference=(
                        {
                            "schema_version": 1,
                            "curation_mode": args.planning_mode,
                            "source": args.planning_source,
                        }
                        if args.planning_mode
                        else None
                    ),
                ),
            )
        elif cmd == "validate":
            return _finish(
                cmd,
                aa.validate(
                    _load_recipe(args.recipe),
                    data=args.data,
                    expected_outputs=args.expected_outputs,
                    acceptance_criteria=_criteria_list(_load_doc(args.acceptance_criteria)),
                    request_type=args.request_type,
                ),
            )
        elif cmd == "smoke":
            return _finish(
                cmd,
                aa.smoke(
                    _load_recipe(args.recipe),
                    sample=args.sample,
                    data=args.data,
                    output_dir=args.output_dir,
                    bootstrap_ray=args.bootstrap_ray,
                    calibration=_calibration_arg(args.calibration),
                ),
            )
        elif cmd == "run":
            return _finish(
                cmd,
                aa.run(
                    _load_recipe(args.recipe),
                    confirm=args.confirm,
                    data=args.data,
                    output_dir=args.output_dir,
                    checkpoint_path=args.checkpoint_path,
                    bootstrap_ray=args.bootstrap_ray,
                    smoke_token=args.smoke_token,
                    calibration=_calibration_arg(args.calibration),
                    goal=_parse_goal(args.goal),
                ),
            )
        elif cmd == "report":
            recipe = _load_recipe(args.recipe) if args.recipe else None
            return _finish(
                cmd,
                aa.report(args.output, recipe=recipe, data=args.data),
            )
        elif cmd == "verify":
            frozen = _criteria_list(_load_doc(args.frozen_criteria)) if args.frozen_criteria else None
            rec = _load_recipe(args.verify_recipe) if args.verify_recipe else None
            return _finish(
                cmd,
                aa.verify(
                    _criteria_list(_load_doc(args.criteria)) or [],
                    evidence=_load_doc(args.evidence),
                    frozen_criteria=frozen,
                    recipe=rec,
                ),
            )
        elif cmd == "resolve":
            explicit = json.loads(args.explicit) if args.explicit else None
            return _finish(
                cmd,
                aa.resolve(
                    args.stage,
                    label=args.label,
                    use_case=args.use_case,
                    explicit=explicit,
                    data=args.data,
                ),
            )
        elif cmd == "diagnose":
            error_text = sys.stdin.read() if args.error == "-" else args.error
            diagnosis_recipe = _load_recipe(args.recipe) if args.recipe else None
            return _finish(
                cmd,
                aa.diagnose(
                    error_text,
                    recipe=diagnosis_recipe,
                    operation=args.operation,
                    phase=args.phase,
                    attempted_actions=args.attempted_actions,
                    execution_target=args.execution_target,
                ),
            )
        elif cmd == "runs":
            return _finish(
                cmd,
                aa.runs(
                    run_id=args.run_id,
                    data=args.data,
                    stage=args.stage,
                    since=args.since,
                    limit=args.limit,
                    goal=args.goal,
                ),
            )
        elif cmd == "reuse-scan":
            return _finish(
                cmd,
                aa.reuse_scan(
                    _load_recipe(args.recipe),
                    data=args.data,
                    limit=args.limit,
                ),
            )
        elif cmd == "delta-run":
            return _finish(
                cmd,
                aa.delta_run(
                    _load_recipe(args.recipe) if args.recipe else None,
                    from_run=args.from_run,
                    data=args.data,
                    confirm=args.confirm,
                    bootstrap_ray=args.bootstrap_ray,
                    smoke_token=args.smoke_token,
                    calibration=_calibration_arg(args.calibration),
                    goal=_parse_goal(args.goal),
                ),
            )
        elif cmd == "add-checkpoint":
            return _finish(
                cmd,
                aa.add_checkpoint(
                    _load_recipe(args.recipe),
                    data=args.data,
                    output_path=args.output_path,
                    after=args.after,
                ),
            )
        elif cmd == "plan-checkpoint":
            return _finish(
                cmd,
                aa.plan_checkpoint(
                    _load_recipe(args.recipe) if args.recipe else None,
                    from_run=args.from_run,
                    data=args.data,
                    output_path=args.output_path,
                    decision_stage=args.decision_stage,
                    decision_value=_parse_scalar(args.decision_value),
                    decision_conditions=_parse_conditions(args.decision_conditions),
                    choice=args.choice,
                    retention_sec=args.retention_sec,
                    owner=args.owner,
                ),
            )
        elif cmd == "checkpoints":
            return _finish(cmd, aa.checkpoints(gc=args.gc))
        elif cmd == "reindex":
            return _finish(cmd, aa.reindex())
        elif cmd == "continue":
            return _finish(
                cmd,
                aa.plan_continuation(
                    _load_recipe(args.recipe),
                    args.parent_run_id,
                    data=args.data,
                    execute=args.execute,
                    choice=args.choice,
                    confirm=args.confirm,
                    output_dir=args.output_dir,
                    checkpoint_path=args.checkpoint_path,
                    bootstrap_ray=args.bootstrap_ray,
                    smoke_token=args.smoke_token,
                    calibration=_calibration_arg(args.calibration),
                    goal=_parse_goal(args.goal),
                ),
            )
        elif cmd == "calibrate":
            return _finish(cmd, aa.calibrate(_load_doc(args.smoke) or {}))
        elif cmd == "doctor":
            rep = aa.doctor()
            if getattr(args, "json", False):
                return _finish(cmd, rep)
            print(aa.render_doctor(rep))
            return _result_exit_code(cmd, rep)
        elif cmd == "install-skill":
            return _finish(
                cmd,
                aa.install_skill(
                    scope=args.scope,
                    host=args.host,
                    mode=args.mode,
                    skills=args.skills,
                    dest=args.dest,
                    force=args.force,
                    dry_run=args.dry_run,
                ),
            )
        else:  # pragma: no cover - argparse enforces the choices
            return 2
    except Exception as e:  # noqa: BLE001 - CLI failures must remain structured
        _emit({"error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
