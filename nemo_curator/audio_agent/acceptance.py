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

"""Acceptance layer (1A.1): turn a success contract into deterministic checks.

Three metric-agnostic pieces the deterministic core owns:

* :func:`expected_roles_from_criteria` — compile output/quality criteria into
  producible-role requirements so ``validate``'s output-completeness check
  catches "asked for X, nothing produces X" *before* a run.
* :func:`missing_implied` — request-type sanity: a request implying an output
  must carry the matching criterion (a *filtering* request needs a yield
  criterion; a *transcription* request needs its transcript output). A missing
  implied criterion is surfaced (so success can't be declared while ignoring the
  point of the request).
* :func:`verify` — evaluate each criterion against gathered evidence into an
  :class:`AcceptanceReport` with four honest states (met / not_met /
  unverifiable / unachievable). Never marks ``met`` without evidence, never
  silently relaxes a ``must``.

Everything is generic: metric/output keys are variables (``check.field``); there
are no metric names in the logic. A new metric ships a card block and works here
unchanged.
"""

from __future__ import annotations

import math
import operator
from typing import Any

from nemo_curator.audio_agent.contracts import AcceptanceCriterion, AcceptanceReport, CriterionResult

# Request-type -> the criterion TYPE it implies (type-level, not metric-level, so
# it stays generic). Substring match on the goal's request_type/task keeps it
# robust to phrasing. Unknown request types imply nothing (no false flags).
_REQUEST_TYPE_IMPLIES: tuple[tuple[str, str], ...] = (
    ("filter", "yield"),  # "keep the clean ones" must define how much is kept
    ("transcri", "output_completeness"),  # transcription must declare its transcript output
    ("caption", "output_completeness"),
    ("diariz", "output_completeness"),  # diarization must declare its speaker-label output
    ("align", "output_completeness"),
)
_IMPLIED_HINT = {
    "yield": "a filtering/curation request should carry a 'yield' criterion (define how much to keep, e.g. retained > 0 or ~=N%)",
    "output_completeness": "a request that produces an output should carry an 'output_completeness' criterion naming that output",
}

_OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt}


def parse_criteria(raw: Any) -> list[AcceptanceCriterion]:  # noqa: ANN401
    """Validate and parse a list of mappings (or criterion objects).

    Guards the SDK path: a non-list ``raw`` (e.g. a ``{output_completeness: ...}`` mapping)
    is a shape error, not an empty contract -- raise a clear error instead of silently
    iterating its keys and producing garbage (which is how malformed criteria used to be
    ignored end-to-end). IDs are unique so the honesty guard cannot lose a criterion
    through last-write-wins dictionary construction.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        detail = f" with keys {sorted(raw, key=repr)!r}" if isinstance(raw, dict) else ""
        msg = (
            f"acceptance criteria must be a LIST of criterion mappings, got {type(raw).__name__}{detail}; "
            "each is {id, type, check:{field,op,value}, severity}"
        )
        raise ValueError(msg)  # noqa: TRY004 - one public schema-error type
    parsed: list[AcceptanceCriterion] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        try:
            criterion = AcceptanceCriterion.from_dict(item)
        except (TypeError, ValueError) as e:
            msg = f"acceptance criterion #{i}: {e}"
            raise ValueError(msg) from e
        if criterion.id in seen:
            msg = f"acceptance criterion #{i}: duplicate id {criterion.id!r}"
            raise ValueError(msg)
        seen.add(criterion.id)
        parsed.append(criterion)
    return parsed


def missing_implied(request_type: str | None, criteria: list[AcceptanceCriterion]) -> list[tuple[str, str]]:
    """Implied-but-absent criterion types for the request type, as ``(type, hint)``."""
    if not request_type:
        return []
    rt = request_type.lower()
    present = {c.type for c in criteria}
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for token, implied in _REQUEST_TYPE_IMPLIES:
        if token in rt and implied not in present and implied not in seen:
            seen.add(implied)
            out.append((implied, _IMPLIED_HINT.get(implied, f"request implies a {implied!r} criterion")))
    return out


def expected_roles_from_criteria(criteria: list[AcceptanceCriterion]) -> list[str]:
    """Producible-role requirements compiled from the criteria (for output-completeness).

    A criterion's ``compiles_to`` (explicit role) or ``check.field`` becomes a
    required output role — but only when it resolves to a *known* role, so an
    unrecognized metric key is skipped rather than flagged as a false gap.

    ``yield`` is intentionally excluded: it checks a run-level *retained count*
    (evidence.retained vs op/value), not a produced manifest field, so a stray
    ``check.field`` on a yield (e.g. ``field: retained``) must NOT become a
    producible-role requirement (that produced a spurious ``missing_output_producer``).
    """
    out: list[str] = []
    for c in criteria:
        if not c.is_deterministic:
            continue
        target: str | None = None
        if c.compiles_to and c.compiles_to != "producible_role":
            target = c.compiles_to
        elif c.type in ("output_completeness", "quality_standard"):
            target = c.field_name
        # Keep the target even when it has no known role: metric fields (wer, sig sub-scores)
        # collapse to role "unknown"/"score", so dropping them here silently skipped
        # output-completeness for exactly the metrics the tool exists for. The check now
        # matches a target against produced roles OR literal produced keys.
        if target:
            out.append(target)
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(  # noqa: C901, PLR0912, PLR0915
    criteria: list[AcceptanceCriterion],
    evidence: dict[str, Any],
    *,
    frozen_criteria: list[AcceptanceCriterion] | None = None,
) -> AcceptanceReport:
    """Evaluate each criterion against evidence into an :class:`AcceptanceReport`.

    ``evidence`` (all optional) may carry: ``produced_roles`` / ``produced_keys``
    (from ``validate``), ``metrics`` (aggregate ``{field: value}``; normal
    terminal-manifest evidence uses a complete-corpus arithmetic mean), ``per_item``
    (``[{field: value}, ...]``), ``output_scan`` (a complete terminal-manifest
    coverage summary), ``expected_output_rows`` (only when the serializer proves
    its cardinality), ``retained`` / ``input_count`` (from smoke/run), and
    ``unachievable_fields`` (metrics the data provably cannot meet).

    ``frozen_criteria`` (the user-confirmed contract, e.g. from the recipe) enables
    the honesty guard (1A.3): if the criteria actually verified are weaker than what
    was confirmed (a ``must`` dropped/downgraded/relaxed), it is flagged and
    ``overall`` is forced to ``not_met`` — success can't be declared against a
    silently relaxed bar.
    """
    if evidence is None:
        ev: dict[str, Any] = {}
    elif not isinstance(evidence, dict):
        msg = f"evidence must be a mapping, got {type(evidence).__name__}"
        raise ValueError(msg)
    else:
        ev = evidence

    def _string_set(key: str) -> set[str]:
        raw = ev.get(key)
        if raw is None:
            return set()
        if isinstance(raw, str) or not isinstance(raw, (list, tuple, set)):
            msg = f"evidence.{key} must be a collection of strings, got {type(raw).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        if any(not isinstance(value, str) for value in raw):
            msg = f"evidence.{key} must contain only strings"
            raise ValueError(msg)
        return set(raw)

    produced_roles = _string_set("produced_roles")
    produced_keys = _string_set("produced_keys")
    metrics = ev.get("metrics")
    if metrics is None:
        metrics = {}
    elif not isinstance(metrics, dict):
        msg = f"evidence.metrics must be a mapping, got {type(metrics).__name__}"
        raise ValueError(msg)
    raw_per_item = ev.get("per_item") or []
    if not isinstance(raw_per_item, list):
        msg = f"evidence.per_item must be a list of mappings, got {type(raw_per_item).__name__}"
        raise ValueError(msg)  # noqa: TRY004
    bad_rows = [f"#{i} ({type(row).__name__})" for i, row in enumerate(raw_per_item) if not isinstance(row, dict)]
    if bad_rows:
        msg = f"evidence.per_item entries must be mappings, got {', '.join(bad_rows)}"
        raise ValueError(msg)
    per_item = list(raw_per_item)
    raw_output_scan = ev.get("output_scan")
    if raw_output_scan is None:
        output_scan = {}
    elif not isinstance(raw_output_scan, dict):
        msg = f"evidence.output_scan must be a mapping, got {type(raw_output_scan).__name__}"
        raise ValueError(msg)
    else:
        output_scan = raw_output_scan
    expected_output_rows = ev.get("expected_output_rows")
    retained = ev.get("retained")
    input_count = ev.get("input_count")
    unachievable = _string_set("unachievable_fields")

    results = [
        _verify_one(
            c,
            produced_roles,
            produced_keys,
            metrics,
            per_item,
            output_scan,
            expected_output_rows,
            retained,
            input_count,
            unachievable,
        )
        for c in criteria
    ]
    musts = [r for r in results if r.severity == "must"]
    if not results:  # noqa: SIM108 - the comments explain the non-vacuous empty-contract case
        # Empty contract: nothing was verified, so neither "met" (all([]) is vacuously true --
        # success on zero evidence) nor "not_met" (over-rejects a run that carried no explicit
        # bar). A dropped 'must' is still caught by the honesty guard below. A NON-empty contract
        # of only 'nice' criteria stays "met" by design; see test_nice_only_contract_stays_met.
        overall = "unverifiable"
    else:
        overall = "met" if all(r.status == "met" for r in musts) else "not_met"

    honesty = honesty_review(frozen_criteria, criteria) if frozen_criteria is not None else []
    if honesty:
        overall = "not_met"  # a relaxed/dropped 'must' bar cannot be declared met
    return AcceptanceReport(
        overall=overall, criteria=results, verdict=_summary(results, overall, honesty), honesty=honesty
    )


# --------------------------------------------------------------------------- #
# honesty guard (1A.3): anti-goalpost-moving
# --------------------------------------------------------------------------- #
_STRICTER_WHEN = {">=": "higher", ">": "higher", "<=": "lower", "<": "lower"}


def honesty_review(frozen: list[AcceptanceCriterion], used: list[AcceptanceCriterion]) -> list[dict[str, Any]]:
    """Goalpost-moving violations: a confirmed ``must`` that is dropped, downgraded,
    or relaxed in the criteria actually verified. Empty when ``used`` honors ``frozen``.
    """
    used_by_id = {c.id: c for c in used}
    out: list[dict[str, Any]] = []
    for fc in frozen:
        if fc.severity != "must":
            continue
        uc = used_by_id.get(fc.id)
        if uc is None:
            out.append(
                {
                    "code": "must_dropped",
                    "id": fc.id,
                    "message": f"confirmed must-criterion {fc.id!r} is missing from the verified set",
                }
            )
        elif uc.severity != "must":
            out.append(
                {
                    "code": "must_downgraded",
                    "id": fc.id,
                    "message": f"must-criterion {fc.id!r} downgraded to {uc.severity!r}",
                }
            )
        else:
            reason = _weakened_reason(fc, uc)
            if reason:
                out.append(
                    {"code": "must_relaxed", "id": fc.id, "message": f"must-criterion {fc.id!r} relaxed: {reason}"}
                )
    return out


def _weakened_reason(  # noqa: PLR0911 - explicit comparison outcomes aid auditability
    fc: AcceptanceCriterion,
    uc: AcceptanceCriterion,
) -> str | None:
    """Why ``uc`` is not semantically the same (or provably stricter), else ``None``.

    A changed type, scope, method, output target, or failure policy is not safely
    orderable, even if a numeric threshold happens to look stricter. Such a change
    needs a new confirmation. The sole automatic exception is a same-operator
    numeric strengthening (or a narrower ``~=`` tolerance at the same target).
    """
    frozen = _semantic_contract(fc)
    used = _semantic_contract(uc)
    for key in (
        "type",
        "kind",
        "target",
        "field",
        "scope",
        "method",
        "op",
        "on_unachievable",
    ):
        if frozen[key] != used[key]:
            return f"{key} {frozen[key]!r} -> {used[key]!r}"

    op = str(frozen["op"] or "")
    frozen_value, used_value = frozen["value"], used["value"]
    if op == "~=":
        if frozen_value != used_value:
            return f"value {frozen_value!r} -> {used_value!r}"
        # Keep the tolerance comparison defensive for manually constructed objects
        # (strict schema validation guarantees numerics on the normal path), matching
        # the numeric-threshold branch below -- a non-numeric tolerance must not raise
        # out of the honesty guard.
        try:
            frozen_tol, used_tol = float(frozen["tolerance"]), float(used["tolerance"])
        except (TypeError, ValueError):
            return (
                f"tolerance {frozen['tolerance']!r} -> {used['tolerance']!r}"
                if frozen["tolerance"] != used["tolerance"]
                else None
            )
        if used_tol > frozen_tol:
            return f"tolerance {frozen['tolerance']!r} -> {used['tolerance']!r} (wider/easier to satisfy)"
        return None

    stricter = _STRICTER_WHEN.get(op)
    if stricter is None:
        return f"value {frozen_value!r} -> {used_value!r}" if frozen_value != used_value else None
    # Strict schema validation guarantees numeric values for deterministic metric
    # criteria, but keep this helper defensive for manually constructed objects.
    try:
        frozen_num, used_num = float(frozen_value), float(used_value)
    except (TypeError, ValueError):
        return f"value {frozen_value!r} -> {used_value!r}" if frozen_value != used_value else None
    easier = used_num < frozen_num if stricter == "higher" else used_num > frozen_num
    return f"target {frozen_value} -> {used_value} (easier to satisfy)" if easier else None


def _semantic_contract(c: AcceptanceCriterion) -> dict[str, Any]:
    """Canonical success-bar semantics; display/provenance fields are excluded."""
    check = c.check or {}
    target = c.compiles_to if c.compiles_to != "producible_role" else None
    target = target or check.get("field")
    field = check.get("field") or target
    if c.type == "output_completeness":
        scope = check.get("scope") or "per_retained_item"
        op = check.get("op") or "non_empty"
    else:
        scope = check.get("scope") or "aggregate"
        op = check.get("op")
    method = check.get("method")
    if method is None:
        method = "reviewer_judgment" if c.type in {"semantic_fit", "honesty"} else "deterministic"
    return {
        "type": c.type,
        "kind": c.kind or "absolute",
        "target": target,
        "field": field,
        "scope": scope,
        "method": method,
        "op": op,
        "value": check.get("value"),
        "tolerance": check.get("tolerance", 0),
        "on_unachievable": c.on_unachievable,
    }


def _verify_one(  # noqa: C901, PLR0911, PLR0912, PLR0913 - one honest branch per criterion type
    c: AcceptanceCriterion,
    produced_roles: set[str],
    produced_keys: set[str],
    metrics: dict[str, Any],
    per_item: list[dict[str, Any]],
    output_scan: dict[str, Any],
    expected_output_rows: Any,  # noqa: ANN401
    retained: Any,  # noqa: ANN401
    input_count: Any,  # noqa: ANN401
    unachievable: set[str],
) -> CriterionResult:
    def result(status: str, evidence: str = "", note: str = "") -> CriterionResult:
        return CriterionResult(id=c.id, status=status, severity=c.severity, evidence=evidence, note=note)

    if not c.is_deterministic:
        return result("unverifiable", note="semantic criterion — routed to the reviewer (LLM)")

    field = c.field_name
    chk = c.check or {}

    if c.type == "output_completeness":
        target = c.compiles_to if (c.compiles_to and c.compiles_to != "producible_role") else field
        status, evidence, note = _completeness(
            target=target,
            physical_field=field,
            produced_roles=produced_roles,
            produced_keys=produced_keys,
            per_item=per_item,
            output_scan=output_scan,
            expected_output_rows=expected_output_rows,
        )
        return result(status, evidence=evidence, note=note)

    if field in unachievable:
        return result("unachievable", evidence=f"data cannot meet the target for {field!r}")

    if c.type == "yield":
        if retained is None:
            return result("unverifiable", note="no retained count in evidence")
        retained_count = _retained_count(retained)
        if retained_count is None:
            return result(
                "unverifiable",
                note=f"invalid retained count evidence: {retained!r}",
            )
        op = chk.get("op", ">")
        val = chk.get("value", 0)
        tol = chk.get("tolerance", 0)
        if c.kind == "relative":
            if input_count is None:
                return result(
                    "unverifiable",
                    note="relative yield needs an input_count denominator",
                )
            input_count_value = _retained_count(input_count)
            if input_count_value is None:
                return result(
                    "unverifiable",
                    note=f"invalid input_count evidence: {input_count!r}",
                )
            if input_count_value == 0:
                return result(
                    "unverifiable",
                    note="relative yield needs a positive input_count denominator",
                )
            if retained_count > input_count_value:
                return result(
                    "unverifiable",
                    note=f"inconsistent counts: retained={retained} vs input_count={input_count}",
                )
            frac = 100.0 * retained_count / input_count_value
            ok = _cmp(frac, op or "~=", _num(val), tol)
            return result(
                "met" if ok else "not_met", evidence=f"retained {retained}/{input_count} = {frac:.1f}% vs {op} {val}"
            )
        ok = _cmp(_num(retained_count), op, _num(val), tol)
        return result("met" if ok else "not_met", evidence=f"retained={retained} vs {op} {val}")

    if c.type in ("quality_standard", "distribution"):
        if not field:
            return result("unverifiable", note="no metric field named")
        op = chk.get("op", ">=")
        val = chk.get("value")
        tol = chk.get("tolerance", 0)
        if chk.get("scope") == "per_retained_item":
            scan_result = _per_item_scan_result(
                field,
                op,
                val,
                tol,
                output_scan,
                expected_output_rows,
            )
            if scan_result is not None:
                status, evidence, note = scan_result
                return result(status, evidence=evidence, note=note)
            vals = [d[field] for d in per_item if field in d]
            if not vals:
                return result("unverifiable", note=f"no per-item values for {field!r}")
            expected_rows = _retained_count(expected_output_rows)
            if expected_output_rows is not None and expected_rows is None:
                return result(
                    "unverifiable",
                    note=(f"invalid expected_output_rows evidence: {expected_output_rows!r}"),
                )
            if expected_rows is not None and len(per_item) != expected_rows:
                return result(
                    "not_met",
                    evidence=(
                        f"per-item evidence has {len(per_item)} row(s), but expected_output_rows={expected_rows}"
                    ),
                )
            if len(vals) != len(per_item):
                return result(
                    "not_met",
                    evidence=(f"{field!r} present on only {len(vals)}/{len(per_item)} per-item evidence row(s)"),
                )
            numbers = [_finite_number(item) for item in vals]
            if any(number is None for number in numbers):
                valid = sum(number is not None for number in numbers)
                return result(
                    "not_met",
                    evidence=(
                        f"{field!r} is a finite JSON number on only {valid}/{len(vals)} per-item evidence row(s)"
                    ),
                )
            ok = all(_cmp(number, op, float(val), tol) for number in numbers if number is not None)
            return result("met" if ok else "not_met", evidence=f"{len(vals)} items vs {op} {val}")
        if field in metrics:
            number = _finite_number(metrics[field])
            if number is None:
                return result(
                    "unverifiable",
                    note=f"aggregate evidence for {field!r} is not a finite JSON number",
                )
            ok = _cmp(number, op, float(val), tol)
            return result("met" if ok else "not_met", evidence=f"{field}={metrics[field]} vs {op} {val}")
        return result("unverifiable", note=f"no aggregate metric for {field!r}")

    # honesty and any future types: no deterministic evidence path yet -> reviewer.
    return result("unverifiable", note=f"criterion type {c.type!r} is not deterministically checkable (reviewer)")


def _per_item_scan_result(  # noqa: C901, PLR0911, PLR0912, PLR0913 - evidence states stay explicit
    field: str,
    op: str,
    target: Any,  # noqa: ANN401
    tolerance: Any,  # noqa: ANN401
    output_scan: dict[str, Any],
    expected_output_rows: Any,  # noqa: ANN401
) -> tuple[str, str, str] | None:
    """Exhaustive per-item metric result when a terminal scan is available."""
    if not output_scan:
        return None
    if _is_nested_field(field) and output_scan.get("field_scope") == "top_level":
        return (
            "unverifiable",
            "",
            f"terminal scan covers top-level fields only, not nested field {field!r}",
        )
    status = str(output_scan.get("status") or "unavailable")
    read_errors = int(output_scan.get("read_errors") or 0)
    if read_errors:
        return (
            "unverifiable",
            "",
            f"terminal output scan was {status}; {read_errors} file(s) could not be read",
        )
    invalid = int(output_scan.get("malformed_rows") or 0) + int(output_scan.get("blank_rows") or 0)
    if invalid:
        return (
            "not_met",
            f"terminal output contains {invalid} malformed/blank row(s)",
            "",
        )
    rows = int(output_scan.get("valid_rows") or 0)
    if rows == 0:
        return (
            "unverifiable",
            "",
            f"terminal output scan was {status}; no per-item values were available",
        )
    expected_rows = _retained_count(expected_output_rows)
    if expected_output_rows is None:
        return (
            "unverifiable",
            "",
            "terminal rows were scanned, but no trustworthy serialized-row count "
            "was available to prove per-retained-item coverage",
        )
    if expected_rows is None:
        return (
            "unverifiable",
            "",
            f"invalid expected_output_rows evidence: {expected_output_rows!r}",
        )
    if rows != expected_rows:
        return (
            "not_met",
            f"terminal output has {rows} row(s), but expected_output_rows={expected_rows}",
            "",
        )
    fields = output_scan.get("fields")
    stats = fields.get(field) if isinstance(fields, dict) else None
    if not isinstance(stats, dict):
        return "not_met", f"{field!r} is absent from all {rows} terminal row(s)", ""
    present = int(stats.get("present") or 0)
    numeric = int(stats.get("numeric") or 0)
    if present != rows:
        return (
            "not_met",
            f"{field!r} present on only {present}/{rows} terminal row(s)",
            "",
        )
    if numeric != rows:
        return (
            "not_met",
            f"{field!r} is numeric on only {numeric}/{rows} terminal row(s)",
            "",
        )
    if "min" not in stats or "max" not in stats:
        return (
            "unverifiable",
            "",
            f"terminal scan has no numeric range for {field!r}",
        )

    low, high = float(stats["min"]), float(stats["max"])
    rhs = float(target)
    if op == ">=":
        ok: bool | None = low >= rhs
    elif op == ">":
        ok = low > rhs
    elif op == "<=":
        ok = high <= rhs
    elif op == "<":
        ok = high < rhs
    elif op == "==":
        ok = low == rhs and high == rhs
    elif op == "!=":
        if high < rhs or low > rhs or (low == high and low != rhs):
            ok = True
        elif low == high == rhs:
            ok = False
        else:
            ok = None
    elif op == "~=":
        tol = float(tolerance or 0)
        ok = low >= rhs - tol and high <= rhs + tol
    else:
        ok = False
    if ok is None:
        return (
            "unverifiable",
            "",
            f"terminal min/max cannot prove every {field!r} value satisfies {op} {target}",
        )
    return (
        "met" if ok else "not_met",
        f"all {rows} terminal {field!r} value(s) span [{low:g}, {high:g}] vs {op} {target}",
        "",
    )


def _cmp(lhs: Any, op: str, rhs: Any, tol: Any = 0) -> bool:  # noqa: ANN401
    try:
        if op == "non_empty":
            return bool(lhs)
        if op == "~=":
            return abs(float(lhs) - float(rhs)) <= float(tol or 0)
        return bool(_OPS[op](lhs, rhs))
    except (KeyError, TypeError, ValueError):
        return False


def _num(v: Any) -> float:  # noqa: ANN401
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _finite_number(value: Any) -> float | None:  # noqa: ANN401
    """A finite JSON-style number; booleans and numeric strings are evidence errors."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


_OUTPUT_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    # ``transcript`` is user vocabulary, not a canonical stage role. Plain ASR's
    # documented/default output is ``pred_text``. Deliberately do NOT fall back to
    # ``text``: dataset readers use it for source/reference text, so doing so could
    # let a no-ASR pipeline satisfy a transcript-output criterion.
    "transcript": ("pred_text",),
}


def _completeness(  # noqa: C901, PLR0911, PLR0912, PLR0913 - explicit evidence-quality branches
    *,
    target: str | None,
    physical_field: str | None,
    produced_roles: set[str],
    produced_keys: set[str],
    per_item: list[dict[str, Any]],
    output_scan: dict[str, Any],
    expected_output_rows: Any,  # noqa: ANN401
) -> tuple[str, str, str]:
    """``(status, evidence, note)`` for "did the run actually produce this output?".

    Declarations establish that a pipeline *intends* to produce a role/key; they
    cannot establish that terminal rows contain values. A complete terminal scan
    is authoritative. Bounded row evidence remains useful for callers that supply
    it directly, but the run/reuse paths also provide ``output_scan`` so a bad row
    after the preview limit cannot be hidden.
    """
    if not target:
        return "unverifiable", "", "no output field/role named"

    scan_fields = output_scan.get("fields")
    if not isinstance(scan_fields, dict):
        scan_fields = {}
    observed = set(scan_fields)
    observed.update(k for row in per_item for k in row)
    field, mapped = _resolve_output_field(
        target,
        physical_field=physical_field,
        produced_roles=produced_roles,
        produced_keys=produced_keys,
        observed_fields=observed,
    )

    if output_scan:
        read_errors = int(output_scan.get("read_errors") or 0)
        malformed = int(output_scan.get("malformed_rows") or 0)
        blank = int(output_scan.get("blank_rows") or 0)
        rows = int(output_scan.get("valid_rows") or 0)
        status = str(output_scan.get("status") or "unavailable")

        if read_errors:
            return (
                "unverifiable",
                "",
                f"terminal output scan was {status}; {read_errors} file(s) could not be read",
            )
        if malformed or blank:
            invalid = malformed + blank
            return (
                "not_met",
                f"terminal output contains {invalid} malformed/blank row(s)",
                "",
            )
        if rows == 0:
            return (
                "unverifiable",
                "",
                f"terminal output scan was {status}; no output rows were available to verify",
            )
        expected_rows = _retained_count(expected_output_rows)
        if expected_output_rows is None:
            return (
                "unverifiable",
                "",
                "terminal rows were scanned, but no trustworthy serialized-row count "
                "was available to prove per-retained-item coverage",
            )
        if expected_rows is None:
            return (
                "unverifiable",
                "",
                f"invalid expected_output_rows evidence: {expected_output_rows!r}",
            )
        if rows != expected_rows:
            return (
                "not_met",
                f"terminal output has {rows} row(s), but expected_output_rows={expected_rows}",
                "",
            )
        if not mapped or not field:
            return (
                "unverifiable",
                "",
                f"output role {target!r} could not be mapped to a serialized terminal field",
            )
        if _is_nested_field(field) and output_scan.get("field_scope") == "top_level":
            return (
                "unverifiable",
                "",
                f"terminal scan covers top-level fields only, not nested field {field!r}",
            )

        stats = scan_fields.get(field)
        if not isinstance(stats, dict):
            return (
                "not_met",
                f"{field!r} is absent from all {rows} terminal row(s)",
                "",
            )
        present = int(stats.get("present") or 0)
        filled = int(stats.get("non_empty") or 0)
        if present != rows or filled != rows:
            where = "EMPTY in every row" if filled == 0 else f"only {filled}/{rows} row(s) carry a value"
            return (
                "not_met",
                f"{field!r} produced but {where} (of {rows} read)",
                "",
            )
        return (
            "met",
            f"{field!r} present and non-empty in all {rows} row(s) read",
            "",
        )

    # External callers can provide explicit row evidence without a filesystem scan.
    rows = len(per_item)
    if rows:
        expected_rows = _retained_count(expected_output_rows)
        if expected_output_rows is not None and expected_rows is None:
            return (
                "unverifiable",
                "",
                f"invalid expected_output_rows evidence: {expected_output_rows!r}",
            )
        if expected_rows is not None and rows != expected_rows:
            return (
                "not_met",
                f"per-item evidence has {rows} row(s), but expected_output_rows={expected_rows}",
                "",
            )
        if not mapped or not field:
            return (
                "unverifiable",
                "",
                f"output role {target!r} could not be mapped to a serialized evidence field",
            )
        present = sum(1 for row in per_item if field in row)
        filled = sum(1 for row in per_item if field in row and not _is_empty(row[field]))
        if present != rows or filled != rows:
            where = "EMPTY in every row" if filled == 0 else f"only {filled}/{rows} row(s) carry a value"
            return (
                "not_met",
                f"{field!r} produced but {where} (of {rows} read)",
                "",
            )
        return (
            "met",
            f"{field!r} present and non-empty in all {rows} row(s) read",
            "",
        )

    declared = set(produced_roles) | set(produced_keys)
    declared_names = {target}
    declared_names.update(_OUTPUT_ROLE_ALIASES.get(target, ()))
    if field:
        declared_names.add(field)
    if declared and declared.isdisjoint(declared_names):
        return (
            "not_met",
            f"{target!r} not in produced roles/keys",
            "",
        )
    if declared:
        return (
            "unverifiable",
            "",
            f"{target!r} was declared produced, but no output values were available to verify",
        )
    return (
        "unverifiable",
        "",
        "no observed output values or producer evidence",
    )


def _resolve_output_field(
    target: str,
    *,
    physical_field: str | None,
    produced_roles: set[str],
    produced_keys: set[str],
    observed_fields: set[str],
) -> tuple[str | None, bool]:
    """Resolve a semantic output target to a top-level serialized field.

    Exact physical fields win. The sole user-vocabulary alias is grounded in
    the ASR stage's documented default. Configured/non-default role-to-key
    mappings cannot be recovered from flat role/key sets, so those stay
    unverifiable rather than guessing.
    """
    available = set(produced_keys) | set(observed_fields)
    if physical_field and physical_field != target:
        # A semantic target and a differently named physical field are not
        # interchangeable merely because both strings appeared somewhere in the
        # evidence. Only a known alias may make that binding without a richer
        # role-to-field map. This prevents, for example, ``speaker_id`` from being
        # "proved" by a non-empty ``pred_text`` column.
        aliases = _OUTPUT_ROLE_ALIASES.get(target, ())
        if physical_field in aliases and physical_field in available:
            return physical_field, True
        return None, False
    if target in available:
        return target, True
    for alias in _OUTPUT_ROLE_ALIASES.get(target, ()):
        if alias in available:
            return alias, True
    if target in produced_roles and physical_field == target:
        return target, target in observed_fields
    return None, False


def _is_nested_field(field: str) -> bool:
    """Whether a field names content below the top-level manifest object."""
    return "." in field or "[]" in field


def _retained_count(value: Any) -> int | None:  # noqa: ANN401
    """A valid non-negative integral retained count, otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_empty(v: Any) -> bool:  # noqa: ANN401
    """A produced value that carries no content (None / blank string / empty container)."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict, tuple)):
        return len(v) == 0
    return False


def _summary(results: list[CriterionResult], overall: str, honesty: list[dict[str, Any]] | None = None) -> str:
    if not results and not honesty:
        return "acceptance: UNVERIFIABLE — no acceptance criteria to check (nothing was verified)"
    n_met = sum(1 for r in results if r.status == "met")
    lines = [f"acceptance: {overall.upper()} ({n_met}/{len(results)} criteria met)"]
    for r in results:
        if r.status != "met":
            detail = r.note or r.evidence
            lines.append(f"  - {r.id} [{r.severity}]: {r.status}" + (f" ({detail})" if detail else ""))
    for h in honesty or []:
        lines.append(f"  ! honesty[{h.get('code')}]: {h.get('message')}")
    if honesty:
        lines.append(
            "BLOCKED: the verified contract was weaker than what the user confirmed; re-confirm a new contract instead of relaxing a 'must'."
        )
    if overall == "not_met" and not honesty:
        lines.append(
            "options: adjust the recipe/thresholds and re-run, provide missing references, or relax a 'nice' "
            "criterion — never silently relax a 'must'."
        )
    return "\n".join(lines)
