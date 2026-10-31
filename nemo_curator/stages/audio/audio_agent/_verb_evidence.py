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

"""Output evidence, acceptance checks, and inventory scanning."""

from __future__ import annotations

import contextlib
import json
import math
import os
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _safety
from nemo_curator.stages.audio.audio_agent.recipe import (
    OUTPUT_LOCATION_PARAMS,
    Recipe,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# How many output rows a success contract may read back as evidence. Enough to judge coverage
# on a real corpus; small enough that verifying a finished run never becomes the expensive part.
_EVIDENCE_ROWS = 2000


def _acceptance_result(  # noqa: PLR0913
    rec: Recipe,
    report: Any,  # noqa: ANN401
    produced_roles: list[str] | None,
    produced_keys: list[str] | None,
    outputs: list[str] | None = None,
    *,
    output_scan: dict[str, Any] | None = None,
    finalized_output: tuple[str, int] | None = None,
    source_recipe: Recipe | None = None,
) -> dict[str, Any]:
    """Verify the run's own success contract against the evidence it just produced.

    Recording the OUTCOME (not just the criteria) is what lets a reuse candidate be judged
    later without re-deriving whether it actually met its bar. A contract that could not be
    verified is reported as ``unverifiable`` with the reason: swallowing it would let a run
    claim success while its success bar was never checked.

    ``outputs`` are read back as row-level evidence so the contract judges the DATA, not the
    role labels ``validate`` declared. Declaration-level checking passes a field that every
    row left null -- the precise failure the contract exists to prevent.

    ``output_scan`` lets a caller that already read the output hand the result over instead of
    paying for a second pass. It is accepted only if it names the same terminal output this
    contract would have scanned; anything else is re-read, so passing the wrong scan costs
    time rather than judging one run's contract against another run's bytes.

    ``finalized_output`` is the successful driver's (manifest path, serialized row
    count) receipt after ALM merge and reconciliation. It proves coverage of those
    serialized rows; it does not establish retention relative to the source items.
    """
    from nemo_curator.stages.audio.audio_agent import _verb_publish, _verb_report

    if not rec.acceptance_criteria:
        return {}
    try:
        evidence_outputs, cardinality_proven = _terminal_evidence_outputs(
            rec,
            list(outputs or []),
        )
        per_item: list[dict[str, Any]] = []
        if not _needs_terminal_evidence(rec):
            output_scan = {}
        elif not _scan_covers(output_scan, evidence_outputs):
            per_item, output_scan = _scan_terminal_output(evidence_outputs, limit=0)
        expected_output_rows = int(getattr(report, "accepted", 0)) if cardinality_proven else None
        # ALM reconciliation removes origin stubs and snippets without usable audio.
        # Its completed finalizer proves serialized coverage for its own manifest,
        # independently of the number of in-memory tasks returned by the pipeline.
        if (
            finalized_output is not None
            and evidence_outputs
            and _same_output_target(finalized_output[0], evidence_outputs[-1])
            and isinstance(finalized_output[1], int)
            and not isinstance(finalized_output[1], bool)
            and finalized_output[1] >= 0
        ):
            expected_output_rows = finalized_output[1]
        return _verb_report.verify(
            list(rec.acceptance_criteria),
            {
                "produced_roles": list(produced_roles or []),
                "produced_keys": list(produced_keys or []),
                "per_item": per_item,
                "output_scan": output_scan,
                "metrics": _aggregate_metrics(output_scan),
                "expected_output_rows": expected_output_rows,
                "retained": int(getattr(report, "accepted", 0)),
                "input_count": _verb_publish._yield_input_count(
                    source_recipe or rec, int(getattr(report, "input_count", 0)), int(getattr(report, "accepted", 0))
                ),
            },
        )
    except Exception as e:  # noqa: BLE001 - report the failure, but do not fail a completed run
        return {"overall": "unverifiable", "reason": f"could not verify the success contract: {e}"}


def _scan_covers(output_scan: dict[str, Any] | None, evidence_outputs: list[str]) -> bool:
    """Whether an already-taken scan read the output this contract has to judge.

    The scan records the location it read, so this is a comparison rather than a promise the
    caller has to keep. Sharing a read is only safe if it is provably the same read.
    """
    if not output_scan:
        return False
    terminal = str(evidence_outputs[-1]) if evidence_outputs else ""
    return str(output_scan.get("terminal_output") or "") == terminal


def _needs_terminal_evidence(rec: Recipe) -> bool:
    """Whether this contract needs an exhaustive manifest readback."""
    from nemo_curator.stages.audio.audio_agent.acceptance import parse_criteria

    return any(
        criterion.type == "output_completeness"
        or criterion.check.get("scope") == "per_retained_item"
        or (
            criterion.type in {"quality_standard", "distribution"}
            and (criterion.check.get("scope") or "aggregate") == "aggregate"
        )
        for criterion in parse_criteria(rec.acceptance_criteria)
    )


def _terminal_evidence_outputs(
    rec: Recipe,
    reported_outputs: list[str],
) -> tuple[list[str], bool]:
    """Select the actual per-item serializer and whether it is provably 1:1.

    ``output_dir`` is a legacy verb argument and is currently not injected into
    the recipe. It may therefore appear in ``reported_outputs`` despite not being
    written by the pipeline. Prefer a recipe-declared serializer so acceptance
    never scans that phantom path.

    Final ``ManifestWriterStage`` and ``DocumentBatchJsonlWriterStage`` sinks
    have proven one-row-per-logical-returned-row contracts.
    ``SnippetManifestWriterStage`` intentionally skips origin stubs, and a
    writer followed by another stage need not have the final run's cardinality,
    so those cases remain explicitly unproven.

    A declared primary per-item manifest selects evidence when a writer saves
    several manifests. Its siblings remain required artifacts, not alternative
    evidence for the primary's fields.
    """
    from nemo_curator.stages.audio.audio_agent._capabilities import io_for, output_paths
    from nemo_curator.stages.audio.audio_agent._resolve import resolved_contract_for

    per_item_writers = {
        "DocumentBatchJsonlWriterStage",
        "ManifestWriterStage",
        "SnippetManifestWriterStage",
    }
    for index in range(len(rec.stages) - 1, -1, -1):
        stage = rec.stages[index]
        options = io_for(stage.ref).get("outputs", {})
        paths = output_paths(stage.ref, stage.params) if options else {}
        declared = {key: paths[key] for key, spec in options.items() if key in paths and spec.get("per_item")}
        if declared:
            selected = [path for key, path in declared.items() if options[key].get("primary")] or list(
                declared.values()
            )
            resolved = resolved_contract_for(stage.ref, stage.params)
            proven = (
                len(selected) == 1
                and index == len(rec.stages) - 1
                and resolved.resolved
                and resolved.contract.cardinality == "1:1"
            )
            return selected, proven
        short_ref = stage.ref.rsplit(".", 1)[-1]
        if short_ref not in per_item_writers:
            continue
        output = stage.params.get("output_path")
        if not isinstance(output, str) or not output:
            continue
        cardinality_proven = (
            short_ref in {"DocumentBatchJsonlWriterStage", "ManifestWriterStage"} and index == len(rec.stages) - 1
        )
        return [output], cardinality_proven

    declared_outputs = _recipe_outputs(rec, None)
    if declared_outputs:
        return [declared_outputs[-1]], False
    return ([str(reported_outputs[-1])] if reported_outputs else []), False


def _same_output_target(left: str, right: str) -> bool:
    """Canonical equality for local paths and lexical equality for remote URIs."""
    from nemo_curator.stages.audio.audio_agent.input_identity import canonical_source

    try:
        return canonical_source(left).rstrip("/") == canonical_source(right).rstrip("/")
    except (TypeError, ValueError):
        return False


def _estimate(data_profile: dict[str, Any] | None) -> dict[str, Any]:
    dp = data_profile or {}
    return {
        "input_count": dp.get("num_files", 0),
        "total_duration_sec": dp.get("total_duration_sec", 0.0),
        "note": "precise $/GPU-hour costing is deferred; time estimate comes from a prior smoke run",
    }


def _recipe_outputs(recipe: Recipe, output_dir: str | None) -> list[str]:
    """Every location the recipe writes to, in stage order.

    Scans the shared ``OUTPUT_LOCATION_PARAMS`` set, so intermediates a stage produces as a
    side effect (resampled dirs, per-speaker audio, RTTM dirs) are recorded rather than being
    invisible. ``path`` is kept as a legacy alias. ``raw_data_dir`` is a source
    cache/input, not a curated output. The verb-level ``output_dir`` argument is
    retained for API compatibility but intentionally never invents an output the
    executable recipe did not write.
    """

    from nemo_curator.stages.audio.audio_agent._capabilities import output_paths

    keys = [*sorted(OUTPUT_LOCATION_PARAMS - {"raw_data_dir"}), "path"]
    outs: list[str] = []
    for s in recipe.stages:
        declared = output_paths(s.ref, s.params)
        for key in dict.fromkeys([*keys, *declared]):
            v = declared.get(key, s.params.get(key))
            if isinstance(v, str) and v not in outs:
                outs.append(v)
    _ = output_dir  # legacy no-op; stage params are execution truth
    return outs


def _stage_metrics(results: list[Any] | None) -> dict[str, Any]:
    from nemo_curator.stages.audio.audio_agent.report import _dedup_stage_perf

    return _dedup_stage_perf(results or [])


def _result_rows(results: list[Any] | None) -> Iterator[dict[str, Any]]:
    """Yield logical rows from one-row tasks and batched document carriers.

    AudioTask-like results carry one mapping in ``task.data``. ``DocumentBatch``
    carries a pandas/Arrow table and exposes ``to_pandas``. Keeping this adapter
    structural avoids importing either dataframe dependency in the agent core
    while ensuring evidence counts and previews describe the same logical rows.
    """
    for task in results or []:
        data = getattr(task, "data", None)
        if isinstance(data, dict):
            yield data
            continue
        to_pandas = getattr(task, "to_pandas", None)
        if not callable(to_pandas):
            continue
        frame = to_pandas()
        to_dict = getattr(frame, "to_dict", None)
        if not callable(to_dict):
            continue
        records = to_dict(orient="records")
        if not isinstance(records, list):
            continue
        for row in records:
            if isinstance(row, dict):
                yield row


def _preview_roles(recipe: Recipe | None) -> dict[str, set[str]]:
    from nemo_curator.stages.audio.audio_agent._resolve import resolved_contract_for

    roles: dict[str, set[str]] = {}
    for stage in recipe.stages if recipe is not None else []:
        # Unavailable optional stages leave their string fields private.
        with contextlib.suppress(Exception):
            resolved = resolved_contract_for(stage.ref, stage.params)
            for key, role in resolved.contract.key_roles.items():
                roles.setdefault(key, set()).add(role)
    return roles


def _redact_result(payload: dict[str, Any], recipe: Recipe | None) -> dict[str, Any]:
    """Bind preview privacy to configured roles; unknown string fields stay private."""
    roles = _preview_roles(recipe)
    text_roles = {"text", "pred_text", "reference_text", "words", "alignment"}
    transcript_keys = {key for key, values in roles.items() if values & text_roles}
    safe_keys = {key for key, values in roles.items() if values and not values & (text_roles | {"unknown"})}
    result = _safety.redact(payload)

    def protect_previews(original: Any, redacted: Any) -> None:  # noqa: ANN401
        if isinstance(original, dict) and isinstance(redacted, dict):
            for key, value in original.items():
                if key == "examples" and isinstance(value, list):
                    redacted[key] = _safety.redact(value, transcript_keys=transcript_keys, safe_text_keys=safe_keys)
                else:
                    protect_previews(value, redacted.get(key))
        elif isinstance(original, list) and isinstance(redacted, list):
            for value, clean in zip(original, redacted, strict=True):
                protect_previews(value, clean)

    protect_previews(payload, result)
    return result


def _examples_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if _jsonable(v)} for row in rows[:limit]]


def _examples(results: list[Any] | None, *, limit: int) -> list[dict[str, Any]]:
    """Return at most ``limit`` JSON-safe logical rows from result carriers."""
    rows: list[dict[str, Any]] = []
    for row in _result_rows(results):
        rows.append(row)
        if len(rows) >= limit:
            break
    return _examples_from_rows(rows, limit=limit)


def _is_value_empty(v: Any) -> bool:  # noqa: ANN401
    """A produced value that carries no content (None / blank string / empty container)."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict, tuple)):
        return len(v) == 0
    # DataFrame rows use NaN for a column absent from one record. Treat that as
    # missing evidence without importing pandas/numpy into this module.
    try:
        return bool(v != v)  # noqa: PLR0124
    except (TypeError, ValueError):
        return False


def _empty_required_outputs(rec: Recipe, rows: list[dict[str, Any]]) -> list[str]:
    """Required output keys (from the recipe's ``output_completeness`` criteria) that are
    missing or empty in at least one bounded retained row.

    ``output_completeness`` defaults to per-retained-item semantics, so one
    missing/blank value is enough to withhold the smoke token. If retained rows
    cannot be inspected at all, a deterministic required output is unverifiable
    and therefore also cannot authorize a full run.
    """
    from nemo_curator.stages.audio.audio_agent.acceptance import _resolve_output_field, parse_criteria

    observed = {key for row in rows for key in row}
    empty: list[str] = []
    for criterion in parse_criteria(rec.acceptance_criteria):
        if criterion.type != "output_completeness" or criterion.severity != "must" or not criterion.is_deterministic:
            continue
        target = criterion.compiles_to if criterion.compiles_to != "producible_role" else None
        target = target or criterion.field_name
        if not target:
            continue
        key, mapped = _resolve_output_field(
            target,
            physical_field=criterion.field_name,
            produced_roles=set(),
            produced_keys=set(),
            observed_fields=observed,
        )
        if not mapped or not rows or any(key not in row or _is_value_empty(row.get(key)) for row in rows):
            empty.append(criterion.field_name or target)
    return empty


def _jsonable(v: Any) -> bool:  # noqa: ANN401
    """Preview-safe if a scalar, or a JSON-serializable nested dict/list/tuple.

    Scalars pass directly; nested containers (e.g. ``metrics`` with SQUIM or
    WER/CER scores, or ``segments``) are kept when they round-trip through
    ``json.dumps`` (``default=str`` matches the CLI/MCP emitter and tolerates a
    stray numpy scalar). Bare non-JSON objects (waveform tensors/arrays) are
    dropped. The report is still ``_safety.redact``-ed and capped by ``limit``.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return True
    if isinstance(v, (dict, list, tuple)):
        try:
            json.dumps(v, default=str)
        except (TypeError, ValueError, RecursionError):
            return False
        return True
    return False


def _generated_audio_paths(value: Any, audio_keys: set[str]) -> Iterator[str]:  # noqa: ANN401
    """Read contract-bound paths from ordinary rows and nested segment records."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in audio_keys and isinstance(item, str) and item:
                yield item
            else:
                yield from _generated_audio_paths(item, audio_keys)
    elif isinstance(value, list):
        for item in value:
            yield from _generated_audio_paths(item, audio_keys)


def _manifest_asset_dependencies(output: str, audio_keys: set[str]) -> dict[str, str]:
    """Bind serialized generated audio paths, including implicitly allocated files."""
    from nemo_curator.stages.audio.audio_agent import artifacts

    fs, files, status = _locate_manifest_files(output)
    if fs is None or status != "ok":
        message = f"manifest readback is {status}"
        raise ValueError(message)
    dependencies: dict[str, str] = {}
    for path in files:
        with fs.open(path, "rt", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                for asset in _generated_audio_paths(json.loads(raw), audio_keys):
                    if asset in dependencies:
                        continue
                    digest = artifacts.content_digest(asset)
                    if not digest:
                        message = f"generated audio is missing or unreadable: {asset!r}"
                        raise ValueError(message)
                    dependencies[asset] = digest
    return dependencies


def _locate_manifest_files(
    output: str,
) -> tuple[Any | None, list[str], str]:
    """Resolve a local/cloud manifest location through one fsspec filesystem.

    Paths returned here are filesystem-relative and must be opened through the
    returned ``fs`` object. A directly named file keeps the historical behavior
    of being scannable regardless of suffix; directories recursively include
    only ``.json``/``.jsonl`` files.
    """
    try:
        from fsspec.core import url_to_fs

        fs, resolved = url_to_fs(os.path.expanduser(output))
        if fs.isfile(resolved):
            return fs, [resolved], "ok"
        if fs.isdir(resolved):
            found = sorted(
                str(path)
                for path in fs.find(
                    resolved,
                    withdirs=False,
                    detail=False,
                )
                if str(path).endswith((".jsonl", ".json"))
            )
            return fs, found, "ok" if found else "no_manifest"
        return fs, [], "missing" if not fs.exists(resolved) else "no_manifest"
    except Exception:  # noqa: BLE001 - evidence readback must not fail a completed run
        return None, [], "unreadable"


def _scan_output_inventory(output: str) -> dict[str, Any]:
    """Inventory a non-manifest file/directory without claiming row evidence."""
    summary: dict[str, Any] = {
        "status": "unavailable",
        "output": output,
        "files": 0,
        "bytes": 0,
        "suffixes": {},
        "read_errors": 0,
    }
    try:
        from fsspec.core import url_to_fs

        fs, resolved = url_to_fs(os.path.expanduser(output))
        if fs.isfile(resolved):
            files = [resolved]
        elif fs.isdir(resolved):
            files = sorted(
                str(path)
                for path in fs.find(
                    resolved,
                    withdirs=False,
                    detail=False,
                )
            )
        else:
            summary["status"] = "missing"
            return summary
        for path in files:
            try:
                info = fs.info(path)
                summary["bytes"] += int(info.get("size") or 0)
                suffix = os.path.splitext(str(path))[1].lower() or "<none>"
                suffixes = summary["suffixes"]
                suffixes[suffix] = int(suffixes.get(suffix) or 0) + 1
            except Exception:  # noqa: BLE001 - one unreadable entry stays explicit
                summary["read_errors"] += 1
        summary["files"] = len(files)
        if summary["read_errors"]:
            summary["status"] = "partial"
        else:
            summary["status"] = "complete" if files else "empty"
    except Exception:  # noqa: BLE001 - backend failures are evidence, not crashes
        summary["status"] = "unreadable"
        summary["read_errors"] = 1
    return summary


def _count_output_rows(output: str) -> int:
    fs, files, _status = _locate_manifest_files(output)
    if fs is None:
        return 0
    total = 0
    for path in files:
        try:
            with fs.open(path, "rt", encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
        except Exception:  # noqa: BLE001, S112 - output inventory is best-effort
            continue
    return total


def _scan_terminal_output(  # noqa: C901, PLR0912, PLR0915 - streaming evidence states
    outputs: list[str],
    *,
    limit: int = _EVIDENCE_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan the terminal manifest completely while retaining a bounded row preview.

    The last declared output is authoritative. A missing/empty terminal product
    must never borrow valid rows from an earlier intermediate. The summary stores
    counts only (including per-field coverage), so exhaustive checking does not
    retain transcripts or scale memory with the corpus size.
    """
    terminal = str(outputs[-1]) if outputs else ""
    summary: dict[str, Any] = {
        "status": "unavailable",
        "terminal_output": terminal,
        "files": 0,
        "readable_files": 0,
        "valid_rows": 0,
        "malformed_rows": 0,
        "blank_rows": 0,
        "read_errors": 0,
        "field_scope": "top_level",
        "fields": {},
    }
    if not terminal:
        return [], summary

    fs, files, locate_status = _locate_manifest_files(terminal)
    summary["files"] = len(files)
    if locate_status != "ok" or fs is None:
        summary["status"] = locate_status
        if locate_status == "unreadable":
            summary["read_errors"] = 1
        return [], summary

    preview: list[dict[str, Any]] = []
    fields: dict[str, dict[str, Any]] = summary["fields"]
    for path in files:
        try:
            fh = fs.open(path, "rt", encoding="utf-8")
        except Exception:  # noqa: BLE001 - backend errors become explicit evidence state
            summary["read_errors"] += 1
            continue
        summary["readable_files"] += 1
        try:
            with fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        summary["blank_rows"] += 1
                        continue
                    try:
                        obj = json.loads(line)
                    except (TypeError, ValueError):
                        summary["malformed_rows"] += 1
                        continue
                    if not isinstance(obj, dict):
                        summary["malformed_rows"] += 1
                        continue

                    summary["valid_rows"] += 1
                    if len(preview) < limit:
                        preview.append(obj)
                    for key, value in obj.items():
                        stat = fields.setdefault(
                            str(key),
                            {
                                "present": 0,
                                "non_empty": 0,
                                "numeric": 0,
                            },
                        )
                        stat["present"] += 1
                        if not _is_value_empty(value):
                            stat["non_empty"] += 1
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                        ):
                            number = float(value)
                            stat["numeric"] += 1
                            stat["sum"] = stat.get("sum", 0.0) + number
                            stat["min"] = min(number, stat.get("min", number))
                            stat["max"] = max(number, stat.get("max", number))
        except Exception:  # noqa: BLE001 - remote iteration can raise backend-specific errors
            summary["read_errors"] += 1

    if summary["read_errors"] and not summary["readable_files"]:
        summary["status"] = "unreadable"
    elif summary["read_errors"] or summary["malformed_rows"] or summary["blank_rows"]:
        summary["status"] = "partial"
    elif not summary["valid_rows"]:
        summary["status"] = "empty"
    else:
        summary["status"] = "complete"
    rows = int(summary["valid_rows"])
    for stat in fields.values():
        if rows > 0 and int(stat.get("numeric") or 0) == rows:
            stat["mean"] = float(stat.get("sum") or 0.0) / rows
    return preview, summary


def _aggregate_metrics(output_scan: dict[str, Any]) -> dict[str, float]:
    """Return complete-corpus means for wholly numeric terminal fields.

    Aggregate acceptance is evidence-backed only when every serialized row was
    read successfully and every row carries a finite number for that field.
    Partial scans and mixed/missing values stay unverifiable.
    """
    if str(output_scan.get("status") or "") != "complete":
        return {}
    rows = int(output_scan.get("valid_rows") or 0)
    if rows <= 0:
        return {}
    metrics: dict[str, float] = {}
    for field, field_stats in dict(output_scan.get("fields") or {}).items():
        if (
            int(field_stats.get("present") or 0) == rows
            and int(field_stats.get("numeric") or 0) == rows
            and isinstance(field_stats.get("mean"), (int, float))
            and not isinstance(field_stats.get("mean"), bool)
            and math.isfinite(float(field_stats["mean"]))
        ):
            metrics[str(field)] = float(field_stats["mean"])
    return metrics
