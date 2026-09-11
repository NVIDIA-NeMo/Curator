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

"""Typed RunReport — the evidence artifact (full PRD field set).

Carries recipe_id + config_hash, Curator version, dependency mode, data profile,
per-stage metrics, accepted/rejected + per-filter counts, failure reasons +
examples, output paths, a logs pointer, and a next-action. Built from the tasks
a pipeline returns; fixes the fan-out double-count by aggregating one
StagePerfStats per (stage, source) rather than summing over every child task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from nemo_curator.audio_agent.contracts import _clean


@dataclass
class RunReport:
    """Evidence-backed record of a run (or smoke)."""

    recipe_id: str | None = None
    config_hash: str | None = None
    curator_version: str = ""
    dependency_mode: str = ""
    data_profile: dict[str, Any] | None = None
    input_count: int = 0
    accepted: int = 0
    rejected: int | None = 0
    # Explicit cardinality vocabulary. ``accepted``/``input_count`` remain for
    # compatibility, while these fields prevent fan-out output rows from being
    # mistaken for retained source items.
    source_items: int | None = None
    output_rows: int | None = None
    # Rows found by READING BACK the written output, against ``output_rows`` counted from the
    # tasks still in memory. The two answer different questions and a disagreement is itself
    # the finding: the second says what the pipeline handed on, the first what the user has.
    output_rows_written: int | None = None
    sparse_fields: list[dict[str, Any]] = field(default_factory=list)
    """Written fields that carry no value in some rows, worst first.

    A row count flatters an output whose rows are mostly blank. One real run wrote a 4-row
    manifest holding 21 ALM windows and reported "4 rows"; three of those rows had an empty
    ``filtered_windows`` and 20 of the 21 windows held a single speaker, and it was announced
    as complete. The per-field fill counts that contradict this were already being computed
    -- but only for a run that declared acceptance criteria, so the runs least likely to be
    checked were also the only ones reporting nothing.
    """
    cardinality_proven: bool = False
    per_filter_counts: dict[str, Any] = field(default_factory=dict)
    per_stage_metrics: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    logs_pointer: str = ""
    elapsed_sec: float = 0.0
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


def stage_duration_sec(per_stage: dict[str, Any], runtime_name: str) -> float:
    """Seconds one stage spent, from the metrics THIS module writes; 0 when unmeasured.

    The single reading convention for ``per_stage_metrics``. Two callers used to answer this
    question two ways -- one matching any metric containing "time", the other only ``process_time``
    under the stage's *class* name. Runtime names are a stage's own ``name`` field, which is often
    neither (``manifest_writer``, ``ASR_inference``), so the stricter reader silently found nothing
    for most of the catalogue. Keyed here, beside the writer, so a change to either side is one edit.
    """
    metrics = per_stage.get(runtime_name) or {}
    seconds = 0.0
    for metric, agg in metrics.items():
        if "time" in metric and isinstance(agg, dict):
            seconds = max(seconds, float(agg.get("sum") or 0.0))
    return round(seconds, 3)


def _dedup_stage_perf(tasks: list[Any]) -> dict[str, Any]:
    """Aggregate per-stage metrics WITHOUT the fan-out double-count.

    The backend appends the same StagePerfStats to every child of a fan-out
    batch, so summing over all output tasks over-counts. We de-duplicate by the
    identity of each StagePerfStats object before aggregating.

    Underscored but NOT private: imported by ``verbs``, which reports the same per-stage
    numbers straight off a run rather than from a stored report.
    """
    import numpy as np

    seen: set[int] = set()
    by_stage: dict[str, dict[str, list[float]]] = {}
    for task in tasks or []:
        for perf in getattr(task, "_stage_perf", None) or []:
            if id(perf) in seen:
                continue
            seen.add(id(perf))
            metrics = by_stage.setdefault(perf.stage_name, {})
            for name, value in perf.items():
                metrics.setdefault(name, []).append(float(value))
    out: dict[str, Any] = {}
    for stage, metrics in by_stage.items():
        out[stage] = {}
        for name, vals in metrics.items():
            arr = np.asarray(vals, dtype=float)
            out[stage][name] = {
                "sum": float(arr.sum()),
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
            }
    return out


def _filter_counts(per_stage: dict[str, Any]) -> dict[str, Any]:
    """Extract accept/reject accounting from stages that log it (custom.* metrics)."""
    counts: dict[str, Any] = {}
    for stage, metrics in per_stage.items():
        entry = {}
        for key in ("custom.input_count", "custom.output_count", "custom.filtered_count"):
            if key in metrics:
                entry[key.replace("custom.", "")] = metrics[key]["sum"]
        if entry:
            counts[stage] = entry
    return counts


def _row_count(tasks: list[Any] | None) -> int:
    """Count ROWS across result tasks, not the number of tasks.

    An ``AudioTask`` holds one item (``num_items`` == 1), but a ``DocumentBatch`` holds a
    whole table -- so ``len(tasks)`` under-counts any pipeline that ends in a batch type
    (e.g. after ``AudioToDocumentStage``). Summing ``num_items`` gives the true row count.

    Underscored but NOT private: imported by ``verbs``, so this is the one definition of
    "how many rows came out" behind both the report and the run's own evidence. Two of them
    would let a report and the run that produced it disagree on the same number.
    """
    return sum(int(getattr(t, "num_items", 1) or 0) for t in (tasks or []))


def rows_written_in(output_scan: dict[str, Any] | None) -> int | None:
    """Rows read back from the written output, or ``None`` when it could not be read.

    Zero is a claim -- "the output is there and holds nothing" -- and must not be how an
    unreadable or undeclared output is reported, because that reads as a run that produced
    nothing rather than one whose result was never inspected.
    """
    scan = output_scan or {}
    if int(scan.get("readable_files") or 0) <= 0:
        return None
    return int(scan.get("valid_rows") or 0)


def sparse_fields_in(output_scan: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Which written fields are blank in some rows, worst first.

    Measures every field against the output's total row count, so a field MISSING from a row
    counts the same as one present but empty -- from the reader's side both mean "no value
    here", and distinguishing them would only invite the reply that the field was technically
    present. Fields full in every row are omitted: they are the unremarkable case, and listing
    them would bury the one that isn't.

    The list is bounded by how wide the manifest schema is, not by corpus size, so it stays
    small in a persisted run record.
    """
    scan = output_scan or {}
    rows = int(scan.get("valid_rows") or 0)
    fields = scan.get("fields")
    if rows <= 0 or not isinstance(fields, dict):
        return []
    out: list[dict[str, Any]] = []
    for name, stat in fields.items():
        if not isinstance(stat, dict):
            continue
        filled = min(int(stat.get("non_empty") or 0), rows)
        if filled < rows:
            out.append({"field": str(name), "rows": rows, "non_empty": filled, "empty": rows - filled})
    out.sort(key=lambda entry: (-int(entry["empty"]), str(entry["field"])))
    return out


def build_run_report(  # noqa: PLR0913 - a report intentionally gathers many fields
    *,
    recipe: Any,  # noqa: ANN401
    result_tasks: list[Any] | None,
    data_profile: dict[str, Any] | None = None,
    env_profile: dict[str, Any] | None = None,
    output_paths: list[str] | None = None,
    elapsed_sec: float = 0.0,
    failures: list[dict[str, Any]] | None = None,
    logs_pointer: str = "",
    next_action: str = "",
    examples: list[dict[str, Any]] | None = None,
) -> RunReport:
    """Assemble a RunReport from a pipeline's returned tasks and the run context."""
    result_tasks = result_tasks or []
    per_stage = _dedup_stage_perf(result_tasks)
    accepted = _row_count(result_tasks)  # rows, not task count (DocumentBatch holds a table)
    input_count = int((data_profile or {}).get("num_files", 0)) or accepted
    dep_mode = ""
    if env_profile:
        dep_mode = "cuda12" if env_profile.get("has_gpu") else "cpu"

    return RunReport(
        recipe_id=getattr(recipe, "recipe_id", None),
        config_hash=getattr(recipe, "config_hash", None),
        curator_version=(env_profile or {}).get("curator_version", ""),
        dependency_mode=dep_mode,
        data_profile=data_profile,
        input_count=input_count,
        accepted=accepted,
        rejected=max(0, input_count - accepted),
        source_items=input_count,
        output_rows=accepted,
        per_filter_counts=_filter_counts(per_stage),
        per_stage_metrics=per_stage,
        failure_reasons=failures or [],
        examples=examples or [],
        output_paths=output_paths or [],
        logs_pointer=logs_pointer,
        elapsed_sec=round(elapsed_sec, 3),
        next_action=next_action,
    )
