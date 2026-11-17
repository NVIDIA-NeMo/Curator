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

"""Public reporting and acceptance verification."""

from __future__ import annotations

from typing import Any

from nemo_curator.stages.audio.audio_agent import _safety, _verb_inputs, profiler
from nemo_curator.stages.audio.audio_agent import (
    report as _report_data,
)
from nemo_curator.stages.audio.audio_agent.recipe import (
    Recipe,
)


def report(output: str, *, recipe: Recipe | dict[str, Any] | None = None, data: str | None = None) -> dict[str, Any]:  # noqa: C901, PLR0911, PLR0912
    """Post-hoc report from an output manifest/dir (counts rows vs input scale).

    Without ``recipe``, ``data`` is profiled directly for the input count. With
    a recipe, it is only an optional assertion matching the configured source.
    A supplied recipe also binds ``output`` to its declared terminal serializer
    and carries its frozen identity/acceptance contract into the report.
    """
    from nemo_curator.stages.audio.audio_agent import _verb_evidence, _verb_publish

    rec = _verb_inputs._as_recipe(recipe).freeze() if recipe is not None else None
    pviol = _safety.path_violations([output, data, *(_safety.recipe_path_params(rec) if rec is not None else [])])
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "violations": pviol}
    binding = _verb_inputs._dataset_binding(rec, data) if rec is not None else None
    if binding is not None and _verb_inputs._binding_blocks_execution(binding, data):
        return _safety.redact(_verb_inputs._binding_refusal(binding))
    dp_obj = (
        _verb_inputs._profile_binding(binding)
        if binding is not None
        else (profiler.profile_data(data) if data else None)
    )
    if dp_obj is not None and _verb_inputs._profile_error(dp_obj):
        if binding is not None:
            return _safety.redact(_verb_inputs._profile_refusal(binding, dp_obj))
        return _safety.redact(
            {
                "status": "refused",
                "reason": f"input data is unreadable or malformed: {_verb_inputs._profile_error(dp_obj)}",
                "data_profile": dp_obj.to_dict(),
            }
        )
    data_profile = dp_obj.to_dict() if dp_obj else None
    terminal_outputs: list[str] = []
    cardinality_proven = False
    if rec is not None:
        terminal_outputs, cardinality_proven = _verb_evidence._terminal_evidence_outputs(rec, [])
        if not terminal_outputs:
            return _safety.redact(
                {
                    "status": "refused",
                    "reason": ("the recipe declares no terminal output that can be bound to this post-hoc report"),
                    "recipe_id": rec.recipe_id,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict() if binding is not None else None,
                }
            )
        if not any(_verb_evidence._same_output_target(output, declared) for declared in terminal_outputs):
            return _safety.redact(
                {
                    "status": "refused",
                    "reason": ("reported output does not match the recipe's declared terminal output"),
                    "output": output,
                    "declared_terminal_outputs": terminal_outputs,
                    "recipe_id": rec.recipe_id,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict() if binding is not None else None,
                }
            )

    preview, output_scan = _verb_evidence._scan_terminal_output([output], limit=5)
    locate_status = str(output_scan.get("status") or "unavailable")
    if locate_status == "no_manifest":
        inventory = _verb_evidence._scan_output_inventory(output)
        if inventory.get("status") in {"complete", "empty"}:
            source_items = int((data_profile or {}).get("num_files", 0)) if data_profile is not None else None
            rpt = _report_data.build_run_report(
                recipe=rec or Recipe(),
                result_tasks=[],
                data_profile=data_profile,
                env_profile=profiler.probe_env().to_dict(),
                output_paths=[output],
                examples=[],
                next_action=("review the output inventory; row-level fields are unavailable for this output type"),
            )
            d = rpt.to_dict()
            d.update(
                {
                    "status": "ok",
                    "accepted": None,
                    "rejected": None,
                    "input_count": source_items,
                    "source_items": source_items,
                    "output_rows": None,
                    "output_files": int(inventory.get("files") or 0),
                    "output_scan": output_scan,
                    "output_inventory": inventory,
                }
            )
            if rec is not None and rec.acceptance_criteria:
                d["acceptance"] = verify(
                    list(rec.acceptance_criteria),
                    {
                        "per_item": [],
                        "output_scan": output_scan,
                        "metrics": {},
                        "expected_output_rows": None,
                        "retained": None,
                        "input_count": source_items,
                    },
                    recipe=rec,
                )
            if binding is not None:
                d["data_binding"] = binding.to_dict()
            return _verb_evidence._redact_result(d, rec)
    if locate_status in {"missing", "no_manifest", "unreadable", "unavailable"}:
        return _safety.redact(
            {
                "status": "error",
                "reason": f"terminal output could not be read as a manifest: {locate_status}",
                "output": output,
                "output_scan": output_scan,
                "recipe_id": rec.recipe_id if rec is not None else None,
                "config_hash": rec.config_hash if rec is not None else None,
                "data_binding": binding.to_dict() if binding is not None else None,
            }
        )

    accepted = int(output_scan.get("valid_rows") or 0)
    input_count = int((data_profile or {}).get("num_files", 0)) if data_profile is not None else None
    corrupt = bool(
        int(output_scan.get("read_errors") or 0)
        or int(output_scan.get("malformed_rows") or 0)
        or int(output_scan.get("blank_rows") or 0)
    )
    failures = (
        [
            {
                "code": "terminal_output_incomplete",
                "message": (
                    "terminal output contains unreadable, malformed, or blank "
                    "rows; accepted counts include valid JSON-object rows only"
                ),
                "output_scan": output_scan,
            }
        ]
        if corrupt
        else []
    )
    rpt = _report_data.build_run_report(
        recipe=rec or Recipe(),
        result_tasks=[],
        data_profile=data_profile,
        env_profile=profiler.probe_env().to_dict(),
        output_paths=[output],
        failures=failures,
        examples=preview,
        next_action="compare against expected retention; scale up or adjust thresholds",
    )
    d = rpt.to_dict()
    d["status"] = "error" if corrupt else "ok"
    d["accepted"] = accepted
    d["input_count"] = input_count
    d["source_items"] = input_count
    d["output_rows"] = accepted
    d["rejected"] = (
        _report_data.source_rejection_count(rec, input_count, accepted)
        if rec is not None and input_count is not None and cardinality_proven
        else None
    )
    d["output_scan"] = output_scan
    if rec is not None and rec.acceptance_criteria:
        d["acceptance"] = verify(
            list(rec.acceptance_criteria),
            {
                "per_item": [],
                "output_scan": output_scan,
                "metrics": _verb_evidence._aggregate_metrics(output_scan),
                "expected_output_rows": accepted if cardinality_proven else None,
                "retained": accepted,
                # A relative-yield denominator must come from actual source
                # evidence; substituting accepted would manufacture 100%.
                "input_count": _verb_publish._yield_input_count(rec, input_count, accepted),
            },
            recipe=rec,
        )
    if binding is not None:
        d["data_binding"] = binding.to_dict()
    return _verb_evidence._redact_result(d, rec)


def verify(
    acceptance_criteria: list[dict[str, Any]],
    evidence: dict[str, Any] | None = None,
    *,
    frozen_criteria: list[dict[str, Any]] | None = None,
    recipe: Recipe | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate acceptance criteria against gathered evidence -> AcceptanceReport (1A.1/1A.3).

    Deterministic verifier: it runs nothing itself, it judges the ``evidence`` the
    host assembled (from ``validate`` — ``produced_roles``/``produced_keys`` — and
    from ``smoke``/``run`` — ``metrics``/``per_item``/``retained``/``input_count``,
    plus optional ``unachievable_fields``). Returns per-criterion states
    (met / not_met / unverifiable / unachievable) and an ``overall`` that is
    ``met`` iff every ``must`` criterion is met — the anti-goalpost-moving gate.

    Pass the confirmed contract as ``frozen_criteria`` (or a ``recipe`` carrying
    ``acceptance_criteria``) to run the honesty guard (1A.3): if the criteria being
    verified are weaker than confirmed, it is flagged and ``overall`` is ``not_met``.
    """
    from nemo_curator.stages.audio.audio_agent.acceptance import parse_criteria
    from nemo_curator.stages.audio.audio_agent.acceptance import verify as _verify

    frozen = None
    if recipe is not None:
        frozen = parse_criteria(_verb_inputs._as_recipe(recipe).acceptance_criteria)
        if frozen_criteria is not None:
            explicit_frozen = parse_criteria(frozen_criteria)
            if [criterion.to_dict() for criterion in explicit_frozen] != [criterion.to_dict() for criterion in frozen]:
                msg = (
                    "frozen_criteria conflicts with recipe.acceptance_criteria; "
                    "provide one honesty source or make them identical"
                )
                raise ValueError(msg)
    elif frozen_criteria is not None:
        frozen = parse_criteria(frozen_criteria)
    report_obj = _verify(
        parse_criteria(acceptance_criteria),
        evidence if evidence is not None else {},
        frozen_criteria=frozen,
    )
    return _safety.redact(report_obj.to_dict())
