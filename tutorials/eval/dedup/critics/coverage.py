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

"""
Veto-only coverage critic for the main judge's "X can replace Y" calls.

`CoveragePrepareStage` picks the rows worth reviewing, a DataDesigner column asks the
model to point at spans carrying uncovered meaning, and `CoverageApplyStage` turns that
answer into a final decision. The model never writes the decision itself: it only cites
span IDs and a conflict type, and the code derives what changes. A critic answer that
cannot be proven from the span packet leaves the main decision unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import data_designer.config as dd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

from .models import DECISION_ARROW_TYPE, Decision
from .spans import DEFAULT_MAX_EVIDENCE_CHARS, PACKET_ARROW_TYPE, SpanPacket
from .stage_utils import NO_ANSWER_REASON, replace_columns

_PROMPTS = Path(__file__).resolve().parent / "prompts"

# conflict -> (primary_material_difference, primary_risk_factor) written into a vetoed decision
CONFLICTS = {
    "IDENTITY_CONFLICT": ("document_identity_change", "template_slot_collision"),
    "STATE_CONFLICT": ("other_material", "identifier_underweighting"),
    "POLICY_CONFLICT": ("legal_context_change", "legal_context_collision"),
    "ROLE_CONFLICT": ("page_role_change", "page_role_collision"),
    "MEMBERSHIP_CONFLICT": ("result_set_change", "list_snapshot_collision"),
}
# action -> replacement directions the veto turns off
_REJECTED_DIRECTIONS = {
    "REJECT_A_REPLACES_B": ("a_can_replace_b",),
    "REJECT_B_REPLACES_A": ("b_can_replace_a",),
    "REJECT_BOTH": ("a_can_replace_b", "b_can_replace_a"),
    "REJECT_EMPTY_ANCHOR": ("a_can_replace_b", "b_can_replace_a"),
}

_EVIDENCE_ARROW_TYPE = pa.list_(
    pa.struct([("side", pa.string()), ("start_char", pa.int64()), ("end_char", pa.int64()), ("quote", pa.string())])
)


class CoverageReview(BaseModel):
    """The model's structured answer. Loss IDs may be empty; context IDs may not."""

    model_config = ConfigDict(extra="forbid")

    a_loss_span_id: str
    b_loss_span_id: str
    a_context_span_id: str = Field(min_length=1)
    b_context_span_id: str = Field(min_length=1)
    conflict: Literal[
        "NONE", "IDENTITY_CONFLICT", "STATE_CONFLICT", "POLICY_CONFLICT", "ROLE_CONFLICT", "MEMBERSHIP_CONFLICT"
    ]
    overlap_basis: Literal["RETAINED_CONTENT", "INTERFACE_ONLY", "UNCERTAIN"]
    shared_anchor_ids: list[str]
    explanation: str = Field(min_length=1)

    @field_validator("shared_anchor_ids")
    @classmethod
    def _anchors_are_unique(cls, ids: list[str]) -> list[str]:
        if len(ids) != len(set(ids)):
            msg = "shared_anchor_ids must be unique"
            raise ValueError(msg)
        return ids


def coverage_column(model_alias: str) -> dd.LLMStructuredColumnConfig:
    """The DataDesigner column that produces `coverage_review` for rows marked `coverage_should_run`."""
    return dd.LLMStructuredColumnConfig(
        name="coverage_review",
        model_alias=model_alias,
        prompt=(_PROMPTS / "coverage_pair.jinja").read_text(encoding="utf-8"),
        system_prompt=(_PROMPTS / "coverage_system.jinja").read_text(encoding="utf-8"),
        output_format=CoverageReview,
        skip=dd.SkipConfig(when="{{ not coverage_should_run }}"),
    )


@dataclass
class CoveragePrepareStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Decide which rows the critic reviews and render their span packet for the prompt."""

    source_judge: str
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    name: str = "coverage_prepare"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "text_a", "text_b", "semantic_diff", "truncated", self.source_judge]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["coverage_should_run", "coverage_reason", "coverage_packet"]

    def _triage(self, record: dict) -> tuple[str, SpanPacket | None]:
        """Return why a row is or isn't reviewed, plus its span packet when one could be built."""
        try:
            main = Decision.from_judge(record.get(self.source_judge))
        except ValueError:
            return "PRESERVE_INVALID_MAIN", None
        if main.inconsistencies():
            return "PRESERVE_INVALID_MAIN", None
        try:
            packet = SpanPacket.from_record(record, max_evidence_chars=self.max_evidence_chars)
        except ValueError:
            return "PRESERVE_INVALID_PACKET", None

        if "yes" not in (main.a_can_replace_b, main.b_can_replace_a):
            return "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED", packet
        if packet.complete and record["text_a"].strip() and record["text_a"] == record["text_b"]:
            return "PRESERVE_COMPLETE_EXACT_INPUT", packet
        return "REVIEW_POSITIVE_DIRECTIONS_ONLY", packet

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        clashes = {"coverage_should_run", "coverage_reason", "coverage_packet"} & set(table.column_names)
        if clashes:
            msg = f"{self.name} would overwrite input columns: {sorted(clashes)}"
            raise ValueError(msg)

        triage = [self._triage(record) for record in table.to_pylist()]
        reasons = [reason for reason, _ in triage]
        table = replace_columns(
            table,
            {
                "coverage_should_run": pa.array(
                    [reason == "REVIEW_POSITIVE_DIRECTIONS_ONLY" for reason in reasons], type=pa.bool_()
                ),
                "coverage_reason": pa.array(reasons, type=pa.string()),
                "coverage_packet": pa.array(
                    [packet.for_prompt() if packet else None for _, packet in triage], type=PACKET_ARROW_TYPE
                ),
            },
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=table, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )


@dataclass
class CoverageApplyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Turn each `coverage_review` into `final_decision`, `coverage_action` and `coverage_evidence`."""

    source_judge: str
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    name: str = "coverage_apply"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "pair_id",
            "text_a",
            "text_b",
            "semantic_diff",
            "truncated",
            self.source_judge,
            "coverage_should_run",
            "coverage_reason",
            "coverage_review",
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["coverage_action", "coverage_reason", "coverage_evidence", "final_decision"]

    @staticmethod
    def _check_proof(review: CoverageReview, packet: SpanPacket) -> tuple[str, list[dict]]:  # noqa: C901, PLR0912
        """Check the review's cited spans against the packet; return the action it supports and its evidence."""
        shared_ids = {span_id for span_id, span in packet.spans.items() if span.kind == "SHARED"}
        if not set(review.shared_anchor_ids) <= shared_ids:
            msg = "anchors must be existing shared span IDs"
            raise ValueError(msg)

        evidence, sides_with_unique_span = [], set()
        for side, loss_id, context_id in (
            ("A", review.a_loss_span_id, review.a_context_span_id),
            ("B", review.b_loss_span_id, review.b_context_span_id),
        ):
            for span_id in dict.fromkeys([loss_id, context_id]):
                if not span_id:
                    continue
                span = packet.spans.get(span_id)
                if span is None:
                    msg = f"unknown span ID {span_id!r}"
                    raise ValueError(msg)
                is_own_unique = span.kind == f"{side}_ONLY"
                if not (is_own_unique or (span_id != loss_id and span.kind == "SHARED")):
                    msg = f"wrong side or non-unique loss: {span_id}"
                    raise ValueError(msg)
                evidence.append(span.excerpt(side).as_evidence())
                if is_own_unique:
                    sides_with_unique_span.add(side)

        a_loss, b_loss = bool(review.a_loss_span_id), bool(review.b_loss_span_id)
        if review.conflict != "NONE":
            if not sides_with_unique_span:
                msg = "conflict requires a unique delta witness"
                raise ValueError(msg)
            action = "REJECT_BOTH"
        elif review.overlap_basis == "UNCERTAIN":
            action = "ABSTAIN"
        elif review.overlap_basis == "INTERFACE_ONLY" and (a_loss or b_loss):
            if not shared_ids or set(review.shared_anchor_ids) != shared_ids:
                msg = "empty-anchor veto requires all shared spans"
                raise ValueError(msg)
            action = "REJECT_EMPTY_ANCHOR"
        else:
            action = {
                (False, False): "KEEP_MAIN",
                (True, False): "REJECT_B_REPLACES_A",
                (False, True): "REJECT_A_REPLACES_B",
                (True, True): "REJECT_BOTH",
            }[(a_loss, b_loss)]

        if action not in {"KEEP_MAIN", "ABSTAIN"} and not packet.complete:
            msg = "veto requires complete, untruncated evidence"
            raise ValueError(msg)
        return action, evidence

    @staticmethod
    def _apply_action(main: Decision, review: CoverageReview, action: str) -> tuple[Decision, str, str]:
        """Return the final decision, the action actually taken, and the reason for it."""
        if action == "KEEP_MAIN":
            return main, action, "NO_SUPPORTED_OBJECTION_KEEP_MAIN"
        if action == "ABSTAIN":
            unresolved = dict.fromkeys(Decision.model_fields, "unresolved")
            unresolved.update(primary_risk_factor="extraction_or_payload_limit", confidence_tier="low")
            return main.model_copy(update=unresolved), action, "EXPLICIT_CRITIC_ABSTENTION"
        if action == "REJECT_EMPTY_ANCHOR" and main.relation_type != "containment":
            return main, "KEEP_MAIN", "EMPTY_ANCHOR_VETO_OUTSIDE_CONTAINMENT_SCOPE"

        rejected = _REJECTED_DIRECTIONS[action]
        if not any(getattr(main, direction) == "yes" for direction in rejected):
            return main, "KEEP_MAIN", "OBJECTION_ONLY_TO_ALREADY_UNSAFE_DIRECTION"

        update = {"a_can_replace_b": main.a_can_replace_b, "b_can_replace_a": main.b_can_replace_a}
        update.update(dict.fromkeys(rejected, "no"))
        primary, risk = CONFLICTS.get(review.conflict, ("other_material", "boilerplate_dominated_similarity"))
        if "yes" in (update["a_can_replace_b"], update["b_can_replace_a"]):
            update.update(
                relation_type="containment",
                primary_material_difference="main_content_addition_deletion",
                primary_risk_factor="containment_asymmetry",
            )
        else:
            update.update(
                relation_type="version_related" if review.conflict == "STATE_CONFLICT" else "related_non_duplicate",
                primary_material_difference=primary,
                primary_risk_factor=risk,
            )
        update.update(material_difference="major", confidence_tier="medium")

        final = main.model_copy(update=update)
        if final.inconsistencies():
            msg = f"veto produced an incoherent decision: {final.inconsistencies()}"
            raise ValueError(msg)
        return final, action, "SUPPORTED_DIRECTIONAL_VETO"

    def _apply(self, record: dict) -> tuple[str, str, list[dict], Decision | None]:
        reason = record["coverage_reason"]
        try:
            main = Decision.from_judge(record.get(self.source_judge))
        except ValueError:
            return "SKIP", reason, [], None
        if not record["coverage_should_run"]:
            return "SKIP", reason, [], main
        if record.get("coverage_review") is None:
            return "UNVALIDATED_KEEP_MAIN", NO_ANSWER_REASON, [], main

        try:
            review = CoverageReview.model_validate(record["coverage_review"])
            packet = SpanPacket.from_record(record, max_evidence_chars=self.max_evidence_chars)
            action, evidence = self._check_proof(review, packet)
            final, action, reason = self._apply_action(main, review, action)
        except ValueError as error:
            return "UNVALIDATED_KEEP_MAIN", f"INVALID_CRITIC_OUTPUT: {str(error)[:300]}", [], main
        return action, reason, evidence, final

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        results = [self._apply(record) for record in table.to_pylist()]

        table = replace_columns(
            table,
            {
                "coverage_action": pa.array([action for action, *_ in results], type=pa.string()),
                "coverage_reason": pa.array([reason for _, reason, *_ in results], type=pa.string()),
                "coverage_evidence": pa.array([evidence for _, _, evidence, _ in results], type=_EVIDENCE_ARROW_TYPE),
                "final_decision": pa.array(
                    [final.model_dump() if final else None for *_, final in results], type=DECISION_ARROW_TYPE
                ),
            },
            drop=["coverage_packet"],
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=table, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )
