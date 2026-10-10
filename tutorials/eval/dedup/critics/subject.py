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
Subject critic and verifier, run after the coverage critic.

The critic asks the model whether the same claim (who is liable, which service a rule covers,
which record failed) is bound to a different named target on each side. If it proposes one,
the verifier re-reads both full documents to confirm the two targets are specific names. Only a
verified proposal changes `final_decision`; every other outcome keeps the coverage result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import data_designer.config as dd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

from .models import DECISION_ARROW_TYPE, Decision
from .spans import DEFAULT_MAX_EVIDENCE_CHARS, PACKET_ARROW_TYPE, SpanPacket
from .stage_utils import NO_ANSWER_REASON, replace_columns

_PROMPTS = Path(__file__).resolve().parent / "prompts"

# Only these bindings are within the critic's authority to veto.
VETO_BINDINGS = {"LIABILITY_PARTY", "POLICY_SERVICE", "FAILED_OBJECT"}

_TARGET_ARROW_TYPE = pa.struct([("span_id", pa.string()), ("quote", pa.string())])
_SIDE_ARROW_TYPE = pa.struct([("subject", _TARGET_ARROW_TYPE), ("predicate", _TARGET_ARROW_TYPE)])
_CANDIDATE_ARROW_TYPE = pa.struct(
    [("binding_type_hypothesis", pa.string()), ("a", _SIDE_ARROW_TYPE), ("b", _SIDE_ARROW_TYPE)]
)
_EVIDENCE_ARROW_TYPE = pa.list_(
    pa.struct(
        [
            ("span_id", pa.string()),
            ("quote", pa.string()),
            ("side", pa.string()),
            ("role", pa.string()),
            ("start_char", pa.int64()),
            ("end_char", pa.int64()),
        ]
    )
)


class SubjectReview(BaseModel):
    """The critic's answer. Span IDs may be empty when no comparison is supported."""

    model_config = ConfigDict(extra="forbid")

    a_subject_span_id: str
    a_predicate_span_id: str
    b_subject_span_id: str
    b_predicate_span_id: str
    binding_type: Literal[
        "NONE", "LIABILITY_PARTY", "POLICY_SERVICE", "ACCESS_TARGET", "FAILED_OBJECT", "RECORD_SUBJECT"
    ]
    target_relation: Literal["SAME", "DIFFERENT", "UNCERTAIN", "NOT_APPLICABLE"]
    explanation: str = Field(min_length=1)


class SubjectVerification(BaseModel):
    """The verifier's answer about the two selected subjects."""

    model_config = ConfigDict(extra="forbid")

    a_subject_kind: Literal["NAMED_ACTUAL_TARGET", "GENERIC_ROLE_OR_OBJECT", "UI_LABEL_OR_INSTRUCTION", "UNSUPPORTED"]
    b_subject_kind: Literal["NAMED_ACTUAL_TARGET", "GENERIC_ROLE_OR_OBJECT", "UI_LABEL_OR_INSTRUCTION", "UNSUPPORTED"]
    comparison: Literal["SUPPORTED_DIFFERENT_NAMED_TARGETS", "UNSUPPORTED_COMPARISON", "UNCERTAIN"]
    explanation: str = Field(min_length=1)


def subject_column(model_alias: str) -> dd.LLMStructuredColumnConfig:
    """The DataDesigner column that produces `subject_review` for rows marked `subject_should_run`."""
    return dd.LLMStructuredColumnConfig(
        name="subject_review",
        model_alias=model_alias,
        prompt=(_PROMPTS / "subject_pair.jinja").read_text(encoding="utf-8"),
        system_prompt=(_PROMPTS / "subject_system.jinja").read_text(encoding="utf-8"),
        output_format=SubjectReview,
        skip=dd.SkipConfig(when="{{ not subject_should_run }}"),
    )


def verifier_column(model_alias: str) -> dd.LLMStructuredColumnConfig:
    """The DataDesigner column that produces `subject_verifier_review` for rows marked `subject_verifier_should_run`."""
    return dd.LLMStructuredColumnConfig(
        name="subject_verifier_review",
        model_alias=model_alias,
        prompt=(_PROMPTS / "subject_verifier_pair.jinja").read_text(encoding="utf-8"),
        system_prompt=(_PROMPTS / "subject_verifier_system.jinja").read_text(encoding="utf-8"),
        output_format=SubjectVerification,
        skip=dd.SkipConfig(when="{{ not subject_verifier_should_run }}"),
    )


@dataclass
class SubjectPrepareStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Decide which rows the subject critic reviews, based on the coverage `final_decision`."""

    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    name: str = "subject_prepare"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "text_a", "text_b", "semantic_diff", "truncated", "final_decision"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["subject_should_run", "subject_reason", "subject_packet"]

    def _triage(self, record: dict) -> tuple[str, SpanPacket | None]:  # noqa: PLR0911
        """Return why a row is or isn't reviewed, plus its span packet when one could be built."""
        try:
            base = Decision.model_validate(record["final_decision"])
        except ValueError:
            return "PRESERVE_INVALID_MAIN", None
        if base.inconsistencies():
            return "PRESERVE_INVALID_MAIN", None
        try:
            packet = SpanPacket.from_record(record, max_evidence_chars=self.max_evidence_chars)
        except ValueError:
            return "PRESERVE_INVALID_PACKET", None

        kinds = {span.kind for span in packet.spans.values()}
        if "yes" not in (base.a_can_replace_b, base.b_can_replace_a):
            return "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED", packet
        if packet.complete and record["text_a"].strip() and record["text_a"] == record["text_b"]:
            return "PRESERVE_COMPLETE_EXACT_INPUT", packet
        if not {"A_ONLY", "B_ONLY"} <= kinds:
            return "PRESERVE_WITHOUT_BILATERAL_UNIQUE_SUBJECTS", packet
        if packet.truncated:
            return "PRESERVE_INCOMPLETE_SUBJECT_CONTEXT", packet
        return "REVIEW_BILATERAL_SUBJECTS", packet

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        clashes = {"subject_should_run", "subject_reason", "subject_packet"} & set(table.column_names)
        if clashes:
            msg = f"{self.name} would overwrite input columns: {sorted(clashes)}"
            raise ValueError(msg)

        triage = [self._triage(record) for record in table.to_pylist()]
        reasons = [reason for reason, _ in triage]
        table = replace_columns(
            table,
            {
                "subject_should_run": pa.array(
                    [reason == "REVIEW_BILATERAL_SUBJECTS" for reason in reasons], type=pa.bool_()
                ),
                "subject_reason": pa.array(reasons, type=pa.string()),
                "subject_packet": pa.array(
                    [packet.for_prompt() if packet else None for _, packet in triage], type=PACKET_ARROW_TYPE
                ),
            },
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=table, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )


@dataclass
class SubjectApplyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Turn each `subject_review` into a fixed proposal for the verifier, or a reason to keep coverage."""

    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    name: str = "subject_apply"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "pair_id",
            "text_a",
            "text_b",
            "semantic_diff",
            "truncated",
            "subject_should_run",
            "subject_reason",
            "subject_review",
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "subject_action",
            "subject_reason",
            "subject_evidence",
            "subject_verifier_should_run",
            "subject_candidate",
        ]

    @staticmethod
    def _propose(review: SubjectReview, packet: SpanPacket) -> tuple[dict | None, list[dict], str]:
        """Check the cited spans; return the proposal to verify (or None), its evidence, and the reason."""
        selected: dict[str, dict] = {"a": {}, "b": {}}
        evidence = []
        for side in ("a", "b"):
            for role in ("subject", "predicate"):
                span_id = getattr(review, f"{side}_{role}_span_id")
                if not span_id:
                    continue
                span = packet.spans.get(span_id)
                if span is None:
                    msg = f"unknown subject evidence ID {span_id!r}"
                    raise ValueError(msg)
                if not (span.kind == f"{side.upper()}_ONLY" or (role == "predicate" and span.kind == "SHARED")):
                    msg = f"wrong side or non-unique subject: {span_id}"
                    raise ValueError(msg)
                excerpt = span.excerpt(side.upper())
                selected[side][role] = {"span_id": span_id, "quote": excerpt.text}
                evidence.append(
                    {
                        **selected[side][role],
                        "side": side.upper(),
                        "role": role,
                        "start_char": excerpt.start_char,
                        "end_char": excerpt.end_char,
                    }
                )

        if review.target_relation != "DIFFERENT":
            return None, evidence, "NO_SUPPORTED_SUBJECT_VETO_KEEP_COVERAGE"
        if review.binding_type == "NONE" or any(len(selected[side]) < 2 for side in ("a", "b")):  # noqa: PLR2004
            msg = "subject veto needs both own-unique subjects and both predicate witnesses"
            raise ValueError(msg)
        if not packet.complete:
            msg = "subject veto requires complete, untruncated evidence"
            raise ValueError(msg)
        if review.binding_type not in VETO_BINDINGS:
            return None, evidence, "SUBJECT_OBJECTION_OUTSIDE_SPECIALIST_AUTHORITY"
        return {"binding_type_hypothesis": review.binding_type, **selected}, evidence, "VERIFY_FIXED_SUBJECT_VETO"

    def _apply(self, record: dict) -> tuple[str, str, list[dict], dict | None]:
        reason = record["subject_reason"]
        if not record["subject_should_run"]:
            return "SKIP", reason, [], None
        if record.get("subject_review") is None:
            return "UNVALIDATED_KEEP_COVERAGE", NO_ANSWER_REASON, [], None
        try:
            review = SubjectReview.model_validate(record["subject_review"])
            packet = SpanPacket.from_record(record, max_evidence_chars=self.max_evidence_chars)
            candidate, evidence, reason = self._propose(review, packet)
        except ValueError as error:
            return "UNVALIDATED_KEEP_COVERAGE", f"INVALID_CRITIC_OUTPUT: {str(error)[:300]}", [], None
        return ("PENDING_VERIFICATION" if candidate else "KEEP_COVERAGE"), reason, evidence, candidate

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        results = [self._apply(record) for record in table.to_pylist()]
        table = replace_columns(
            table,
            {
                "subject_action": pa.array([action for action, *_ in results], type=pa.string()),
                "subject_reason": pa.array([reason for _, reason, *_ in results], type=pa.string()),
                "subject_evidence": pa.array([evidence for _, _, evidence, _ in results], type=_EVIDENCE_ARROW_TYPE),
                "subject_verifier_should_run": pa.array(
                    [candidate is not None for *_, candidate in results], type=pa.bool_()
                ),
                "subject_candidate": pa.array([candidate for *_, candidate in results], type=_CANDIDATE_ARROW_TYPE),
            },
            drop=["subject_packet"],
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=table, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )


@dataclass
class SubjectVerifierApplyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Apply a verified subject proposal to `final_decision`; otherwise keep the coverage result."""

    name: str = "subject_verifier_apply"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "final_decision",
            "subject_action",
            "subject_reason",
            "subject_verifier_should_run",
            "subject_verifier_review",
            "subject_candidate",
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["subject_action", "subject_reason", "final_decision"]

    @staticmethod
    def _apply(record: dict) -> tuple[str, str, Decision | None]:
        action, reason = record["subject_action"], record["subject_reason"]
        try:
            main = Decision.model_validate(record["final_decision"])
        except ValueError:
            return action, reason, None
        if not record["subject_verifier_should_run"]:
            return action, reason, main
        if record.get("subject_verifier_review") is None:
            return "UNVALIDATED_KEEP_COVERAGE", NO_ANSWER_REASON, main

        try:
            review = SubjectVerification.model_validate(record["subject_verifier_review"])
            if review.comparison != "SUPPORTED_DIFFERENT_NAMED_TARGETS":
                return "KEEP_COVERAGE", "UNSUPPORTED_FIXED_PROPOSAL_KEEP_COVERAGE", main
            if not review.a_subject_kind == review.b_subject_kind == "NAMED_ACTUAL_TARGET":
                msg = "supported subject comparison requires two named actual targets"
                raise ValueError(msg)  # noqa: TRY301
            final = main.model_copy(
                update={
                    "a_can_replace_b": "no",
                    "b_can_replace_a": "no",
                    "relation_type": "related_non_duplicate",
                    "material_difference": "major",
                    "primary_material_difference": "document_identity_change",
                    "primary_risk_factor": "template_slot_collision",
                    "confidence_tier": "medium",
                }
            )
            if final.inconsistencies():
                msg = f"veto produced an incoherent decision: {final.inconsistencies()}"
                raise ValueError(msg)  # noqa: TRY301
        except ValueError as error:
            return "UNVALIDATED_KEEP_COVERAGE", f"INVALID_CRITIC_OUTPUT: {str(error)[:300]}", main
        return "REJECT_BOTH", "VERIFIED_FIXED_SUBJECT_VETO", final

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        results = [self._apply(record) for record in table.to_pylist()]
        table = replace_columns(
            table,
            {
                "subject_action": pa.array([action for action, *_ in results], type=pa.string()),
                "subject_reason": pa.array([reason for _, reason, _ in results], type=pa.string()),
                "final_decision": pa.array(
                    [final.model_dump() if final else None for *_, final in results], type=DECISION_ARROW_TYPE
                ),
            },
            drop=["subject_candidate"],
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=table, _metadata=batch._metadata, _stage_perf=batch._stage_perf
        )
