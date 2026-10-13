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

"""The main judge's decision as a typed record."""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, ValidationError

YesNo = Literal["yes", "no", "unresolved"]


class Decision(BaseModel):
    """The eight summary fields of the main judge's `pair_semantic_judgment`."""

    model_config = ConfigDict(frozen=True)

    a_can_replace_b: YesNo
    b_can_replace_a: YesNo
    relation_type: Literal[
        "exact",
        "canonical_exact",
        "near_surface",
        "containment",
        "version_related",
        "related_non_duplicate",
        "unrelated",
        "unresolved",
    ]
    material_difference: Literal["none", "minor", "major", "unresolved"]
    primary_material_difference: Literal[
        "none",
        "main_content_addition_deletion",
        "entity_slot_change",
        "document_identity_change",
        "number_change",
        "date_time_change",
        "product_version_change",
        "result_set_change",
        "page_role_change",
        "legal_context_change",
        "negation_change",
        "code_literal_change",
        "code_output_change",
        "other_material",
        "unresolved",
    ]
    dominant_overlap_source: Literal[
        "main_content",
        "shared_page_template",
        "site_chrome",
        "cookie_consent",
        "legal_policy_template",
        "error_auth_paywall",
        "local_passage",
        "parser_artifact",
        "none",
        "unresolved",
    ]
    primary_risk_factor: Literal[
        "none",
        "boilerplate_dominated_similarity",
        "template_slot_collision",
        "identifier_underweighting",
        "topic_only_similarity",
        "list_snapshot_collision",
        "page_role_collision",
        "legal_context_collision",
        "long_document_local_overlap",
        "translation_equivalence",
        "paraphrase_equivalence",
        "containment_asymmetry",
        "extraction_or_payload_limit",
        "parser_artifact_dominance",
        "other",
    ]
    confidence_tier: Literal["high", "medium", "low"]

    @classmethod
    def from_judge(cls, judgment: object) -> Decision:
        """Build from a judge column value shaped like `{score_name: {"score": ...}}`."""
        try:
            return cls(**{name: judgment[name]["score"] for name in cls.model_fields})
        except (KeyError, TypeError, ValidationError) as error:
            msg = f"unreadable judge result: {error}"
            raise ValueError(msg) from error

    def inconsistencies(self) -> list[str]:
        """Return the ways the fields contradict each other; empty means the decision is coherent."""
        a, b, relation = self.a_can_replace_b, self.b_can_replace_a, self.relation_type
        material, primary = self.material_difference, self.primary_material_difference

        if "unresolved" in (a, b, relation):
            fully_unresolved = (
                a == b == relation == material == primary == self.dominant_overlap_source == "unresolved"
                and self.primary_risk_factor == "extraction_or_payload_limit"
                and self.confidence_tier == "low"
            )
            return [] if fully_unresolved else ["partly unresolved decision"]

        problems = []
        if "unresolved" in (material, primary, self.dominant_overlap_source):
            problems.append("resolved relation with unresolved summary fields")

        if relation in {"exact", "canonical_exact"}:
            coherent = a == b == "yes" and material == "none"
        elif relation == "near_surface":
            coherent = a == b == "yes" and material in {"none", "minor"}
        elif relation == "containment":
            coherent = {a, b} == {"yes", "no"} and material == "major" and primary == "main_content_addition_deletion"
        else:
            coherent = a == b == "no" and material == "major"
        if not coherent:
            problems.append("replacement directions, relation and materiality disagree")

        if (material == "none") != (primary == "none"):
            problems.append("materiality and primary difference disagree")
        return problems


DECISION_ARROW_TYPE = pa.struct([(name, pa.string()) for name in Decision.model_fields])
