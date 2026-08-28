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

"""Audio Agent (P1) — host-driven pipeline builder.

A thin, deterministic tool core that a host LLM (Claude/Cursor/Codex) drives to
turn a natural-language audio-curation goal into a validated, runnable NeMo
Curator recipe, with pre-flight checks and failure triage.

The host LLM plays the reasoning roles (interpret / route / plan / critique);
this package provides the deterministic verbs, the knowledge index, and the
typed data contracts that ground every LLM decision. See ``verbs`` for the
public tool surface and ``AGENTS.md`` / the skill for the driving instructions.

    from nemo_curator import audio_agent

    audio_agent.discover()                 # what stages exist
    audio_agent.catalog_tree()             # L0 category tree for routing
    audio_agent.validate(recipe)           # Verdict (roles/keys/cards/gates)
    audio_agent.smoke(recipe, sample=10)   # bounded evidence
    audio_agent.run(recipe, confirm=True)  # confirm-gated full run + report
"""

from __future__ import annotations

from nemo_curator.audio_agent.contracts import (
    AcceptanceCriterion,
    AcceptanceReport,
    ConfigStrategyEntry,
    CriterionResult,
    DataProfile,
    EnvProfile,
    PlanningContext,
    PlanResult,
    RunRecord,
    SmokeReport,
    Verdict,
)
from nemo_curator.audio_agent.diagnostics import EnvironmentDecision
from nemo_curator.audio_agent.env_health import (
    RemediationOption,
    doctor,
    env_report,
    render_doctor,
)
from nemo_curator.audio_agent.recipe import Recipe, StageRef
from nemo_curator.audio_agent.report import RunReport
from nemo_curator.audio_agent.run_store import scratch_dir
from nemo_curator.audio_agent.semantic_review import (
    build_semantic_review,
    semantic_response_contract,
)
from nemo_curator.audio_agent.verbs import (
    add_checkpoint,
    available_skills,
    calibrate,
    cards,
    catalog_tree,
    checkpoints,
    context,
    delta_run,
    describe,
    diagnose,
    discover,
    install_skill,
    plan_checkpoint,
    plan_continuation,
    producers,
    reindex,
    report,
    resolve,
    reuse_scan,
    run,
    runs,
    skills_dir,
    smoke,
    validate,
    verify,
)

__all__ = [
    "AcceptanceCriterion",
    "AcceptanceReport",
    "ConfigStrategyEntry",
    "CriterionResult",
    "DataProfile",
    "EnvProfile",
    "EnvironmentDecision",
    "PlanResult",
    "PlanningContext",
    "Recipe",
    "RemediationOption",
    "RunRecord",
    "RunReport",
    "SmokeReport",
    "StageRef",
    "Verdict",
    "add_checkpoint",
    "available_skills",
    "build_semantic_review",
    "calibrate",
    "cards",
    "catalog_tree",
    "checkpoints",
    "context",
    "delta_run",
    "describe",
    "diagnose",
    "discover",
    "doctor",
    "env_report",
    "install_skill",
    "plan_checkpoint",
    "plan_continuation",
    "producers",
    "reindex",
    "render_doctor",
    "report",
    "resolve",
    "reuse_scan",
    "run",
    "runs",
    "scratch_dir",
    "semantic_response_contract",
    "skills_dir",
    "smoke",
    "validate",
    "verify",
]
