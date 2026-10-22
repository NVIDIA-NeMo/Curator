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

"""The deterministic verb surface the host LLM (and CLI/MCP) drives.

Each verb returns a JSON-safe dict. Retrieval/validation/execution/reporting is
ours (deterministic, grounded); interpret/route/plan/critique is the host's.

    discover / describe / catalog_tree / cards / context   -> knowledge + routing
    validate                                               -> Verdict (grounds the plan)
    smoke                                                  -> bounded evidence
    run                                                    -> confirm-gated full run + report
    report                                                 -> post-hoc evidence from outputs"""

from nemo_curator.stages.audio.audio_agent._verb_calibration import calibrate
from nemo_curator.stages.audio.audio_agent._verb_checkpoint import add_checkpoint, plan_checkpoint
from nemo_curator.stages.audio.audio_agent._verb_continuation import plan_continuation
from nemo_curator.stages.audio.audio_agent._verb_delta import delta_run
from nemo_curator.stages.audio.audio_agent._verb_inspection import (
    cards,
    catalog_tree,
    context,
    describe,
    diagnose,
    discover,
    producers,
    resolve,
)
from nemo_curator.stages.audio.audio_agent._verb_report import report, verify
from nemo_curator.stages.audio.audio_agent._verb_reuse import checkpoints, reindex, reuse_scan, runs
from nemo_curator.stages.audio.audio_agent._verb_run import run
from nemo_curator.stages.audio.audio_agent._verb_skills import available_skills, install_skill, skills_dir
from nemo_curator.stages.audio.audio_agent._verb_smoke import smoke
from nemo_curator.stages.audio.audio_agent._verb_validation import validate

__all__ = [
    "add_checkpoint",
    "available_skills",
    "calibrate",
    "cards",
    "catalog_tree",
    "checkpoints",
    "context",
    "delta_run",
    "describe",
    "diagnose",
    "discover",
    "install_skill",
    "plan_checkpoint",
    "plan_continuation",
    "producers",
    "reindex",
    "report",
    "resolve",
    "reuse_scan",
    "run",
    "runs",
    "skills_dir",
    "smoke",
    "validate",
    "verify",
]
