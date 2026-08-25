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

"""Public entry point for agent-driven audio pipeline construction.

Everything an agent (or agent-tool layer) needs to discover, compose,
configure, and validate audio pipelines, re-exported from the private
foundation modules. Import from here, not from the underscore modules.

The intended loop::

    from nemo_curator.stages.audio import agent

    # 1. DISCOVER — what stages exist and what they do
    names = agent.list_agent_ready_stages()
    catalog = agent.audio_stage_catalog()      # contracts + params_schema per stage

    # 2. PLAN — how roles connect stages
    agent.find_producers("segments")           # -> stages that can supply a role
    agent.find_consumers("pred_text")          # [] from find_producers => unproducible

    # 3. CONFIGURE + BUILD
    cls = agent.get_agent_ready_stage_class("UTMOSFilterStage")
    stage = cls(mos_threshold=3.5)             # params_schema documents the knobs

    # 4. VALIDATE (before ever running)
    report = agent.validate_pipeline([stage, ...], initial_keys={"audio_filepath", "text"})
    report.ok        # role-level composability (necessary condition)
    report.keys_ok   # literal keys connect (mechanical flow, not intent meaning)
    report.summary() # human/agent-readable issues incl. dangling_key / tensor_into_sink

``describe_stage(name)`` returns a contract marked
``static_params_and_hints``; pass an instance to ``build_contract`` for the
configured dynamic contract with resolved I/O/key values.
``StageContract.to_dict()`` is JSON-safe by construction. Open-ended intent fit
is reviewed by the host LLM over ``audio_agent.validate(...).semantic_review``.
"""

from __future__ import annotations

from nemo_curator.stages.audio._agent._agent_ready import (
    ConditionalWrite,
    Gates,
    IOSpec,
    ParamSpec,
    Role,
    SizeEnvelope,
    StageContract,
    to_json_schema,
)
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._catalog import (
    audio_stage_catalog,
    catalog_as_json,
    describe_stage,
    find_consumers,
    find_producers,
    get_agent_ready_stage_class,
    list_agent_ready_stages,
    role_index,
)
from nemo_curator.stages.audio._agent._conformance import (
    assert_contract_wellformed,
    produced_roles,
    reads_satisfied_by_role,
)
from nemo_curator.stages.audio._agent._planning import PipelineIssue, PipelineReport, validate_pipeline

__all__ = [
    "ConditionalWrite",
    "Gates",
    "IOSpec",
    "ParamSpec",
    "PipelineIssue",
    "PipelineReport",
    "Role",
    "SizeEnvelope",
    "StageContract",
    "assert_contract_wellformed",
    "audio_stage_catalog",
    "build_contract",
    "catalog_as_json",
    "describe_stage",
    "find_consumers",
    "find_producers",
    "get_agent_ready_stage_class",
    "list_agent_ready_stages",
    "produced_roles",
    "reads_satisfied_by_role",
    "role_index",
    "static_contract",
    "to_json_schema",
    "validate_pipeline",
]
