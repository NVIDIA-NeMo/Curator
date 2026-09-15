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

"""Private foundation for agent-driven audio pipeline construction.

These modules are the STAGE-SIDE declaration layer: the vocabulary a stage uses
to describe itself (``StageContract``, ``Gates``, ``IOSpec``, the ``AgentReady``
mixin, the shared role names) plus the discovery, planning and conformance code
that reads it. They live under ``stages/audio`` -- not under ``nemo_curator.audio_agent``
-- on purpose: 43 stage modules import ``_agent_ready`` and 16 call into
``_residency`` from inside ``process()``. Moving them into the agent package would
make ``nemo_curator.stages.audio`` unusable without the agent installed, inverting
a dependency that today points one way only.

Grouped into this subpackage purely so the stage tree reads as stages. Import the
public facade -- :mod:`nemo_curator.stages.audio.agent` -- rather than these
modules directly.
"""
