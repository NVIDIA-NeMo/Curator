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
"""Worker-side helpers to talk to the resumability actor; all no-ops when no
actor is registered, so unchecked pipelines pay nothing.
"""

from __future__ import annotations

import ray

# Defined here (not imported from resumability_actor) so the always-imported
# worker path doesn't pull in lmdb until resumability is actually used.
ACTOR_NAME = "nemo_curator_resumability"


def _actor() -> ray.actor.ActorHandle | None:
    """The resumability actor handle, or None if Ray is down / no actor registered."""
    if not ray.is_initialized():
        return None
    try:
        return ray.get_actor(ACTOR_NAME)
    except ValueError:
        return None


def _is_active() -> bool:
    """True if a resumability actor is registered in this Ray cluster."""
    return _actor() is not None


def _flush_deltas(per_task: list[tuple[str, str, int]]) -> None:
    """Fire-and-forget per-task deltas ``(task_id, source_id, delta)``. No
    ``ray.get`` — the actor never raises, so there's no error path; backpressure
    is the actor's ``max_pending_calls`` cap."""
    a = _actor()
    if a is not None and per_task:
        a.apply_deltas.remote(per_task)  # type: ignore[attr-defined]


def _skip_completed_sources(source_ids: list[str]) -> set[str]:
    """Set of ``source_ids`` already marked complete; the source stage uses it to skip them."""
    a = _actor()
    if a is None or not source_ids:
        return set()
    flags = ray.get(a.are_completed.remote(source_ids))  # type: ignore[attr-defined]
    return {sid for sid, done in zip(source_ids, flags, strict=True) if done}
