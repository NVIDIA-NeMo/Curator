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
"""Module-level helpers that workers use to talk to the resumability
actor. All helpers are no-ops when the actor isn't registered, so
unchecked pipelines pay nothing.
"""

from __future__ import annotations

import ray

# Name of the detached resumability actor. Defined here — NOT imported from
# resumability_actor — so the always-imported worker path
# (BaseStageAdapter -> this module) never pulls in resumability_actor, which
# imports lmdb. lmdb is only needed once resumability is actually used (the
# actor process and Pipeline._run_with_resumability).
ACTOR_NAME = "nemo_curator_resumability"


def _actor() -> ray.actor.ActorHandle | None:
    """Return the resumability actor handle, or None if Ray is not
    initialized or no such actor is registered."""
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
    """Fire-and-forget per-task counter deltas to the actor.

    Each entry is ``(task_hash, source_id, delta)``. Workers do NOT
    ``ray.get`` the returned ref — errors surface to the executor via the
    actor's watchdog poll, not synchronously on this call. Backpressure
    is handled by Ray's ``_max_pending_calls`` cap on the actor.
    """
    a = _actor()
    if a is not None and per_task:
        a.apply_deltas.remote(per_task)  # type: ignore[attr-defined]


def _skip_completed_sources(source_ids: list[str]) -> set[str]:
    """Synchronous lookup of which source_ids are already marked complete
    in LMDB. Used by the source-stage adapter to drop already-done
    sources from its output before downstream stages see them."""
    a = _actor()
    if a is None or not source_ids:
        return set()
    flags = ray.get(a.are_completed.remote(source_ids))  # type: ignore[attr-defined]
    return {sid for sid, done in zip(source_ids, flags, strict=True) if done}
