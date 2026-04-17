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

"""Dynamo-specific placement, naming, and CLI-translation helpers.

Kept separate from ``subprocess_mgr`` so the generic Ray/subprocess
infrastructure there stays reusable and free of Dynamo conventions
(infra services, worker-actor naming, vLLM CLI flag shape).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.core.serve.constants import PLACEMENT_GROUP_READY_TIMEOUT_S, WORKER_NODE_LABEL
from nemo_curator.core.serve.placement import build_pg
from nemo_curator.core.utils import ignore_ray_head_node

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


def build_infra_pg(
    *,
    name: str,
    num_bundles: int,
    ready_timeout_s: float = PLACEMENT_GROUP_READY_TIMEOUT_S,
) -> PlacementGroup:
    """Create a ``STRICT_PACK`` PG for Dynamo infra services (etcd + NATS + frontend).

    All bundles co-locate on one node so infra chatter stays off the wire.
    When ``CURATOR_IGNORE_RAY_HEAD_NODE`` is set, every bundle requires a
    non-head (worker-labeled) node.
    """
    selector = [WORKER_NODE_LABEL] * num_bundles if ignore_ray_head_node() else None
    return build_pg(
        [{"CPU": 1}] * num_bundles,
        "STRICT_PACK",
        name=name,
        bundle_label_selector=selector,
        ready_timeout_s=ready_timeout_s,
    )


def build_worker_actor_name(
    model_name: str,
    replica_index: int,
    node_rank: int,
    tp_size: int,
    *,
    role: Literal["decode", "prefill"] | None = None,
) -> str:
    """Build a descriptive Dynamo worker actor name for Ray dashboard visibility.

    Format: ``Dynamo_[<role>_]DP<n>[_TP<n>]_<model>``.

    Examples::

        build_worker_actor_name("Qwen3-0.6B", 0, 0, 1)           # Dynamo_DP0_Qwen3-0.6B
        build_worker_actor_name("Qwen3-0.6B", 1, 0, 4)           # Dynamo_DP1_TP0_Qwen3-0.6B
        build_worker_actor_name("Qwen3-0.6B", 0, 0, 2, role="decode")  # Dynamo_decode_DP0_TP0_Qwen3-0.6B
    """
    short_name = model_name.rsplit("/", 1)[-1]
    parts = ["Dynamo"]
    if role:
        parts.append(role)
    parts.append(f"DP{replica_index}")
    if tp_size > 1:
        parts.append(f"TP{node_rank}")
    parts.append(short_name)
    return "_".join(parts)


def engine_kwargs_to_cli_flags(engine_kwargs: dict[str, Any]) -> list[str]:
    """Convert a vLLM ``engine_kwargs`` dict to a list of CLI flags.

    Example: ``{"tensor_parallel_size": 4, "enforce_eager": True}``
    becomes ``["--tensor-parallel-size", "4", "--enforce-eager"]``.
    """
    flags: list[str] = []
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif isinstance(value, (dict, list)):
            flags.extend([flag, json.dumps(value)])
        else:
            flags.extend([flag, str(value)])
    return flags
