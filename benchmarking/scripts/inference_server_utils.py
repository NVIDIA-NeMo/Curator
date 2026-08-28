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

# ruff: noqa: PLR0913

"""Shared helpers for inference servers used by benchmark scripts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nemo_curator.core.serve import InferenceServer

InferenceServerBackend = Literal["ray-serve", "dynamo"]


def parse_json_object(value: str | None, *, argument: str) -> dict[str, Any]:
    """Parse an optional command-line JSON object."""
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        msg = f"{argument} must be valid JSON: {error}"
        raise ValueError(msg) from error
    if not isinstance(parsed, dict):
        msg = f"{argument} must decode to a JSON object"
        raise TypeError(msg)
    return parsed


def static_num_replicas(autoscaling_config: dict[str, Any] | None) -> int:
    """Resolve a fixed replica count from a Ray-style autoscaling config."""
    if not autoscaling_config:
        return 1
    min_replicas = int(autoscaling_config.get("min_replicas", 1))
    max_replicas = int(autoscaling_config.get("max_replicas", min_replicas))
    if min_replicas != max_replicas:
        msg = (
            "Dynamo does not support autoscaling in benchmarks; "
            f"min_replicas ({min_replicas}) must equal max_replicas ({max_replicas})."
        )
        raise ValueError(msg)
    if min_replicas < 1:
        msg = f"num_replicas must be at least 1, got {min_replicas}"
        raise ValueError(msg)
    return min_replicas


def start_inference_server(
    *,
    backend: InferenceServerBackend,
    model_id: str,
    num_replicas: int,
    engine_kwargs: dict[str, Any] | None = None,
    model_path: str | None = None,
    model_runtime_env: dict[str, Any] | None = None,
    dynamo_kwargs: dict[str, Any] | None = None,
    dynamo_router_kwargs: dict[str, Any] | None = None,
    dynamo_subprocess_env: dict[str, str] | None = None,
    ray_serve_deployment_config: dict[str, Any] | None = None,
    health_check_timeout_s: int = 900,
) -> InferenceServer:
    """Build, start, and return an inference server."""
    from nemo_curator.core.serve import InferenceServer

    if num_replicas < 1:
        msg = f"num_replicas must be at least 1, got {num_replicas}"
        raise ValueError(msg)
    if backend == "dynamo":
        from nemo_curator.core.serve import DynamoRouterConfig, DynamoServerConfig, DynamoVLLMModelConfig

        model = DynamoVLLMModelConfig(
            model_identifier=model_path or model_id,
            model_name=model_id if model_path else None,
            engine_kwargs=engine_kwargs or {},
            num_replicas=num_replicas,
            dynamo_kwargs=dynamo_kwargs or {},
            runtime_env=model_runtime_env or {},
        )
        server = InferenceServer(
            models=[model],
            backend=DynamoServerConfig(
                request_plane="tcp",
                router=DynamoRouterConfig(router_kwargs=dynamo_router_kwargs or {}),
                subprocess_env=dynamo_subprocess_env or {},
            ),
            health_check_timeout_s=health_check_timeout_s,
        )
    else:
        if backend != "ray-serve":
            msg = f"Unsupported inference server backend: {backend}"
            raise ValueError(msg)

        from nemo_curator.core.serve import RayServeModelConfig

        model = RayServeModelConfig(
            model_identifier=model_path or model_id,
            model_name=model_id if model_path else None,
            deployment_config=(
                ray_serve_deployment_config
                if ray_serve_deployment_config is not None
                else {"num_replicas": num_replicas}
            ),
            engine_kwargs=engine_kwargs or {},
            runtime_env=model_runtime_env or {},
        )
        server = InferenceServer(models=[model], health_check_timeout_s=health_check_timeout_s)

    server.start()
    return server
