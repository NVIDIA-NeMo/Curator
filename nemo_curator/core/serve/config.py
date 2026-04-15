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

from dataclasses import dataclass, field
from typing import Any, ClassVar

from nemo_curator.core.serve.internal.constants import (
    DEFAULT_DYNAMO_EVENT_PLANE,
    DEFAULT_DYNAMO_NAMESPACE,
    DEFAULT_DYNAMO_REQUEST_PLANE,
)


@dataclass
class BaseModelConfig:
    """Base public model config shared by inference backends."""

    family: ClassVar[str]

    model_identifier: str
    model_name: str | None = None
    runtime_env: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or self.model_identifier

    @staticmethod
    def merge_runtime_envs(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
        """Merge two runtime_env dicts while preserving package lists."""
        if not base and not override:
            return {}
        if not override:
            return {**base}
        if not base:
            return {**override}

        merged = {**base, **override}

        base_env_vars = base.get("env_vars", {})
        override_env_vars = override.get("env_vars", {})
        if base_env_vars or override_env_vars:
            merged["env_vars"] = {**base_env_vars, **override_env_vars}

        for key in ("pip", "uv"):
            base_packages = base.get(key, [])
            override_packages = override.get(key, [])
            if base_packages and override_packages:
                merged[key] = [*base_packages, *override_packages]

        return merged


@dataclass
class RayServeModelConfig(BaseModelConfig):
    """Ray Serve model config."""

    family: ClassVar[str] = "ray_serve"

    deployment_config: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamoRoleConfig:
    """Per-role config for disaggregated Dynamo serving."""

    num_replicas: int = 1
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamoVLLMModelConfig(BaseModelConfig):
    """Dynamo vLLM model config."""

    family: ClassVar[str] = "dynamo"

    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    num_replicas: int = 1
    mode: str | None = None
    prefill: DynamoRoleConfig | None = None
    decode: DynamoRoleConfig | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    custom_jinja_template: str | None = None
    model_express_url: str | None = None
    kv_events_config: dict[str, Any] = field(default_factory=dict)
    kv_transfer_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamoRouterConfig:
    """Frontend router config for Dynamo."""

    mode: str | None = None
    kv_events: bool = True
    kv_overlap_score_weight: float = 1.0
    temperature: float = 0.0
    queue_threshold: int | None = None
    ttl_secs: float = 120.0
    max_tree_size: int = 2**20
    prune_target_ratio: float = 0.8
    reset_states: bool = False


@dataclass
class RayServeServerConfig:
    """Server-level Ray Serve config."""

    family: ClassVar[str] = "ray_serve"


@dataclass
class DynamoServerConfig:
    """Server-level Dynamo config."""

    family: ClassVar[str] = "dynamo"

    etcd_endpoint: str | None = None
    nats_url: str | None = None
    namespace: str = DEFAULT_DYNAMO_NAMESPACE
    request_plane: str = DEFAULT_DYNAMO_REQUEST_PLANE
    event_plane: str = DEFAULT_DYNAMO_EVENT_PLANE
    router: DynamoRouterConfig = field(default_factory=DynamoRouterConfig)
