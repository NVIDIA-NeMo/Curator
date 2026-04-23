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
from typing import Any, ClassVar, Literal

from nemo_curator.core.serve.base import BaseModelConfig, BaseServerConfig
from nemo_curator.core.serve.dynamo.constants import (
    DEFAULT_DYNAMO_EVENT_PLANE,
    DEFAULT_DYNAMO_NAMESPACE,
    DEFAULT_DYNAMO_REQUEST_PLANE,
)


@dataclass
class DynamoRoleConfig:
    """Per-role config for disaggregated Dynamo serving."""

    num_replicas: int = 1
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_replicas < 0:
            msg = f"num_replicas must be >= 0, got {self.num_replicas}"
            raise ValueError(msg)


@dataclass
class DynamoRouterConfig:
    """Frontend router config for Dynamo.

    ``mode=None`` means "auto": Curator picks ``"kv"`` if any model uses
    ``mode="disagg"``, else leaves ``--router-mode`` unset so the Dynamo
    frontend falls back to its own ``round_robin`` default. ``kv_events``
    only applies when ``mode == "kv"``: pass ``kv_events=True`` to opt into
    exact ZMQ KV-cache event publishing; the default uses the router's
    approximate tree-based tracking. Anything else is forwarded to the
    Dynamo frontend as CLI args via ``router_kwargs``.
    """

    mode: Literal["round_robin", "random", "kv", "direct"] | None = None
    kv_events: bool = False
    router_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode != "kv" and self.kv_events:
            msg = f"kv_events=True is only meaningful when mode='kv'; got mode={self.mode!r}."
            raise ValueError(msg)


@dataclass
class DynamoVLLMModelConfig(BaseModelConfig):
    """Dynamo vLLM model config.

    Typed fields cover deployment/placement knobs Curator branches on; anything
    else is forwarded to ``python -m dynamo.vllm`` via ``dynamo_kwargs``.
    ``kv_events_config`` and ``kv_transfer_config`` are Curator-managed
    (``init=False``): events are derived from router state + port allocation,
    transfer defaults to NixlConnector for disagg.
    """

    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    num_replicas: int = 1
    mode: Literal["aggregated", "disagg"] = "aggregated"
    prefill: DynamoRoleConfig | None = None
    decode: DynamoRoleConfig | None = None
    dynamo_kwargs: dict[str, Any] = field(default_factory=dict)
    kv_events_config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    kv_transfer_config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_replicas < 1:
            msg = f"num_replicas must be >= 1, got {self.num_replicas}"
            raise ValueError(msg)
        if self.mode == "aggregated" and (self.prefill is not None or self.decode is not None):
            msg = "prefill/decode are only valid with mode='disagg'"
            raise ValueError(msg)
        if self.mode == "disagg":
            if (self.prefill is None) != (self.decode is None):
                msg = "mode='disagg' requires both prefill and decode to be specified, or neither"
                raise ValueError(msg)
            if self.prefill is not None and (self.prefill.num_replicas < 1 or self.decode.num_replicas < 1):
                msg = "mode='disagg' requires prefill.num_replicas >= 1 and decode.num_replicas >= 1"
                raise ValueError(msg)


@dataclass
class DynamoServerConfig(BaseServerConfig):
    """Server-level Dynamo config."""

    model_configs: ClassVar[tuple[type[BaseModelConfig], ...]] = (DynamoVLLMModelConfig,)

    etcd_endpoint: str | None = None
    nats_url: str | None = None
    namespace: str = DEFAULT_DYNAMO_NAMESPACE
    request_plane: str = DEFAULT_DYNAMO_REQUEST_PLANE
    event_plane: str = DEFAULT_DYNAMO_EVENT_PLANE
    router: DynamoRouterConfig = field(default_factory=DynamoRouterConfig)
