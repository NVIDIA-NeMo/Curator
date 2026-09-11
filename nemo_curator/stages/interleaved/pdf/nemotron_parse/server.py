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

"""Inference-server configuration for Nemotron-Parse."""

from __future__ import annotations

from typing import Any, Literal

from nemo_curator.core.serve import (
    DynamoRouterConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    RayServeModelConfig,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import DEFAULT_MODEL_PATH

NemotronParseServerBackend = Literal["ray-serve", "dynamo"]

_DEFAULT_ENGINE_KWARGS: dict[str, Any] = {
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "limit_mm_per_prompt": {"image": 1},
    "enable_prefix_caching": False,
    "disable_hybrid_kv_cache_manager": False,
}


def create_nemotron_parse_inference_server(  # noqa: PLR0913
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    model_name: str | None = None,
    backend: NemotronParseServerBackend = "dynamo",
    num_replicas: int = 1,
    engine_kwargs: dict[str, Any] | None = None,
    request_timeout_s: float = 300.0,
    health_check_timeout_s: int = 900,
) -> InferenceServer:
    """Return an inference server configured for Nemotron-Parse PDFs.

    The returned server is not started. Use it as a context manager or call
    :meth:`InferenceServer.start` and :meth:`InferenceServer.stop` explicitly.
    """
    if num_replicas < 1:
        msg = f"num_replicas must be at least 1, got {num_replicas}"
        raise ValueError(msg)
    if request_timeout_s < 1:
        msg = f"request_timeout_s must be at least 1, got {request_timeout_s}"
        raise ValueError(msg)

    resolved_engine_kwargs = {**_DEFAULT_ENGINE_KWARGS, **(engine_kwargs or {})}
    model_kwargs = {
        "model_identifier": model_path,
        "model_name": model_name,
        "engine_kwargs": resolved_engine_kwargs,
        "runtime_env": {"uv": {"packages": ["albumentations==2.0.8"]}},
    }

    if backend == "dynamo":
        model = DynamoVLLMModelConfig(
            **model_kwargs,
            num_replicas=num_replicas,
            dynamo_kwargs={"enable_multimodal": True},
        )
        server_config = DynamoServerConfig(
            request_plane="tcp",
            router=DynamoRouterConfig(router_kwargs={"trust_remote_code": True}),
            subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": str(int(request_timeout_s))},
        )
        return InferenceServer(
            models=[model],
            backend=server_config,
            health_check_timeout_s=health_check_timeout_s,
        )
    if backend == "ray-serve":
        model = RayServeModelConfig(
            **model_kwargs,
            deployment_config={"num_replicas": num_replicas},
        )
        return InferenceServer(models=[model], health_check_timeout_s=health_check_timeout_s)

    msg = f"Unsupported inference server backend: {backend}"
    raise ValueError(msg)
