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

import os
from pathlib import Path
from typing import Any, Literal

from nemo_curator.core.serve import (
    DynamoRouterConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    RayServeModelConfig,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
    DEFAULT_MODEL_PATH,
    set_nemotron_parse_attention_backend,
)

NemotronParseServerBackend = Literal["ray-serve", "dynamo"]

_DEFAULT_ENGINE_KWARGS: dict[str, Any] = {
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "limit_mm_per_prompt": {"image": 1},
    "enable_prefix_caching": False,
    "disable_hybrid_kv_cache_manager": False,
}

_PARSE35_ENGINE_KWARGS: dict[str, Any] = {
    "tensor_parallel_size": 1,
    "distributed_executor_backend": "uni",
    "max_model_len": 16384,
    "max_num_batched_tokens": 16384,
}
# Serving workers start from a clone of the driver environment, whose openai
# client can predate the response types vLLM's tool parsers import.
_PARSE35_WORKER_PACKAGES = ["openai>=3.37,<4"]


def create_nemotron_parse_inference_server(  # noqa: PLR0913
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    model_name: str | None = None,
    backend: NemotronParseServerBackend = "dynamo",
    num_replicas: int = 1,
    engine_kwargs: dict[str, Any] | None = None,
    request_timeout_s: float = 300.0,
    health_check_timeout_s: int = 900,
    nemotron_parse_path: str | None = None,
) -> InferenceServer:
    """Return an inference server configured for Nemotron-Parse PDFs.

    The returned server is not started. Use it as a context manager or call
    :meth:`InferenceServer.start` and :meth:`InferenceServer.stop` explicitly.
    Set ``nemotron_parse_path`` to a local Nemotron Parse 3.5 model repository
    to serve that model through Dynamo: its package is installed on every
    serving worker and its chat template is applied by the frontend.
    """
    if num_replicas < 1:
        msg = f"num_replicas must be at least 1, got {num_replicas}"
        raise ValueError(msg)
    if request_timeout_s < 1:
        msg = f"request_timeout_s must be at least 1, got {request_timeout_s}"
        raise ValueError(msg)
    # Ray Serve runs the driver's vLLM, which is older than Parse 3.5 supports.
    if nemotron_parse_path is not None and backend != "dynamo":
        msg = f"nemotron_parse_path requires the dynamo backend, got {backend}"
        raise ValueError(msg)

    default_engine_kwargs = _DEFAULT_ENGINE_KWARGS
    packages = ["albumentations==2.0.8"]
    if nemotron_parse_path is not None:
        default_engine_kwargs = {**_DEFAULT_ENGINE_KWARGS, **_PARSE35_ENGINE_KWARGS}
        packages = [*packages, *_PARSE35_WORKER_PACKAGES, nemotron_parse_path]

    resolved_engine_kwargs = {**default_engine_kwargs, **(engine_kwargs or {})}
    model_kwargs = {
        "model_identifier": model_path,
        "model_name": model_name,
        "engine_kwargs": resolved_engine_kwargs,
        "runtime_env": {"uv": {"packages": packages}},
    }

    if backend == "dynamo":
        router_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if nemotron_parse_path is not None:
            # Dynamo's frontend applies chat templates; workers accept engine flags only.
            router_kwargs.update(
                dyn_chat_processor="vllm",
                model_name=model_name or model_path,
                chat_template=os.path.join(nemotron_parse_path, "chat_template.jinja"),
            )
            # The frontend's --model-path accepts only local directories.
            if Path(model_path).is_dir():
                router_kwargs["model_path"] = model_path
        model = DynamoVLLMModelConfig(
            **model_kwargs,
            num_replicas=num_replicas,
            dynamo_kwargs={"enable_multimodal": True},
        )
        server_config = DynamoServerConfig(
            request_plane="tcp",
            router=DynamoRouterConfig(router_kwargs=router_kwargs),
            subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": str(int(request_timeout_s))},
        )
        return InferenceServer(
            models=[model],
            backend=server_config,
            health_check_timeout_s=health_check_timeout_s,
        )
    if backend == "ray-serve":
        # Assumes the driver's GPU architecture matches the serving replicas;
        # a CPU-only or heterogeneous driver cannot select their fallback.
        set_nemotron_parse_attention_backend(resolved_engine_kwargs)
        model = RayServeModelConfig(
            **model_kwargs,
            deployment_config={"num_replicas": num_replicas},
        )
        return InferenceServer(models=[model], health_check_timeout_s=health_check_timeout_s)

    msg = f"Unsupported inference server backend: {backend}"
    raise ValueError(msg)
