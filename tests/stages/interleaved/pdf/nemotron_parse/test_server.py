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

"""Tests for Nemotron-Parse inference-server configuration."""

from pathlib import Path

import pytest

from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, RayServeModelConfig
from nemo_curator.stages.interleaved.pdf.nemotron_parse.server import create_nemotron_parse_inference_server


def test_dynamo_server_has_pdf_defaults_and_overrides() -> None:
    server = create_nemotron_parse_inference_server(
        model_path="/models/nemotron-parse",
        model_name="nemotron-parse",
        num_replicas=2,
        engine_kwargs={"enforce_eager": True},
        request_timeout_s=123,
    )

    model = server.models[0]
    assert isinstance(model, DynamoVLLMModelConfig)
    assert model.model_identifier == "/models/nemotron-parse"
    assert model.model_name == "nemotron-parse"
    assert model.num_replicas == 2
    assert model.engine_kwargs["limit_mm_per_prompt"] == {"image": 1}
    assert model.engine_kwargs["enforce_eager"] is True
    assert "attention_backend" not in model.engine_kwargs
    assert model.dynamo_kwargs == {"enable_multimodal": True}
    assert model.runtime_env == {"uv": {"packages": ["albumentations==2.0.8"]}}
    assert isinstance(server.backend, DynamoServerConfig)
    assert server.backend.request_plane == "tcp"
    assert server.backend.router.router_kwargs == {"trust_remote_code": True}
    assert server.backend.subprocess_env == {"DYN_TCP_REQUEST_TIMEOUT": "123"}


def test_ray_serve_server_uses_driver_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "nemo_curator.stages.interleaved.pdf.nemotron_parse.inference.importlib.metadata.version",
        lambda _package: "0.22.0",
    )
    server = create_nemotron_parse_inference_server(backend="ray-serve", num_replicas=3)

    model = server.models[0]
    assert isinstance(model, RayServeModelConfig)
    assert model.deployment_config == {"num_replicas": 3}
    assert model.engine_kwargs["limit_mm_per_prompt"] == {"image": 1}
    assert model.engine_kwargs["attention_backend"] == "TRITON_ATTN"


@pytest.mark.parametrize("num_replicas", [0, -1])
def test_rejects_non_positive_replica_count(num_replicas: int) -> None:
    with pytest.raises(ValueError, match="num_replicas must be at least 1"):
        create_nemotron_parse_inference_server(num_replicas=num_replicas)


def test_local_parse_model_configures_dynamo_workers(tmp_path: Path) -> None:
    model_path = str(tmp_path)
    server = create_nemotron_parse_inference_server(
        model_path=model_path,
        model_name="nemotron-parse",
        engine_kwargs={"max_num_seqs": 64},
        nemotron_parse_path=model_path,
    )

    model = server.models[0]
    assert model.runtime_env == {"uv": {"packages": ["albumentations==2.0.8", "openai>=3.37,<4", model_path]}}
    assert model.engine_kwargs["tensor_parallel_size"] == 1
    assert model.engine_kwargs["distributed_executor_backend"] == "uni"
    assert model.engine_kwargs["max_model_len"] == 16384
    assert model.engine_kwargs["max_num_batched_tokens"] == 16384
    assert model.engine_kwargs["max_num_seqs"] == 64
    assert "attention_config" not in model.engine_kwargs
    assert server.backend.router.router_kwargs == {
        "trust_remote_code": True,
        "dyn_chat_processor": "vllm",
        "model_name": "nemotron-parse",
        "chat_template": str(tmp_path / "chat_template.jinja"),
        "model_path": model_path,
    }


def test_parse_hub_identifier_is_not_a_frontend_local_path(tmp_path: Path) -> None:
    server = create_nemotron_parse_inference_server(
        model_path="nvidia/Nemotron-3.5-Parse",
        nemotron_parse_path=str(tmp_path),
    )
    router = server.backend.router.router_kwargs
    assert router["model_name"] == "nvidia/Nemotron-3.5-Parse"
    assert "model_path" not in router


def test_local_parse_model_rejects_ray_serve() -> None:
    with pytest.raises(ValueError, match="nemotron_parse_path requires the dynamo backend"):
        create_nemotron_parse_inference_server(backend="ray-serve", nemotron_parse_path="/models/parse35")
