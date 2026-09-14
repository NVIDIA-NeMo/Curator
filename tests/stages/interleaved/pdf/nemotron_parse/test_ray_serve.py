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

"""Tests for Nemotron-Parse's Ray Serve worker customization."""

import asyncio

import pytest

ray_serve_llm = pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed")
ray_serve = pytest.importorskip(
    "nemo_curator.stages.interleaved.pdf.nemotron_parse.ray_serve",
    reason="ray[serve] not installed",
)


def test_attention_backend_is_resolved_inside_replica(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def set_backend(engine_kwargs: dict) -> None:
        calls.append("backend")
        engine_kwargs["attention_backend"] = "TRITON_ATTN"

    async def initialize_server(_self: object, llm_config: object) -> None:
        assert llm_config.engine_kwargs["attention_backend"] == "TRITON_ATTN"
        calls.append("server")

    monkeypatch.setattr(ray_serve, "set_nemotron_parse_attention_backend", set_backend)
    monkeypatch.setattr(ray_serve_llm.LLMServer, "__init__", initialize_server)

    server = object.__new__(ray_serve.NemotronParseRayServeServer)
    llm_config = ray_serve_llm.LLMConfig(model_loading_config={"model_id": "nemotron-parse"})
    asyncio.run(server.__init__(llm_config))

    assert calls == ["backend", "server"]
