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

from __future__ import annotations

import pytest

from nemo_curator.core.serve.dynamo.infra import build_worker_actor_name, engine_kwargs_to_cli_flags


@pytest.mark.parametrize(
    ("engine_kwargs", "expected"),
    [
        ({}, []),
        ({"enforce_eager": False}, []),
        ({"enforce_eager": True}, ["--enforce-eager"]),
        (
            {"tensor_parallel_size": 4, "max_model_len": 8192},
            ["--tensor-parallel-size", "4", "--max-model-len", "8192"],
        ),
        (
            {"served_model_name": ["alias1", "alias2"]},
            ["--served-model-name", '["alias1", "alias2"]'],
        ),
        (
            {"generation_config": {"temperature": 0.7}},
            ["--generation-config", '{"temperature": 0.7}'],
        ),
    ],
)
def test_engine_kwargs_to_cli_flags(engine_kwargs: dict, expected: list[str]) -> None:
    assert engine_kwargs_to_cli_flags(engine_kwargs) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("Qwen3-0.6B", 0, 0, 1, None), "Dynamo_DP0_Qwen3-0.6B"),
        (("Qwen/Qwen3-0.6B", 0, 0, 1, None), "Dynamo_DP0_Qwen3-0.6B"),
        (("Qwen3-0.6B", 1, 0, 4, None), "Dynamo_DP1_TP0_Qwen3-0.6B"),
        (("Qwen3-0.6B", 0, 2, 4, None), "Dynamo_DP0_TP2_Qwen3-0.6B"),
        (("Qwen3-0.6B", 0, 0, 2, "decode"), "Dynamo_decode_DP0_TP0_Qwen3-0.6B"),
        (("Qwen3-0.6B", 1, 0, 2, "prefill"), "Dynamo_prefill_DP1_TP0_Qwen3-0.6B"),
    ],
)
def test_build_worker_actor_name(args: tuple, expected: str) -> None:
    model, replica, rank, tp, role = args
    assert build_worker_actor_name(model, replica, rank, tp, role=role) == expected
