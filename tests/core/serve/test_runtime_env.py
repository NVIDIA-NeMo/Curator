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

"""Runtime-env merging behaviour exposed through the Dynamo backend.

These tests live at the serve-package level (not the dynamo subpackage)
because ``runtime_env`` is a user-facing promise of ``BaseModelConfig``:
Curator-owned defaults (e.g. ``ai-dynamo[vllm]``) must not clobber a
user's ``env_vars``, ``pip``, ``uv``, or ``working_dir``.
"""

from __future__ import annotations

from nemo_curator.core.serve import DynamoVLLMModelConfig
from nemo_curator.core.serve.dynamo.vllm import (
    DYNAMO_VLLM_RUNTIME_ENV,
    dynamo_runtime_env,
    merge_model_runtime_envs,
)


class TestDynamoRuntimeEnv:
    def test_default_runtime_env_only(self) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B")
        env = dynamo_runtime_env(mc)
        assert env == DYNAMO_VLLM_RUNTIME_ENV

    def test_user_runtime_env_merges_uv_packages(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"uv": ["mypkg==1.0"]},
        )
        env = dynamo_runtime_env(mc)
        expected_packages = [*DYNAMO_VLLM_RUNTIME_ENV["uv"]["packages"], "mypkg==1.0"]
        assert env["uv"]["packages"] == expected_packages
        assert env["uv"]["uv_pip_install_options"] == DYNAMO_VLLM_RUNTIME_ENV["uv"]["uv_pip_install_options"]

    def test_user_env_vars_are_preserved(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"env_vars": {"HF_TOKEN": "abc", "TRANSFORMERS_OFFLINE": "1"}},
        )
        env = dynamo_runtime_env(mc)
        assert env["uv"] == DYNAMO_VLLM_RUNTIME_ENV["uv"]
        assert env["env_vars"] == {"HF_TOKEN": "abc", "TRANSFORMERS_OFFLINE": "1"}

    def test_working_dir_is_passed_through(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"working_dir": "/workspace"},
        )
        env = dynamo_runtime_env(mc)
        assert env["working_dir"] == "/workspace"


class TestMergeModelRuntimeEnvs:
    def test_no_models_yields_default_env(self) -> None:
        assert merge_model_runtime_envs([]) == DYNAMO_VLLM_RUNTIME_ENV

    def test_merges_env_vars_across_models(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m1",
                runtime_env={"env_vars": {"A": "1"}},
            ),
            DynamoVLLMModelConfig(
                model_identifier="m2",
                runtime_env={"env_vars": {"B": "2"}, "uv": ["userpkg"]},
            ),
        ]
        env = merge_model_runtime_envs(models)
        assert env["env_vars"] == {"A": "1", "B": "2"}
        expected_packages = [*DYNAMO_VLLM_RUNTIME_ENV["uv"]["packages"], "userpkg"]
        assert env["uv"]["packages"] == expected_packages

    def test_later_model_env_var_overrides_earlier(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m1",
                runtime_env={"env_vars": {"HF_HOME": "/cache/v1"}},
            ),
            DynamoVLLMModelConfig(
                model_identifier="m2",
                runtime_env={"env_vars": {"HF_HOME": "/cache/v2"}},
            ),
        ]
        env = merge_model_runtime_envs(models)
        assert env["env_vars"]["HF_HOME"] == "/cache/v2"

    def test_ignores_models_with_empty_runtime_env(self) -> None:
        models = [
            DynamoVLLMModelConfig(model_identifier="m1"),
            DynamoVLLMModelConfig(
                model_identifier="m2",
                runtime_env={"env_vars": {"A": "1"}},
            ),
        ]
        env = merge_model_runtime_envs(models)
        assert env["env_vars"] == {"A": "1"}
