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
from nemo_curator.core.serve.base import BaseModelConfig
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
        assert env["uv"] == ["ai-dynamo[vllm]", "mypkg==1.0"]

    def test_user_env_vars_are_preserved(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"env_vars": {"HF_TOKEN": "abc", "TRANSFORMERS_OFFLINE": "1"}},
        )
        env = dynamo_runtime_env(mc)
        assert env["uv"] == ["ai-dynamo[vllm]"]
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
        assert env["uv"] == ["ai-dynamo[vllm]", "userpkg"]

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


class TestMergeRuntimeEnvsDictForm:
    """Cover ``BaseModelConfig.merge_runtime_envs`` paths where ``pip`` / ``uv``
    arrive in Ray's structured dict form (``{"packages": [...],
    "uv_pip_install_options": [...]}``) rather than the legacy list form."""

    def test_dict_base_list_override_preserves_install_options(self) -> None:
        base = {"uv": {"packages": ["pkg-a"], "uv_pip_install_options": ["--no-cache"]}}
        override = {"uv": ["pkg-b"]}
        env = BaseModelConfig.merge_runtime_envs(base, override)

        assert env["uv"]["packages"] == ["pkg-a", "pkg-b"]
        assert env["uv"]["uv_pip_install_options"] == ["--no-cache"]

    def test_list_base_dict_override_carries_extra_keys(self) -> None:
        base = {"pip": ["pkg-a"]}
        override = {"pip": {"packages": ["pkg-b"], "pip_check": True}}
        env = BaseModelConfig.merge_runtime_envs(base, override)

        assert env["pip"]["packages"] == ["pkg-a", "pkg-b"]
        assert env["pip"]["pip_check"] is True

    def test_dict_base_dict_override_concatenates_options(self) -> None:
        base = {"pip": {"packages": ["pkg-a"], "pip_install_options": ["--no-cache"]}}
        override = {"pip": {"packages": ["pkg-b"], "pip_install_options": ["--prefer-binary"]}}
        env = BaseModelConfig.merge_runtime_envs(base, override)

        assert env["pip"]["packages"] == ["pkg-a", "pkg-b"]
        assert env["pip"]["pip_install_options"] == ["--no-cache", "--prefer-binary"]

    def test_dict_form_base_alone_is_returned_verbatim(self) -> None:
        # User did not override ``uv`` — base's installer options must survive.
        base = {"uv": {"packages": ["pkg-a"], "uv_pip_install_options": ["--no-cache"]}}
        env = BaseModelConfig.merge_runtime_envs(base, {"env_vars": {"X": "1"}})

        assert env["uv"] == base["uv"]
        assert env["env_vars"] == {"X": "1"}
