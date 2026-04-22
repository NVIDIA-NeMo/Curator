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

from nemo_curator.core.serve import BaseModelConfig


class TestBaseModelConfig:
    def test_merge_runtime_envs_merges_packages_and_env_vars(self) -> None:
        base = {
            "pip": ["pkg-a"],
            "uv": ["uv-a"],
            "env_vars": {"A": "1", "B": "1"},
            "working_dir": ".",
        }
        override = {
            "pip": ["pkg-b"],
            "uv": ["uv-b"],
            "env_vars": {"B": "2", "C": "3"},
        }

        result = BaseModelConfig.merge_runtime_envs(base, override)

        assert result["pip"] == ["pkg-a", "pkg-b"]
        assert result["uv"] == ["uv-a", "uv-b"]
        assert result["env_vars"] == {"A": "1", "B": "2", "C": "3"}
        assert result["working_dir"] == "."
