# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.backends.utils import merge_executor_configs


class TestMergeExecutorConfig:
    """Test class for merge_executor_configs function."""

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {
            "runtime_env": {
                "env_vars": {"A": "1", "B": "2"},
                "pip": ["package1"],
            },
            "other_config": "value1",
        }

        override = {
            "runtime_env": {
                "env_vars": {"B": "3", "C": "4"},
                "working_dir": ".",
            },
            "some_other_top_key": "value2",
        }

        result = merge_executor_configs(base, override)

        # Check that nested dicts are merged
        assert result["runtime_env"]["env_vars"]["A"] == "1"
        assert result["runtime_env"]["env_vars"]["B"] == "3"
        assert result["runtime_env"]["env_vars"]["C"] == "4"
        # Check that other keys are preserved
        assert result["runtime_env"]["pip"] == ["package1"]
        assert result["runtime_env"]["working_dir"] == "."
        assert result["other_config"] == "value1"
        assert result["some_other_top_key"] == "value2"

    def test_merge_with_none(self):
        """Test merging when base config is None."""
        assert merge_executor_configs(None, {"key": "value"}) == {"key": "value"}
        assert merge_executor_configs({"key": "value"}, None) == {"key": "value"}
        assert merge_executor_configs(None, None) == {}
