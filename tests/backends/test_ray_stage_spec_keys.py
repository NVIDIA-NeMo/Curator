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

"""Test RayStageSpecKeys enum membership in Python >= 3.10."""

import pytest

from nemo_curator.backends.experimental.utils import RayStageSpecKeys


class TestRayStageSpecKeys:
    """Test cases for RayStageSpecKeys enum membership checking."""

    def test_valid_keys_membership(self):
        """Test that valid keys are recognized as members."""
        valid_keys = [
            "is_actor_stage", 
            "is_fanout_stage", 
            "is_raft_actor", 
            "is_lsh_stage", 
            "is_shuffle_stage"
        ]
        
        for key in valid_keys:
            # This should work without throwing TypeError in all Python versions >= 3.10
            assert key in [item.value for item in RayStageSpecKeys], f"Valid key '{key}' should be recognized"

    def test_invalid_keys_membership(self):
        """Test that invalid keys are not recognized as members."""
        invalid_keys = [
            "invalid_key",
            "not_a_stage", 
            "random_string",
            "",
            None
        ]
        
        for key in invalid_keys:
            if key is not None:
                # This should work without throwing TypeError in all Python versions >= 3.10
                assert key not in [item.value for item in RayStageSpecKeys], f"Invalid key '{key}' should not be recognized"

    def test_enum_values_accessibility(self):
        """Test that enum values are accessible."""
        assert RayStageSpecKeys.IS_ACTOR_STAGE.value == "is_actor_stage"
        assert RayStageSpecKeys.IS_FANOUT_STAGE.value == "is_fanout_stage"
        assert RayStageSpecKeys.IS_RAFT_ACTOR.value == "is_raft_actor"
        assert RayStageSpecKeys.IS_LSH_STAGE.value == "is_lsh_stage"
        assert RayStageSpecKeys.IS_SHUFFLE_STAGE.value == "is_shuffle_stage"

    def test_enum_membership_with_values_set(self):
        """Test membership using enum values set - the approach used in the fix."""
        enum_values = {item.value for item in RayStageSpecKeys}
        
        # Test valid keys
        assert "is_actor_stage" in enum_values
        assert "is_fanout_stage" in enum_values
        
        # Test invalid keys  
        assert "invalid_key" not in enum_values
        assert "not_a_stage" not in enum_values

    def test_python_compatibility_approach(self):
        """Test that our fix approach works for Python 3.10+ compatibility."""
        # This tests the exact approach used in RayDataStageAdapter
        valid_enum_values = {item.value for item in RayStageSpecKeys}
        
        # Simulate ray_stage_spec() return values that should pass
        valid_specs = [
            {"is_actor_stage": True},
            {"is_fanout_stage": False, "is_lsh_stage": True},
            {}  # empty spec should also work
        ]
        
        for spec in valid_specs:
            for key in spec:
                assert key in valid_enum_values, f"Valid key '{key}' should be accepted"
        
        # Simulate ray_stage_spec() return values that should fail
        invalid_specs = [
            {"invalid_key": True},
            {"is_actor_stage": True, "bad_key": False},
        ]
        
        for spec in invalid_specs:
            invalid_keys = [key for key in spec if key not in valid_enum_values]
            assert len(invalid_keys) > 0, f"Spec {spec} should have invalid keys"