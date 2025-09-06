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

"""Test the compatibility fix for RayStageSpecKeys enum membership check."""

import sys

from nemo_curator.backends.experimental.utils import RayStageSpecKeys


def test_enum_membership_compatibility():
    """Test that the fixed pattern works across Python versions."""
    print(f"Python version: {sys.version_info}")
    print(f"Enum has __contains__: {hasattr(RayStageSpecKeys, '__contains__')}")

    # Test data
    valid_keys = ["is_actor_stage", "is_fanout_stage", "is_lsh_stage"]
    invalid_keys = ["invalid_key", "another_bad_key"]

    # Test the fixed pattern - this is what's now used in the adapter
    enum_values = {e.value for e in RayStageSpecKeys}

    print("\nTesting valid keys:")
    for key in valid_keys:
        result = key not in enum_values
        print(f"  '{key}' not in enum_values: {result} (should be False)")
        assert result is False, f"Valid key '{key}' should be found in enum values"

    print("\nTesting invalid keys:")
    for key in invalid_keys:
        result = key not in enum_values
        print(f"  '{key}' not in enum_values: {result} (should be True)")
        assert result is True, f"Invalid key '{key}' should not be found in enum values"

    print("\n✅ All tests passed! The fix works correctly.")


if __name__ == "__main__":
    test_enum_membership_compatibility()
