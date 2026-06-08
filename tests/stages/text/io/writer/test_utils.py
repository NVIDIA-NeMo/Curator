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

from nemo_curator.stages.text.io.writer.utils import get_deterministic_hash


class TestGetDeterministicHash:
    """Test suite for get_deterministic_hash function."""

    def test_consistent_output_same_inputs(self):
        inputs1 = ["file1.txt", "file2.txt"]
        inputs2 = ["file3.txt", "file4.txt"]
        seed = "test_seed"

        """ Test that the same inputs produce the same hash. """
        assert get_deterministic_hash(inputs1, seed) == get_deterministic_hash(inputs1, seed)

        """Test that different inputs produce different hashes."""
        assert get_deterministic_hash(inputs1, seed) != get_deterministic_hash(inputs2, seed)

        """Test that different seeds produce different hashes."""
        assert get_deterministic_hash(inputs1, "seed1") != get_deterministic_hash(inputs1, "seed2")

        """ Test empty seed also produces a hash. """
        assert get_deterministic_hash(inputs1, "") == get_deterministic_hash(inputs1, "")
        assert get_deterministic_hash(inputs1, "") != get_deterministic_hash(inputs2, "")

    def test_order_independence(self):
        """Test that input order doesn't affect the hash (inputs are sorted)."""
        inputs1 = ["file1.txt", "file2.txt", "file3.txt"]
        inputs2 = ["file3.txt", "file1.txt", "file2.txt"]
        seed = "test_seed"

        assert get_deterministic_hash(inputs1, seed) == get_deterministic_hash(inputs2, seed)

        """ Test if there are duplicate inputs, the hash is the same due to sorting. """
        inputs3 = ["file1.txt", "file2.txt", "file1.txt"]
        inputs4 = ["file1.txt", "file1.txt", "file2.txt"]
        assert get_deterministic_hash(inputs3, seed) == get_deterministic_hash(inputs4, seed)

    def test_empty_inputs(self):
        """Test behavior with empty inputs list."""
        inputs = []
        seed = "test_seed"

        hash_result = get_deterministic_hash(inputs, seed)

        assert len(hash_result) == 12
        assert all(c in "0123456789abcdef" for c in hash_result)
