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

import pytest

from nemo_curator.stages.text.utils.text_utils import is_paragraph_indices_in_top_or_bottom_only


class TestIsParagraphIndicesInTopOrBottomOnly:
    @pytest.mark.parametrize(
        ("indices", "expected"),
        [
            ([0], True),
            ([0, 1], True),
            ([10], True),
            ([9, 10], True),
            ([0, 1, 9, 10], True),
            ([0, 10], True),
            ([5], False),
            ([1, 2], False),
            ([8, 9], False),
            ([0, 1, 3, 9, 10], False),
            ([0, 1, 3, 5, 6, 9, 10], False),
            ([0, 2], False),
            ([8, 10], False),
            (list(range(11)), False),
        ],
    )
    def test_indices(self, indices: list[int], expected: bool) -> None:
        assert is_paragraph_indices_in_top_or_bottom_only(indices, 11) is expected
