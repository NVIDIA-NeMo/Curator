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

import pytest

from nemo_curator.utils.column_utils import resolve_filename_column


class TestResolveFilenameColumn:
    """Test cases for resolve_filename_column function."""

    def test_resolve_filename_column_true(self):
        """Test that True returns the default filename column name."""
        assert resolve_filename_column(True) == "file_name"

        assert resolve_filename_column(False) is None

        custom_name = "custom_filename"
        assert resolve_filename_column(custom_name) == custom_name

        assert resolve_filename_column("") == ""

        with pytest.raises(ValueError, match="Invalid value for add_filename_column"):
            resolve_filename_column(None)  # type: ignore[reportArgumentType]
