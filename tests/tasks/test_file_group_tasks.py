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

from nemo_curator.tasks import FileGroupTask


class TestFileGroupTask:
    def test_deterministic_ids(self) -> None:
        """``get_deterministic_id`` hashes the sorted file paths, so it is:
        order-independent, distinct for distinct file sets, and a 12-char
        hex string."""
        # Order-independent: same files in different orders → same id.
        a = FileGroupTask(dataset_name="d", data=["b.parquet", "a.parquet"])
        b = FileGroupTask(dataset_name="d", data=["a.parquet", "b.parquet"])
        assert a.get_deterministic_id() == b.get_deterministic_id()

        # Distinct file sets → distinct ids.
        c = FileGroupTask(dataset_name="d", data=["c.parquet"])
        assert a.get_deterministic_id() != c.get_deterministic_id()

        # 12-char hex string.
        result = c.get_deterministic_id()
        assert isinstance(result, str)
        assert len(result) == 12
        assert all(ch in "0123456789abcdef" for ch in result)
