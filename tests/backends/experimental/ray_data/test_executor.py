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

from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader


class TestRayDataExecutorFailFast:
    """Test fail-fast behavior when reader finds no files (wrong path or wrong file type)."""

    def test_jsonl_reader_on_empty_dir_fails_fast(self, tmp_path: Path):
        """JsonlReader on empty directory should raise immediately, not run forever."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        pipeline = Pipeline(name="test_fail_fast")
        pipeline.add_stage(JsonlReader(str(empty_dir)))
        pipeline.build()

        executor = RayDataExecutor()

        with pytest.raises(ValueError, match="No files found or no tasks produced"):
            pipeline.run(executor=executor)

    def test_jsonl_reader_on_nonexistent_path_fails_fast(self, tmp_path: Path):
        """JsonlReader on non-existent path should raise immediately."""
        nonexistent = tmp_path / "does_not_exist" / "nested"

        pipeline = Pipeline(name="test_fail_fast")
        pipeline.add_stage(JsonlReader(str(nonexistent)))
        pipeline.build()

        executor = RayDataExecutor()

        with pytest.raises(ValueError, match="No files found or no tasks produced"):
            pipeline.run(executor=executor)

    def test_jsonl_reader_on_parquet_data_fails_fast(self, tmp_path: Path):
        """JsonlReader on directory of Parquet files (wrong file type) should raise immediately."""
        parquet_dir = tmp_path / "parquet_data"
        parquet_dir.mkdir()

        df = pd.DataFrame({"text": ["doc1", "doc2"], "id": [1, 2]})
        df.to_parquet(parquet_dir / "data.parquet", index=False)

        pipeline = Pipeline(name="test_fail_fast")
        pipeline.add_stage(JsonlReader(str(parquet_dir)))
        pipeline.build()

        executor = RayDataExecutor()

        with pytest.raises(ValueError, match="No files found or no tasks produced"):
            pipeline.run(executor=executor)
