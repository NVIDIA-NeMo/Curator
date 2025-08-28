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

from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader, JsonlReaderStage
from nemo_curator.tasks import FileGroupTask


@pytest.fixture
def sample_jsonl_files(tmp_path: Path) -> list[str]:
    """Create multiple JSONL files for testing."""
    files = []
    for i in range(3):
        data = pd.DataFrame({"text": [f"Doc {i}-1", f"Doc {i}-2"]})
        file_path = tmp_path / f"test_{i}.jsonl"
        data.to_json(file_path, orient="records", lines=True)
        files.append(str(file_path))
    return files


@pytest.fixture
def file_group_tasks(sample_jsonl_files: list[str]) -> list[FileGroupTask]:
    """Create multiple FileGroupTasks."""
    return [
        FileGroupTask(task_id=f"task_{i}", dataset_name="test_dataset", data=[file_path], _metadata={})
        for i, file_path in enumerate(sample_jsonl_files)
    ]


class TestJsonlReaderWithoutIdGenerator:
    """Test JSONL reader without ID generation."""

    def test_processing_without_ids(self, file_group_tasks: list[FileGroupTask]) -> None:
        """Test processing without ID generation."""
        for task in file_group_tasks:
            stage = JsonlReaderStage()
            result = stage.process(task)
            df = result.to_pandas()
            assert CURATOR_DEDUP_ID_STR not in df.columns
            assert len(df) == 2  # Each file has 2 rows

    def test_columns_selection(self, file_group_tasks: list[FileGroupTask]) -> None:
        """When columns are provided, only those are returned (existing ones)."""
        for task in file_group_tasks:
            stage = JsonlReaderStage(fields=["text"])  # select single column
            result = stage.process(task)
            df = result.to_pandas()
            assert list(df.columns) == ["text"]
            assert len(df) == 2

    def test_storage_options_via_read_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reader should use storage options from reader.read_kwargs."""
        # Create a file
        file_path = tmp_path / "one.jsonl"
        pd.DataFrame({"a": [1]}).to_json(file_path, orient="records", lines=True)

        # Reader uses read_kwargs storage options
        task = FileGroupTask(task_id="t1", dataset_name="ds", data=[str(file_path)], _metadata={})
        stage = JsonlReaderStage(read_kwargs={"storage_options": {"auto_mkdir": True}})

        seen: dict[str, object] = {}

        def fake_read_json(_path: object, *_args: object, **kwargs: object) -> pd.DataFrame:
            seen["storage_options"] = kwargs.get("storage_options") if isinstance(kwargs, dict) else None
            return pd.DataFrame({"a": [1]})

        monkeypatch.setattr(pd, "read_json", fake_read_json)

        out = stage.process(task)
        assert seen["storage_options"] == {"auto_mkdir": True}
        df = out.to_pandas()
        assert len(df) == 1

    def test_composite_reader_propagates_storage_options(self, tmp_path: Path) -> None:
        """Composite JsonlReader should pass storage options to partitioning stage and underlying stage."""
        f = tmp_path / "a.jsonl"
        pd.DataFrame({"text": ["x"]}).to_json(f, orient="records", lines=True)
        reader = JsonlReader(
            file_paths=str(tmp_path), read_kwargs={"storage_options": {"anon": True}}, fields=["text"]
        )
        stages = reader.decompose()
        # First stage is file partitioning, ensure storage options are set
        first = stages[0]
        assert getattr(first, "storage_options", None) == {"anon": True}

    def test_reader_uses_storage_options_from_read_kwargs_when_task_has_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "b.jsonl"
        pd.DataFrame({"x": [1, 2]}).to_json(f, orient="records", lines=True)

        seen: dict[str, object] = {}

        def fake_read_json(_path: object, *_args: object, **kwargs: object) -> pd.DataFrame:
            seen["storage_options"] = kwargs.get("storage_options") if isinstance(kwargs, dict) else None
            return pd.DataFrame({"x": [1, 2]})

        monkeypatch.setattr(pd, "read_json", fake_read_json)
        task = FileGroupTask(task_id="t2", dataset_name="ds", data=[str(f)], _metadata={})
        stage = JsonlReaderStage(read_kwargs={"storage_options": {"auto_mkdir": True}})
        out = stage.process(task)
        assert seen["storage_options"] == {"auto_mkdir": True}
        df = out.to_pandas()
        assert len(df) == 2


class TestJsonlReaderWithIdGenerator:
    """Test JSONL reader with ID generation."""

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_sequential_id_generation_and_assignment(self, file_group_tasks: list[FileGroupTask]) -> None:
        """Test sequential ID generation across multiple batches."""
        generation_stage = JsonlReaderStage(_generate_ids=True)
        generation_stage.setup()

        all_ids = []
        for task in file_group_tasks:
            result = generation_stage.process(task)
            ids = result.to_pandas()[CURATOR_DEDUP_ID_STR].tolist()
            all_ids.extend(ids)

        # IDs should be monotonically increasing: [0,1,2,3,4,5]
        assert all_ids == list(range(6))

        """If the same batch is processed again (when generate_id=True), the IDs should be the same."""
        repeated_ids = []
        for task in file_group_tasks:
            result = generation_stage.process(task)
            ids = result.to_pandas()[CURATOR_DEDUP_ID_STR].tolist()
            repeated_ids.extend(ids)

        # IDs should be the same as the first time: [0,1,2,3,4,5]
        assert repeated_ids == list(range(6))

        """ If we now create a new stage with _assign_ids=True, the IDs should be the same as the previous batch."""
        all_ids = []
        assign_stage = JsonlReaderStage(_assign_ids=True)
        assign_stage.setup()
        for i, task in enumerate(file_group_tasks):
            result = assign_stage.process(task)
            df = result.to_pandas()
            expected_ids = [i * 2, i * 2 + 1]  # Task 0: [0,1], Task 1: [2,3], Task 2: [4,5]
            assert (
                df[CURATOR_DEDUP_ID_STR].tolist() == expected_ids
            )  # These ids should be the same as the previous batch
            all_ids.extend(df[CURATOR_DEDUP_ID_STR].tolist())

        assert all_ids == list(range(6))

    def test_generate_ids_no_actor_error(self) -> None:
        """Test error when actor doesn't exist and ID generation is requested."""
        stage = JsonlReaderStage(_generate_ids=True)

        with pytest.raises(RuntimeError, match="actor 'id_generator' does not exist"):
            stage.setup()

        stage = JsonlReaderStage(_assign_ids=True)

        with pytest.raises(RuntimeError, match="actor 'id_generator' does not exist"):
            stage.setup()
