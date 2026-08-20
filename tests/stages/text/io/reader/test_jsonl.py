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

import json
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader, JsonlReaderStage
from nemo_curator.tasks import EmptyTask, FileGroupTask


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
        FileGroupTask(dataset_name="test_dataset", data=[file_path], _metadata={})
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

    def test_default_reader_uses_pyarrow_fast_path(self, sample_jsonl_files: list[str]) -> None:
        """The default reader should retain its DataFrame contract with Arrow-backed strings."""
        task = FileGroupTask(dataset_name="ds", data=sample_jsonl_files, _metadata={})

        result = JsonlReaderStage(fields=["text"]).process(task)

        assert isinstance(result.data, pd.DataFrame)
        assert result.data["text"].dtype.storage == "pyarrow"
        assert result.data["text"].tolist() == [
            "Doc 0-1",
            "Doc 0-2",
            "Doc 1-1",
            "Doc 1-2",
            "Doc 2-1",
            "Doc 2-2",
        ]

    def test_auto_reader_falls_back_to_pandas_for_mixed_column_types(self, tmp_path: Path) -> None:
        """Auto mode should preserve compatibility when PyArrow rejects mixed JSON types."""
        file_path = tmp_path / "mixed.jsonl"
        file_path.write_text('{"value":1}\n{"value":"one"}\n', encoding="utf-8")
        task = FileGroupTask(dataset_name="ds", data=[str(file_path)], _metadata={})

        result = JsonlReaderStage(read_kwargs={"engine": "auto"}).process(task)

        assert result.data["value"].tolist() == [1, "one"]

    def test_pandas_and_pyarrow_direct_document_inference_difference(self, tmp_path: Path) -> None:
        """Document when callers need pandas inference instead of the faster direct parser."""
        file_path = tmp_path / "timestamp.jsonl"
        expected = pd.Timestamp("2026-08-20T12:34:56Z")
        pd.DataFrame({"created_at": [expected], "text": ["hello"]}).to_json(
            file_path,
            orient="records",
            lines=True,
            date_format="iso",
        )
        task = FileGroupTask(dataset_name="ds", data=[str(file_path)], _metadata={})

        pandas_result = JsonlReaderStage(read_kwargs={"engine": "pandas"}).process(task).data
        arrow_result = JsonlReaderStage(read_kwargs={"engine": "pyarrow_direct"}).process(task).data
        auto_with_pandas_option = JsonlReaderStage(read_kwargs={"convert_dates": ["created_at"]}).process(task).data

        # Prefer the default/pyarrow_direct path for throughput. Select pandas when
        # pandas-specific inference is required; auto also preserves pandas-only options.
        assert pandas_result["created_at"].tolist() == [expected]
        assert auto_with_pandas_option["created_at"].tolist() == [expected]
        assert isinstance(arrow_result["created_at"].iloc[0], str)
        assert pd.Timestamp(arrow_result["created_at"].iloc[0]) == expected

    def test_pyarrow_direct_does_not_fall_back(self, tmp_path: Path) -> None:
        """The explicit direct engine should expose unsupported Arrow input instead of silently changing engines."""
        file_path = tmp_path / "mixed.jsonl"
        file_path.write_text('{"value":1}\n{"value":"one"}\n', encoding="utf-8")
        task = FileGroupTask(dataset_name="ds", data=[str(file_path)], _metadata={})

        with pytest.raises(pa.ArrowInvalid, match="changed from number to string"):
            JsonlReaderStage(read_kwargs={"engine": "pyarrow_direct"}).process(task)

    def test_pyarrow_direct_grows_block_for_large_record(self, tmp_path: Path) -> None:
        """A record larger than the initial parse block should be retried with a larger block."""
        text = "x" * (1024 * 1024 + 64 * 1024)
        file_path = tmp_path / "large.jsonl"
        file_path.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
        task = FileGroupTask(dataset_name="ds", data=[str(file_path)], _metadata={})
        stage = JsonlReaderStage(
            read_kwargs={
                "engine": "pyarrow_direct",
                "pyarrow_block_size": 1024 * 1024,
                "pyarrow_max_block_size": 2 * 1024 * 1024,
            }
        )

        result = stage.process(task)

        assert result.data["text"].tolist() == [text]

    def test_pyarrow_direct_reads_fsspec_url(self) -> None:
        """The direct engine should retain JsonlReader's remote-filesystem behavior."""
        file_path = "memory://jsonl-reader/direct.jsonl"
        with fsspec.open(file_path, mode="wt", encoding="utf-8") as stream:
            stream.write('{"text":"first"}\n{"text":"second"}\n')
        task = FileGroupTask(dataset_name="ds", data=[file_path], _metadata={})

        result = JsonlReaderStage(read_kwargs={"engine": "pyarrow_direct"}).process(task)

        assert result.data["text"].tolist() == ["first", "second"]

    def test_storage_options_via_read_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reader should use storage options from reader.read_kwargs."""
        # Create a file
        file_path = tmp_path / "one.jsonl"
        pd.DataFrame({"a": [1]}).to_json(file_path, orient="records", lines=True)

        # Reader uses read_kwargs storage options
        task = FileGroupTask(dataset_name="ds", data=[str(file_path)], _metadata={})
        stage = JsonlReaderStage(read_kwargs={"engine": "pandas", "storage_options": {"auto_mkdir": True}})

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
        task = FileGroupTask(dataset_name="ds", data=[str(f)], _metadata={})
        stage = JsonlReaderStage(read_kwargs={"engine": "pandas", "storage_options": {"auto_mkdir": True}})
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


def test_jsonl_reader_with_blocksize_limit(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # Storage size is larger than 10 million bytes
    # In-memory size is also larger than 10 million bytes
    size = 1000
    df = pd.DataFrame({"id": list(range(size)), "text": ["a" * 4000] * size, "other_field": ["b" * 10_000] * size})
    df.to_json(tmp_path / "test.jsonl", orient="records", lines=True)

    stage = JsonlReader(file_paths=str(tmp_path), blocksize=10_000_000)
    assert len(stage.decompose()) == 2

    # Since the storage size is larger than 10 million bytes, the FilePartitioningStage should warn
    file_partitioning_stage = stage.decompose()[0]
    with caplog.at_level("WARNING"):
        file_partitioning_stage.process(EmptyTask)
    assert "File group task has exceeded the storage limit per partition" in caplog.text
