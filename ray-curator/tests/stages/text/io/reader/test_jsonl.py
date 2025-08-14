"""Simple tests for JsonlReaderStage ID generation functionality."""

from pathlib import Path

import pandas as pd
import pytest

from ray_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
)
from ray_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from ray_curator.tasks import FileGroupTask


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
