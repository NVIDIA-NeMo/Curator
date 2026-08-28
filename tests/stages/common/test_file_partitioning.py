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

import os
import zipfile
from pathlib import Path

import fsspec
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import EmptyTask, FileGroupTask


def _create_test_jsonl_files(base_dir: Path | str, num_files: int, subdir: str | None = None) -> list[str]:
    """Create num_files minimal JSONL files in base_dir[/subdir] and return their paths."""
    base = Path(base_dir)
    target_dir = base / subdir if subdir else base
    os.makedirs(target_dir, exist_ok=True)
    files: list[str] = []
    for i in range(num_files):
        file_path = target_dir / f"file{i}.jsonl"
        file_path.write_text("{}\n")
        files.append(str(file_path))
    return files


class TestFilePartitioningStage:
    """Test suite for FilePartitioningStage."""

    @pytest.fixture
    def temp_files(self, tmp_path: Path) -> list[str]:
        """Create temporary test files for testing."""
        files = []
        for i in range(10):
            file_path = tmp_path / f"test_file_{i}.jsonl"
            file_path.write_text(f'{{"id": {i}, "text": "Test content {i}"}}')
            files.append(str(file_path))
        return files

    @pytest.fixture
    def empty_task(self) -> EmptyTask:
        """Create an empty task for testing."""
        return EmptyTask(
            dataset_name="test_dataset",
            data=None,
            _metadata={"source": "test"},
        )

    def test_initialization_default_values(self):
        """Test initialization with default parameter values."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.file_paths == "/test/path"
        assert stage.files_per_partition is None
        assert stage.blocksize is None
        assert stage.file_extensions == [".jsonl", ".json", ".parquet"]
        assert stage.storage_options == {}
        assert stage.limit is None
        assert stage.name == "file_partitioning"
        assert stage.allow_empty is False

    def test_initialization_custom_values_with_files_per_partition(self):
        """Test initialization with custom parameter values using files_per_partition."""
        stage = FilePartitioningStage(
            file_paths="/custom/path",
            files_per_partition=5,
            file_extensions=[".txt", ".json"],
            storage_options={"key": "value"},
            limit=3,
        )

        assert stage.file_paths == "/custom/path"
        assert stage.files_per_partition == 5
        assert stage.blocksize is None
        assert stage.file_extensions == [".txt", ".json"]
        assert stage.storage_options == {"key": "value"}
        assert stage.limit == 3

    def test_initialization_custom_values_with_blocksize(self):
        """Test initialization with custom parameter values using blocksize."""
        stage = FilePartitioningStage(
            file_paths="/custom/path",
            blocksize="128MB",
            file_extensions=[".txt", ".json"],
            storage_options={"key": "value"},
            limit=3,
        )

        assert stage.file_paths == "/custom/path"
        assert stage.files_per_partition is None
        assert stage.blocksize == "128MB"
        assert stage.file_extensions == [".txt", ".json"]
        assert stage.storage_options == {"key": "value"}
        assert stage.limit == 3

    def test_inputs_outputs(self):
        """Test inputs and outputs methods."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.inputs() == ([], [])
        assert stage.outputs() == ([], [])

    def test_resources(self):
        """Test resource requirements."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.resources.cpus == 0.5

    def test_ray_stage_spec(self):
        """Test ray stage specification."""
        stage = FilePartitioningStage(file_paths="/test/path")

        spec = stage.ray_stage_spec()
        assert spec["is_fanout_stage"] is True

    def test_worker_defaults(self):
        """Test worker defaults for source partitioning."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.num_workers() == 1
        assert stage.xenna_stage_spec() == {}

    def test_process_with_file_list(self, empty_task: EmptyTask, tmp_path: Path):
        """Test processing with a list of files."""
        # Create these files in the tmp_path:
        test_files = _create_test_jsonl_files(tmp_path, num_files=3, subdir="path")
        stage = FilePartitioningStage(file_paths=test_files)

        result = stage.process(empty_task)

        assert len(result) == len(test_files)  # 3 files in 3 groups
        assert isinstance(result[0], FileGroupTask)
        assert result[0].data == [test_files[0]]
        assert result[0].dataset_name == "path"

    def test_relative_directory_is_resolved_before_worker_cwd_changes(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Local paths retain driver-relative meaning when workers use another cwd."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        test_files = _create_test_jsonl_files(driver_dir, num_files=2, subdir="manifests")
        worker_dir.mkdir()

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths=Path("manifests"))
        assert stage.file_paths == "manifests"
        assert stage._get_file_paths_for_discovery() == str(driver_dir / "manifests")

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)

        assert [task.data[0] for task in result] == test_files

    def test_worker_relative_fallback_supports_executor_working_dir(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An executor-level working directory can supply a source absent on the driver."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_dir.mkdir()
        worker_files = _create_test_jsonl_files(worker_dir, num_files=2, subdir="manifests")

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="manifests")
        assert stage._get_discovery_path_candidates("manifests") == [
            str(driver_dir / "manifests"),
            "manifests",
        ]

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)

        assert [task.data[0] for task in result] == worker_files

    def test_construction_anchored_candidate_wins_over_worker_relative_fallback(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A worker-relative candidate is not consulted when the anchored source matches."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_files = _create_test_jsonl_files(driver_dir, num_files=1, subdir="manifests")
        _create_test_jsonl_files(worker_dir, num_files=2, subdir="manifests")

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="manifests")

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)

        assert [task.data[0] for task in result] == driver_files

    def test_list_fallback_is_per_path_and_deduplicates_records(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Each list item falls back independently and repeated matches emit once."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_files = _create_test_jsonl_files(driver_dir, num_files=1, subdir="anchored")
        worker_files = _create_test_jsonl_files(worker_dir, num_files=1, subdir="relative")

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(
            file_paths=[
                "anchored/*.jsonl",
                "relative/*.jsonl",
                "relative/*.jsonl",
                "relative/file0.jsonl",
            ]
        )

        monkeypatch.chdir(worker_dir)
        records, attempted_paths = stage._discover_file_list_with_sizes(sort_by_size=False)
        result = stage.process(empty_task)

        assert [path for path, _ in records] == sorted([*driver_files, *worker_files])
        assert len(attempted_paths) == len(set(attempted_paths))
        assert [task.data[0] for task in result] == sorted([*driver_files, *worker_files])

    def test_relative_list_and_globs_are_resolved_before_worker_cwd_changes(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Every local entry in a list, including globs, is frozen on the driver."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        glob_files = _create_test_jsonl_files(driver_dir, num_files=2, subdir="globbed")
        listed_file = _create_test_jsonl_files(driver_dir, num_files=1, subdir="listed")[0]
        worker_dir.mkdir()

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths=[Path("globbed/*.jsonl"), Path("listed/file0.jsonl")])
        assert stage.file_paths == ["globbed/*.jsonl", "listed/file0.jsonl"]
        assert stage._get_file_paths_for_discovery() == [
            str(driver_dir / "globbed" / "*.jsonl"),
            listed_file,
        ]

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)

        assert [task.data[0] for task in result] == sorted([*glob_files, listed_file])

    def test_runtime_working_dir_keeps_plain_relative_path(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A Ray runtime working directory owns the meaning of a relative path."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_dir.mkdir()
        worker_files = _create_test_jsonl_files(worker_dir, num_files=1, subdir="manifests")

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="manifests").with_(runtime_env={"working_dir": "."})
        assert stage._get_file_paths_for_discovery() == "manifests"

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)

        assert [task.data[0] for task in result] == worker_files

    def test_relative_file_uri_is_resolved_before_worker_cwd_changes(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A relative local URI is anchored without dropping its protocol."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        test_files = _create_test_jsonl_files(driver_dir, num_files=1, subdir="manifests")
        worker_dir.mkdir()

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="file://./manifests")
        assert stage._get_file_paths_for_discovery() == f"file://{driver_dir}/manifests"

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)

        assert [task.data[0] for task in result] == test_files

    def test_chained_filesystem_local_target_is_anchored(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Local targets in an fsspec chain are anchored while the outer protocol survives."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_dir.mkdir()
        worker_dir.mkdir()
        archive_path = driver_dir / "archive.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("manifest.jsonl", "{}\n")

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="zip://*.jsonl::archive.zip")
        discovery_path = stage._get_file_paths_for_discovery()
        assert discovery_path == f"zip://*.jsonl::{archive_path}"

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)
        emitted_path = result[0].data[0]

        assert emitted_path == f"zip://manifest.jsonl::{archive_path}"
        with fsspec.open(emitted_path, mode="rt") as manifest:
            assert manifest.read() == "{}\n"

    def test_chained_filesystem_retries_worker_relative_local_target(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A missing anchored archive falls back without rewriting the chain protocol."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_dir.mkdir()
        worker_dir.mkdir()
        archive_path = worker_dir / "archive.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("manifest.jsonl", "{}\n")

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="zip://*.jsonl::archive.zip")
        assert stage._get_discovery_path_candidates(stage.file_paths) == [
            f"zip://*.jsonl::{driver_dir}/archive.zip",
            "zip://*.jsonl::archive.zip",
        ]

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)
        emitted_path = result[0].data[0]

        assert emitted_path == "zip://manifest.jsonl::archive.zip"
        with fsspec.open(emitted_path, mode="rt") as manifest:
            assert manifest.read() == "{}\n"

    def test_cache_chain_is_preserved_while_local_target_is_anchored(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A protocol-only cache wrapper must not become part of a local filename."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        test_files = _create_test_jsonl_files(driver_dir, num_files=1, subdir="manifests")
        worker_dir.mkdir()

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="simplecache::manifests/*.jsonl")
        discovery_path = stage._get_file_paths_for_discovery()
        assert discovery_path == f"simplecache::{driver_dir}/manifests/*.jsonl"

        monkeypatch.chdir(worker_dir)
        result = stage.process(empty_task)
        emitted_path = result[0].data[0]

        assert emitted_path == f"simplecache::{test_files[0]}"
        with fsspec.open(emitted_path, mode="rt") as manifest:
            assert manifest.read() == "{}\n"

    @pytest.mark.parametrize(
        "file_paths",
        [
            "s3://bucket/manifests/*.jsonl",
            ["gs://bucket/a.jsonl", "https://example.test/b.jsonl"],
            "file:///tmp/manifests/*.jsonl",
            "zip://*.jsonl::s3://bucket/archive.zip",
        ],
    )
    def test_explicit_filesystem_uris_are_preserved(self, file_paths: str | list[str]):
        """URI inputs must not be rewritten as driver-local paths."""
        stage = FilePartitioningStage(file_paths=file_paths)

        assert stage.file_paths == file_paths
        assert stage._get_file_paths_for_discovery() == file_paths

    def test_process_with_files_per_partition(self, empty_task: EmptyTask, tmp_path: Path):
        """Test processing with files_per_partition setting."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=4, subdir="path")
        stage = FilePartitioningStage(file_paths=test_files, files_per_partition=2)

        result = stage.process(empty_task)

        assert len(result) == 2  # 4 files / 2 per partition
        assert result[0].data == test_files[:2]
        assert result[1].data == test_files[2:]

    def test_process_with_limit(self, empty_task: EmptyTask, tmp_path: Path):
        """Test processing with limit parameter - this is the main test for the limit functionality."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=10, subdir="path")
        stage = FilePartitioningStage(
            file_paths=test_files,
            files_per_partition=2,  # This would normally create 5 groups
            limit=3,  # But limit to only 3 groups
        )

        result = stage.process(empty_task)

        # Should only return 3 file groups due to limit
        assert len(result) == 3
        assert result[0].data == test_files[:2]
        assert result[1].data == test_files[2:4]
        assert result[2].data == test_files[4:6]

        # Verify metadata
        for i, task in enumerate(result):
            assert task._metadata["partition_index"] == i
            assert task._metadata["total_partitions"] == 5  # Total partitions before limit

    def test_process_with_limit_single_partition(self, empty_task: EmptyTask, tmp_path: Path):
        """Test limit when all files would be in a single partition."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=5, subdir="path")
        stage = FilePartitioningStage(
            file_paths=test_files,
            limit=1,  # Limit to 1 group, TODO: Ask ayush why this is the behavior
        )
        result = stage.process(empty_task)

        assert len(result) == 1
        assert result[0].data == [test_files[0]]

    def test_process_with_limit_zero(self, empty_task: EmptyTask, tmp_path: Path):
        """Test processing with limit set to 0."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=5, subdir="path")
        stage = FilePartitioningStage(
            file_paths=test_files,
            files_per_partition=1,
            limit=0,  # No groups should be created
        )

        result = stage.process(empty_task)

        assert len(result) == 0

    def test_process_with_blocksize(self, empty_task: EmptyTask, tmp_path: Path):
        """Test processing with blocksize setting."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=6)
        # Test files are 3 bytes each, so blocksize of 3B should create 6 partitions
        stage = FilePartitioningStage(file_paths=test_files, blocksize="3B")

        result = stage.process(empty_task)

        # With default avg_file_size of 100MB and blocksize of ~52MB,
        # files_per_block should be max(1, 52MB // 100MB) = 1
        assert len(result) == 6
        for i, task in enumerate(result):
            assert len(task.data) == 1
            assert task.data[0] == test_files[i]

    def test_large_blocksize_warning(self, caplog: pytest.LogCaptureFixture):
        """Test that a warning is raised if the blocksize is greater than 512 MB."""
        with caplog.at_level("WARNING"):
            FilePartitioningStage(file_paths="/test/path", blocksize="1GiB")
        assert "Blocksize is greater than 512 MB" in caplog.text

    def test_both_blocksize_and_files_per_partition_errors(self):
        """Test that specifying both blocksize and files_per_partition errors."""
        with pytest.raises(ValueError, match="only one is allowed"):
            FilePartitioningStage(
                file_paths="/test/path",
                files_per_partition=2,
                blocksize="128MB",
            )

    @pytest.mark.parametrize("input_kind", ["missing", "empty", "wrong_extension", "empty_list"])
    def test_process_fails_when_no_supported_files_are_discovered(
        self, input_kind: str, empty_task: EmptyTask, tmp_path: Path
    ):
        """Missing, empty, and unsupported inputs fail instead of silently succeeding."""
        input_path = tmp_path / input_kind
        if input_kind == "empty":
            input_path.mkdir()
            file_paths: str | list[str] = str(input_path)
        elif input_kind == "wrong_extension":
            input_path.mkdir()
            (input_path / "manifest.txt").write_text("{}\n")
            file_paths = str(input_path)
        elif input_kind == "empty_list":
            file_paths = []
        else:
            file_paths = str(input_path)

        stage = FilePartitioningStage(file_paths=file_paths)

        with pytest.raises(FileNotFoundError) as exc_info:
            stage.process(empty_task)

        message = str(exc_info.value)
        assert "No supported input files were found" in message
        assert repr(stage.file_paths) in message
        assert ".jsonl, .json, .parquet" in message
        assert "glob pattern" in message

    def test_empty_source_error_reports_anchored_and_worker_relative_attempts(
        self, empty_task: EmptyTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Fail-fast diagnostics include every path that discovery actually tried."""
        driver_dir = tmp_path / "driver"
        worker_dir = tmp_path / "worker"
        driver_dir.mkdir()
        worker_dir.mkdir()

        monkeypatch.chdir(driver_dir)
        stage = FilePartitioningStage(file_paths="missing/*.jsonl")

        monkeypatch.chdir(worker_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            stage.process(empty_task)

        message = str(exc_info.value)
        assert str(driver_dir / "missing" / "*.jsonl") in message
        assert "'missing/*.jsonl'" in message

    def test_process_allows_intentionally_empty_source(self, empty_task: EmptyTask, tmp_path: Path):
        """Callers can opt into the historical empty-source no-op behavior."""
        stage = FilePartitioningStage(file_paths=tmp_path / "missing", allow_empty=True)

        result = stage.process(empty_task)

        assert result == []

    def test_get_dataset_name(self, tmp_path: Path):
        """Test dataset name extraction."""
        stage = FilePartitioningStage(file_paths=[])

        # Test with files
        files = _create_test_jsonl_files(tmp_path, num_files=2, subdir="parent/dir")
        dataset_name = stage._get_dataset_name(files)
        assert dataset_name == "dir"

        # Test with empty files
        dataset_name = stage._get_dataset_name([])
        assert dataset_name == "dataset"

    def test_partition_by_count(self):
        """Test _partition_by_count method."""
        stage = FilePartitioningStage(file_paths=[])
        files = ["file1", "file2", "file3", "file4", "file5"]

        partitions = stage._partition_by_count(files, 2)

        assert len(partitions) == 3
        assert partitions[0] == ["file1", "file2"]
        assert partitions[1] == ["file3", "file4"]
        assert partitions[2] == ["file5"]

    def test_task_metadata(self, empty_task: EmptyTask, tmp_path: Path):
        """Test that created tasks have proper metadata."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=2, subdir="path")
        storage_options = {"option1": "value1"}
        stage = FilePartitioningStage(file_paths=test_files, storage_options=storage_options)

        result = stage.process(empty_task)

        assert len(result) == 2
        task = result[0]

        assert task._metadata["partition_index"] == 0
        assert task._metadata["total_partitions"] == 2
        assert task._metadata["source_files"] == [test_files[0]]
        assert task.reader_config == {}
