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
from dataclasses import dataclass, field
from typing import Any

from fsspec import available_protocols
from fsspec.core import split_protocol, url_to_fs
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import (
    _split_files_as_per_blocksize,
    get_all_file_paths_and_size_under,
    infer_dataset_name_from_path,
    parse_bytes_string_to_int,
)


@dataclass
class FilePartitioningStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Stage that partitions input file paths into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.

    Parameters
    ----------
    file_paths: str | os.PathLike[str] | list[str | os.PathLike[str]]
        Path, glob, or list of paths/globs for the input files. Local paths are
        first resolved relative to the working directory where the stage is
        constructed. If that candidate contains no supported files, discovery
        retries the original worker-relative path. When the stage itself has a
        runtime ``working_dir``, relative paths resolve there directly. Remote
        URI components are preserved.
    files_per_partition: int | None = None
        Number of files per partition.
        If both files_per_partition and blocksize are not provided,
        then default to files_per_partition = 1 and enforce a blocksize <= 512 MB per partition safeguard.
        Errors if both files_per_partition and blocksize are provided.
    blocksize: int | str | None = None
        Target size of the partitions. A blocksize of 512 MB or less is recommended.
        Errors if both files_per_partition and blocksize are provided.
        Note: For compressed files, the compressed size is used for blocksize estimation.
    file_extensions: list[str] | None = None
        File extensions to filter.
    storage_options: dict[str, Any] | None = None
        Storage options to pass to the file system.
    limit: int | None = None
        Maximum number of partitions to create.
    allow_empty: bool = False
        If True, return no tasks when no supported files are discovered. By
        default an empty source raises ``FileNotFoundError``.

    Raises
    ------
    FileNotFoundError
        If no files with the configured extensions are discovered and
        ``allow_empty`` is False.
    """

    file_paths: str | os.PathLike[str] | list[str | os.PathLike[str]]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    limit: int | None = None
    name: str = "file_partitioning"
    allow_empty: bool = False

    _construction_cwd: str = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize default values."""
        if self.files_per_partition is not None and self.blocksize is not None:
            msg = "Both 'files_per_partition' and 'blocksize' were specified, but only one is allowed"
            raise ValueError(msg)
        if self.file_extensions is None:
            self.file_extensions = [".jsonl", ".json", ".parquet"]
        if self.storage_options is None:
            self.storage_options = {}

        self.file_paths = self._coerce_file_paths(self.file_paths)
        self._construction_cwd = os.getcwd()

        # self.blocksize is the value set by the user
        # self._blocksize is the value used internally
        if self.blocksize is not None:
            self._blocksize = parse_bytes_string_to_int(self.blocksize)
        else:
            self._blocksize = parse_bytes_string_to_int("512MB")

        if self._blocksize > parse_bytes_string_to_int("512MB"):
            msg = (
                f"Blocksize is greater than 512 MB, which is not recommended: {self.blocksize} "
                "Consider using a smaller blocksize to avoid potential memory issues."
            )
            logger.warning(msg)

        self.resources = Resources(cpus=0.5)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def process(self, _: EmptyTask) -> list[FileGroupTask]:
        """Process the initial task to create file group tasks.

        This stage expects a simple Task with file paths information
        and outputs multiple FileGroupTasks for parallel processing.
        """
        sort_by_size = self.blocksize is not None
        files_with_sizes, attempted_paths = self._discover_file_list_with_sizes(sort_by_size)
        # Extract list[str] from list[tuple[str, int]]
        files = [file[0] for file in files_with_sizes]

        logger.info(f"Found {len(files)} files")
        if len(files) == 0:
            return self._handle_empty_source(attempted_paths)

        # Partition files
        if self.files_per_partition:
            partitions = self._partition_by_count(files, self.files_per_partition)
        elif self.blocksize:
            partitions = self._partition_by_size(files_with_sizes, self._blocksize)
        else:
            # Default to one file per partition
            logger.info("No partitions specified, defaulting to one file per partition")
            partitions = self._partition_by_count(files, 1)

        # Build a dictionary of path: size of all files
        path_to_size: dict[str, int] = dict(files_with_sizes)

        # Check that no files have size less than 0 (since -1 is used to indicate unknown size)
        if any(size < 0 for size in path_to_size.values()):
            msg = "Skipping storage limit check because some files have unknown size"
            logger.warning(msg)
        else:
            # Verify storage size of input files is not greater than self._blocksize (512 MB by default)
            # This should be a very quick check per file, so we do it first before reading the data
            for partition in partitions:
                total_storage_size = sum(path_to_size[path] for path in partition)
                # Scenario 1: The user specified blocksize and the partition created is too large
                # This means at least one file is larger than the blocksize
                if self.blocksize is not None and total_storage_size > self._blocksize:
                    msg = (
                        f"File group task has exceeded the storage limit per partition: {partition}. "
                        f"Total storage size is {total_storage_size} bytes (limit {self._blocksize} bytes). "
                        "Please increase blocksize if possible (the maximum recommended blocksize is 512 MB). "
                        "Any individual file(s) larger than the storage limit should be split into smaller chunks using nemo_curator.utils.split_large_files."
                    )
                    logger.warning(msg)
                # Scenario 2: The user did not specify blocksize and the partition created is too large
                elif total_storage_size > self._blocksize:
                    msg = (
                        f"File group task has exceeded the storage limit per partition: {partition}. "
                        f"Total storage size is {total_storage_size} bytes (limit {self._blocksize} bytes). "
                        "Please reduce files_per_partition if possible, or set blocksize instead (the maximum recommended blocksize is 512 MB). "
                        "Any individual file(s) larger than the storage limit should be split into smaller chunks using nemo_curator.utils.split_large_files."
                    )
                    logger.warning(msg)

        # Create FileGroupTask for each partition
        tasks = []
        dataset_name = self._get_dataset_name(files)

        for i, file_group in enumerate(partitions):
            if self.limit is not None and len(tasks) >= self.limit:
                # We should revisit this behavior.
                # https://github.com/NVIDIA-NeMo/Curator/issues/948
                logger.info(f"Reached limit of {self.limit} file groups")
                break
            file_task = FileGroupTask(
                dataset_name=dataset_name,
                data=file_group,
                _metadata={
                    "partition_index": i,
                    "total_partitions": len(partitions),
                    "source_files": file_group,  # Add source files for deterministic naming during write stage
                },
                reader_config={},  # Empty - will be populated by reader stage
            )
            tasks.append(file_task)

        logger.info(f"Created {len(tasks)} file groups from {len(files)} files")
        return tasks

    def _handle_empty_source(self, attempted_paths: list[str]) -> list[FileGroupTask]:
        if self.allow_empty:
            logger.warning(f"No files found after trying {attempted_paths}")
            return []
        expected_extensions = ", ".join(self.file_extensions)
        msg = (
            f"No supported input files were found for configured path(s) {self.file_paths!r}, "
            f"after trying these discovery path(s) in order: {attempted_paths!r}. "
            f"Expected file extensions: {expected_extensions}. "
            "Check that the path exists, any glob pattern matches files, and the execution environment "
            "can access the input."
        )
        raise FileNotFoundError(msg)

    @staticmethod
    def _coerce_path(path: str | os.PathLike[str]) -> str:
        if not isinstance(path, (str, os.PathLike)):
            msg = f"Invalid file path: {path!r}, must be a string or path-like object"
            raise TypeError(msg)
        coerced_path = os.fspath(path)
        if not isinstance(coerced_path, str):
            msg = f"Invalid file path: {path!r}, must resolve to a string"
            raise TypeError(msg)
        return coerced_path

    @classmethod
    def _coerce_file_paths(
        cls,
        file_paths: str | os.PathLike[str] | list[str | os.PathLike[str]],
    ) -> str | list[str]:
        if isinstance(file_paths, (str, os.PathLike)):
            return cls._coerce_path(file_paths)
        if isinstance(file_paths, list):
            return [cls._coerce_path(path) for path in file_paths]
        msg = f"Invalid file paths: {file_paths!r}, must be a string, path-like object, or list of them"
        raise TypeError(msg)

    @staticmethod
    def _normalize_local_path(path: str, *, base_dir: str, preserve_relative: bool) -> str:
        expanded_path = os.path.expanduser(path)
        if os.path.isabs(expanded_path):
            return expanded_path
        if preserve_relative:
            return path
        return os.path.abspath(os.path.join(base_dir, expanded_path))

    @classmethod
    def _normalize_input_path(cls, path: str, *, base_dir: str, preserve_relative: bool) -> str:
        """Normalize local parts of an fsspec path while retaining its protocol chain."""
        components = path.split("::")
        is_chain = len(components) > 1
        known_protocols = set(available_protocols()) if is_chain else set()
        normalized_components = []

        for component in components:
            protocol, protocol_path = split_protocol(component)
            if protocol in {"file", "local"}:
                normalized_path = cls._normalize_local_path(
                    protocol_path,
                    base_dir=base_dir,
                    preserve_relative=preserve_relative,
                )
                normalized_components.append(f"{protocol}://{normalized_path}")
            elif protocol is not None:
                normalized_components.append(component)
            elif component.startswith(("file:", "local:")):
                protocol, protocol_path = component.split(":", 1)
                normalized_path = cls._normalize_local_path(
                    protocol_path,
                    base_dir=base_dir,
                    preserve_relative=preserve_relative,
                )
                normalized_components.append(f"{protocol}://{normalized_path}")
            elif is_chain and component in known_protocols:
                normalized_components.append(component)
            else:
                normalized_components.append(
                    cls._normalize_local_path(
                        component,
                        base_dir=base_dir,
                        preserve_relative=preserve_relative,
                    )
                )

        return "::".join(normalized_components)

    def _get_file_paths_for_discovery(self) -> str | list[str]:
        if isinstance(self.file_paths, str):
            return self._get_discovery_path_candidates(self.file_paths)[0]
        return [self._get_discovery_path_candidates(path)[0] for path in self.file_paths]

    def _get_discovery_path_candidates(self, path: str) -> list[str]:
        """Return ordered, unique discovery candidates for one configured path."""
        runtime_working_dir = self.runtime_env.get("working_dir") if self.runtime_env else None
        if runtime_working_dir is not None:
            return [
                self._normalize_input_path(
                    path,
                    base_dir=self._construction_cwd,
                    preserve_relative=True,
                )
            ]

        anchored_path = self._normalize_input_path(
            path,
            base_dir=self._construction_cwd,
            preserve_relative=False,
        )
        worker_relative_path = self._normalize_input_path(
            path,
            base_dir=self._construction_cwd,
            preserve_relative=True,
        )
        return list(dict.fromkeys((anchored_path, worker_relative_path)))

    @staticmethod
    def _replace_chained_path_pattern(path: str, discovered_path: str) -> str:
        """Replace the outer filesystem's path pattern while retaining its chain targets."""
        components = path.split("::")
        known_protocols = set(available_protocols())

        for index, component in enumerate(components):
            protocol, protocol_path = split_protocol(component)
            if protocol is not None:
                if protocol_path:
                    components[index] = f"{protocol}://{discovered_path}"
                    return "::".join(components)
                continue
            if component.startswith(("file:", "local:")):
                protocol, protocol_path = component.split(":", 1)
                if protocol_path:
                    components[index] = f"{protocol}://{discovered_path}"
                    return "::".join(components)
                continue
            if component in known_protocols:
                continue
            components[index] = discovered_path
            return "::".join(components)

        msg = f"Could not locate a replaceable path component in chained URL {path!r}"
        raise ValueError(msg)

    def _get_file_records_for_path(
        self,
        path: str,
        *,
        recurse_subdirectories: bool,
        sort_by_size: bool,
    ) -> list[tuple[str, int]]:
        if "::" not in path:
            return get_all_file_paths_and_size_under(
                path,
                recurse_subdirectories=recurse_subdirectories,
                keep_extensions=self.file_extensions,
                storage_options=self.storage_options,
                sort_by_size=sort_by_size,
            )

        fs, fs_path = url_to_fs(path, **self.storage_options)
        records = get_all_file_paths_and_size_under(
            fs_path,
            recurse_subdirectories=recurse_subdirectories,
            keep_extensions=self.file_extensions,
            fs=fs,
            sort_by_size=sort_by_size,
        )
        return [(self._replace_chained_path_pattern(path, record_path), size) for record_path, size in records]

    def _discover_file_list_with_sizes(self, sort_by_size: bool) -> tuple[list[tuple[str, int]], list[str]]:
        """Discover each configured path, falling back to worker-relative paths independently."""
        configured_paths = [self.file_paths] if isinstance(self.file_paths, str) else self.file_paths
        recurse_subdirectories = isinstance(self.file_paths, str)
        attempted_paths: list[str] = []
        records_by_path: dict[str, int] = {}
        candidate_records_by_path: dict[str, list[tuple[str, int]]] = {}

        for configured_path in configured_paths:
            candidates = self._get_discovery_path_candidates(configured_path)
            for candidate_index, candidate in enumerate(candidates):
                if candidate in candidate_records_by_path:
                    candidate_records = candidate_records_by_path[candidate]
                else:
                    attempted_paths.append(candidate)
                    try:
                        candidate_records = self._get_file_records_for_path(
                            candidate,
                            recurse_subdirectories=recurse_subdirectories,
                            sort_by_size=sort_by_size,
                        )
                    except FileNotFoundError:
                        candidate_records = []
                    candidate_records_by_path[candidate] = candidate_records

                if candidate_records:
                    for record_path, size in candidate_records:
                        records_by_path.setdefault(record_path, size)
                    break

                if candidate_index + 1 < len(candidates):
                    logger.debug(
                        f"No supported files found for {candidate!r}; "
                        f"retrying worker-relative candidate {candidates[candidate_index + 1]!r}"
                    )

        records = list(records_by_path.items())
        return sorted(records, key=lambda x: x[1] if sort_by_size else x[0]), attempted_paths

    def _get_file_list_with_sizes(
        self,
        sort_by_size: bool = True,
        file_paths: str | list[str] | None = None,
    ) -> list[tuple[str, int]]:
        """
        Get the list of files to process.
        """
        if file_paths is None:
            records, _ = self._discover_file_list_with_sizes(sort_by_size)
            return records
        logger.debug(f"Getting file list with sizes for {file_paths}")
        if isinstance(file_paths, str):
            # Directory: list contents (recursively) and filter extensions
            output_ls = self._get_file_records_for_path(
                file_paths,
                recurse_subdirectories=True,
                sort_by_size=sort_by_size,
            )
        elif isinstance(file_paths, list):
            output_ls = []
            for path in file_paths:
                output_ls.extend(
                    self._get_file_records_for_path(
                        path,
                        recurse_subdirectories=False,
                        sort_by_size=sort_by_size,
                    )
                )
        else:
            msg = f"Invalid file paths: {file_paths}, must be a string or list of strings"
            raise TypeError(msg)
        return sorted(output_ls, key=lambda x: x[1] if sort_by_size else x[0])

    def _get_dataset_name(self, files: list[str]) -> str:
        """Extract dataset name from file paths (fsspec-compatible)."""
        if not files:
            return "dataset"

        return infer_dataset_name_from_path(files[0])

    def _partition_by_count(self, files: list[str], count: int) -> list[list[str]]:
        """Partition files by count."""
        partitions = []
        for i in range(0, len(files), count):
            partitions.append(files[i : i + count])
        return partitions

    def _partition_by_size(self, files: list[tuple[str, int]], blocksize: int | str) -> list[list[str]]:
        """Partition files by target size.
        Args:
            files: A list of tuples (file_path, file_size)
            blocksize: The target size of the partitions
        Returns:
            A list of lists, where each inner list contains the file paths of the files in the partitionN
        """
        sorted_files = sorted(files, key=lambda x: x[1])
        return _split_files_as_per_blocksize(sorted_files, blocksize)
