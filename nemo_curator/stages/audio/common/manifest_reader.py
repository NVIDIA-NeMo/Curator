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
import os
import time
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    Gates,
    IOSpec,
    Role,
    StageContract,
    StaticHints,
)
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask

AUDIO_PATH_RESOLUTION_BASENAME = "basename"
AUDIO_PATH_RESOLUTION_RELATIVE = "relative"
AUDIO_PATH_RESOLUTION_AS_IS = "as_is"
_AUDIO_PATH_RESOLUTION_MODES = (
    AUDIO_PATH_RESOLUTION_BASENAME,
    AUDIO_PATH_RESOLUTION_RELATIVE,
    AUDIO_PATH_RESOLUTION_AS_IS,
)


def _resolve_audio_path(audio_dir: str, value: str, mode: str = AUDIO_PATH_RESOLUTION_BASENAME) -> str:
    """Resolve a manifest row's audio path against ``audio_dir``.

    * ``"basename"`` (default): ``audio_dir / basename(value)``. Portable across hosts for flat
      staging dirs; ``ManifestReaderStage`` rejects duplicate basenames in this mode.
    * ``"relative"``: ``audio_dir / value`` (preserves subdirectories; an absolute value is kept,
      per ``os.path.join``).
    * ``"as_is"``: ``value`` unchanged.
    """
    if mode == AUDIO_PATH_RESOLUTION_BASENAME:
        return os.path.join(audio_dir, os.path.basename(value))
    if mode == AUDIO_PATH_RESOLUTION_RELATIVE:
        return os.path.join(audio_dir, value)
    if mode == AUDIO_PATH_RESOLUTION_AS_IS:
        return value
    msg = f"unknown audio_path_resolution {mode!r}; expected one of {_AUDIO_PATH_RESOLUTION_MODES}"
    raise ValueError(msg)


def _row_names_file(row: dict[str, Any], key: str, wanted: set[str]) -> bool:
    """Whether a manifest row's audio path is one of ``wanted`` (absolute-path comparison)."""
    value = row.get(key)
    return isinstance(value, str) and os.path.abspath(os.path.expanduser(value)) in wanted


@dataclass
class ManifestReaderStage(AgentReady, ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS). Every option below defaults to off, which emits
    every non-blank line verbatim.

    Args:
        include_files: Emit only rows whose audio path (under ``include_files_key``) is one of
            these files, comparing absolute paths. ``None`` -- the default -- reads every row.
            It restricts the same reader over the same manifest rather than pointing a delta
            run at a filtered copy, so the rows a partial run emits are the rows a full run
            would have emitted. The comparison uses the manifest's own value, before any
            ``audio_dir`` re-anchoring.
        include_files_key: Which row column holds the audio path (also the column
            ``audio_dir`` re-anchors).
        id_key: When set, every row must carry a non-empty value here that is unique across
            the manifests this stage instance reads; it is emitted as a string. Rows missing
            it, with an empty one, or repeating one already seen are skipped with a warning.
            Uniqueness is tracked per stage instance, so it holds across manifests only while
            one worker reads them (``num_workers`` is 1).
        segments_key: Row field holding the segment list. Declares that field as an output
            (present when the row carries it).
        audio_dir: When set, re-anchor each row's ``include_files_key`` path to this directory
            according to ``audio_path_resolution``, skip rows without a path, and stamp
            ``filepath_key`` on the emitted task.
        audio_path_resolution: ``"basename"`` (default; raises on two rows sharing a basename,
            since both would resolve to one file), ``"relative"`` (``audio_dir / path``), or
            ``"as_is"`` (the manifest value unchanged).
        dataset_name: Dataset tag for emitted tasks; ``None`` keeps the input task's tag.
        skip_invalid_rows: Skip, with a warning, lines that are not valid JSON objects instead
            of raising on invalid JSON.
    """

    name: str = "manifest_reader_stage"
    _: KW_ONLY
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    id_key: str | None = None
    segments_key: str | None = None
    audio_dir: str | None = None
    audio_path_resolution: str = AUDIO_PATH_RESOLUTION_BASENAME
    dataset_name: str | None = None
    skip_invalid_rows: bool = False
    # Declared statically as well as in describe(): a narrowable source has to be readable as
    # safe-to-narrow without constructing it, which is how the instance-free conformance sweep
    # sees it.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(lifecycle_side_effects=True, per_row_independent=True)
    )
    # It points at the column holding a row's audio path, which is an existing role rather
    # than a new one -- a filter comparing something else would not be filtering by file.
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"id_key"})

    def __post_init__(self) -> None:
        if self.audio_path_resolution not in _AUDIO_PATH_RESOLUTION_MODES:
            msg = (
                f"unknown audio_path_resolution {self.audio_path_resolution!r}; "
                f"expected one of {_AUDIO_PATH_RESOLUTION_MODES}"
            )
            raise ValueError(msg)
        self._seen_ids: set[str] = set()
        self._seen_basenames: dict[str, str] = {}

    def _accept(self, line: str, where: str) -> dict[str, Any] | None:  # noqa: C901, PLR0911
        """Parse one manifest line and apply the configured row checks; ``None`` skips it."""
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            if not self.skip_invalid_rows:
                raise
            logger.error(f"[{self.name}] {where}: invalid JSON ({e}); skipping")
            return None
        if self.skip_invalid_rows and not isinstance(row, dict):
            logger.warning(f"[{self.name}] {where}: JSON row is not an object; skipping")
            return None
        if self.id_key is not None:
            row_id = row.get(self.id_key)
            if row_id is None or (isinstance(row_id, str) and not row_id.strip()):
                logger.warning(f"[{self.name}] {where}: missing or empty {self.id_key!r}; skipping")
                return None
            # Emitted as a string: snippet ids, tar members and metrics records key on it.
            row[self.id_key] = row_id = str(row_id)
            if row_id in self._seen_ids:
                logger.warning(f"[{self.name}] {where}: duplicate id {row_id!r}; skipping")
                return None
            self._seen_ids.add(row_id)
        if self.audio_dir is None:
            return row
        path = row.get(self.include_files_key)
        if not path:
            logger.warning(f"[{self.name}] {where}: missing {self.include_files_key!r}; skipping")
            return None
        if self.audio_path_resolution == AUDIO_PATH_RESOLUTION_BASENAME:
            basename = os.path.basename(path)
            first = self._seen_basenames.setdefault(basename, where)
            if first != where:
                msg = (
                    f"[{self.name}] {where}: duplicate audio basename {basename!r} (first seen at {first}); "
                    f"two rows would resolve to the same audio under audio_path_resolution='basename'. "
                    f"Use 'relative' (preserves subdirs) or 'as_is' if this is intentional."
                )
                raise ValueError(msg)
        return row

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        wanted = (
            None
            if self.include_files is None
            else {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
        )
        for manifest in paths:
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = self._accept(line.strip(), f"{manifest}:{lineno}")
                    if row is None:
                        continue
                    if wanted is not None and not _row_names_file(row, self.include_files_key, wanted):
                        continue
                    if self.audio_dir is not None:
                        row[self.include_files_key] = _resolve_audio_path(
                            self.audio_dir, row[self.include_files_key], self.audio_path_resolution
                        )
                    results.append(
                        AudioTask(
                            dataset_name=self.dataset_name or task.dataset_name,
                            data=row,
                            filepath_key=None if self.audio_dir is None else self.include_files_key,
                            _metadata=task._metadata,
                            _stage_perf=list(task._stage_perf),
                        )
                    )
                    count += 1
            logger.info(f"ManifestReaderStage: loaded {count} entries from {manifest}")
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": len(paths),
                "entries_read": len(results),
            }
        )
        return results

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        # The row is emitted verbatim, so the path column this reader was pointed at IS the
        # column downstream sees. Hard-coding ``audio_filepath`` claimed a key the rows do not
        # carry whenever ``include_files_key`` is configured: a manifest of ``recording_path``
        # validated clean against a default consumer, which then found no path at runtime.
        writes = [self.include_files_key]
        if self.id_key:
            writes.append(self.id_key)
        conditional = []
        if self.segments_key:
            conditional.append(
                ConditionalWrite(
                    writes=IOSpec(data_keys=[self.segments_key]),
                    condition=f"the manifest row contains '{self.segments_key}'",
                    value_origin="upstream_same_key",
                )
            )
        return StageContract(
            writes=IOSpec(data_keys=writes),
            conditional_writes=conditional,
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=_per_row_independent(self)),
        )


def _per_row_independent(reader: Any) -> bool:  # noqa: ANN401 - either reader dataclass
    # Id de-duplication and the basename-collision guard look across rows, so a run over a
    # subset of the manifest can emit rows (or succeed) where the full run would not.
    return reader.id_key is None and (
        reader.audio_dir is None or reader.audio_path_resolution != AUDIO_PATH_RESOLUTION_BASENAME
    )


@dataclass
class ManifestReader(AgentReady, CompositeStage[EmptyTask, AudioTask]):
    """Composite stage for reading JSONL manifests.

    Decomposes into:
    1. FilePartitioningStage — discovers and partitions manifest files
    2. ManifestReaderStage — reads each partition line-by-line (no Pandas)

    A manifest path that names no file yields a "No files found" warning and no rows; callers
    that need a missing manifest to fail check it before building the pipeline.

    Args:
        manifest_path: Path or list of paths to JSONL manifests (local or cloud).
        files_per_partition: Number of manifest files per partition. Defaults to 1.
        blocksize: Target size per partition (e.g., "100MB"). Ignored if files_per_partition is set.
        file_extensions: File extensions to filter. Defaults to [".jsonl", ".json"].
        storage_options: Storage options for cloud paths (S3, GCS credentials, endpoints).
        include_files: Read only the rows naming these audio files (see ``ManifestReaderStage``).
        include_files_key: Which row column holds the audio path.
        id_key: Require a non-empty, unique, stringified id per row (see ``ManifestReaderStage``).
        segments_key: Row field holding the segment list (see ``ManifestReaderStage``).
        audio_dir: Re-anchor each row's audio path to this directory.
        audio_path_resolution: ``"basename"`` (default), ``"relative"``, or ``"as_is"``.
        dataset_name: Dataset tag for emitted tasks; ``None`` keeps the partitioner's tag.
        skip_invalid_rows: Skip lines that are not valid JSON objects instead of raising.
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader"
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None
    _: KW_ONLY
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    id_key: str | None = None
    segments_key: str | None = None
    audio_dir: str | None = None
    audio_path_resolution: str = AUDIO_PATH_RESOLUTION_BASENAME
    dataset_name: str | None = None
    skip_invalid_rows: bool = False
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=True))
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"id_key"})

    def __post_init__(self) -> None:
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            ManifestReaderStage(
                include_files=self.include_files,
                include_files_key=self.include_files_key,
                id_key=self.id_key,
                segments_key=self.segments_key,
                audio_dir=self.audio_dir,
                audio_path_resolution=self.audio_path_resolution,
                dataset_name=self.dataset_name,
                skip_invalid_rows=self.skip_invalid_rows,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(per_row_independent=_per_row_independent(self)),
        )
