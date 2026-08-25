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

"""Group manifest rows by a column and export one file per group.

The last mile of a curation run is almost always "give me the result *arranged* this way" --
per-speaker transcripts, per-language splits, per-source shards. Without a stage for it that
turns into hand-written Python, which is exactly the thing the agent must never do.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_EXT = {"txt": "txt", "json": "jsonl", "csv": "csv"}
_TIMELINE = "timeline.txt"


def _safe_name(value: Any) -> str:  # noqa: ANN401
    """A filename-safe group name (``speaker 1/A`` -> ``speaker_1_A``)."""
    name = _UNSAFE.sub("_", str(value)).strip("_")
    return name or "unknown"


def _jsonable(value: Any) -> bool:  # noqa: ANN401
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass
class ManifestGroupExportStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Write one file per distinct value of ``group_by`` (e.g. a transcript per speaker).

    Each row is appended to ``<output_dir>/<group>.<ext>`` in the chosen ``format``:

    * ``txt``  -- one line per row, ``text_key``'s value, optionally timestamp-prefixed
    * ``json`` -- JSONL of the selected (serializable) columns
    * ``csv``  -- header + the selected columns

    With ``write_timeline`` it also writes a single who-spoke-when ``timeline.txt`` ordered by
    start time across all groups. The timeline is rewritten every ``timeline_flush_rows`` rows
    and finalized in ``teardown``, so it is always a correct ordering of what has been flushed.

    Rows pass through unchanged, so this can sit anywhere after the data exists -- it is a
    tee, not a sink. Non-serializable values (e.g. a resident ``waveform`` tensor) are dropped
    from the written columns rather than crashing the export.

    Args:
        output_dir: Directory to write the per-group files into.
        group_by: Column whose value names each group (e.g. ``speaker_id``, ``lang``).
        format: ``txt`` (default), ``json`` or ``csv``.
        columns: Columns to write for ``json``/``csv`` (default: every serializable column).
        text_key: Column used as the line body in ``txt`` format.
        include_timestamps: Prefix ``txt`` lines with ``[start - end]`` when both are present.
        write_timeline: Also write a combined, time-ordered ``timeline.txt``.
        start_key / end_key: Segment timing columns used for ordering and timestamps.
        missing_group: Group name used for rows with no ``group_by`` value.
    """

    output_dir: str
    group_by: str = "speaker_id"
    format: Literal["txt", "json", "csv"] = "txt"
    columns: list[str] | None = None
    text_key: str = "text"
    include_timestamps: bool = True
    write_timeline: bool = False
    start_key: str = "start"
    end_key: str = "end"
    missing_group: str = "unknown"
    timeline_flush_rows: int = 100
    name: str = "manifest_group_export"
    # output_dir is required, so discovery cannot instantiate this stage; declare the gates
    # instance-free too or a planner would see a disk writer with no disk gate.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_dir"],
            lifecycle_side_effects=True,
            per_row_independent=False,
        ),
        error_policy="annotate",
        description="Group manifest rows by a column and write one txt/json/csv file per group",
    )
    _written: set[str] = field(default_factory=set, init=False, repr=False)
    # Column set per csv path, fixed when that file's header is written. See :meth:`_append`.
    _csv_fields: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _csv_dropped: set[str] = field(default_factory=set, init=False, repr=False)
    _timeline: list[tuple[float, str]] = field(default_factory=list, init=False, repr=False)
    _pending: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if not self.output_dir:
            msg = "output_dir is required for ManifestGroupExportStage"
            raise ValueError(msg)
        if self.format not in _EXT:
            msg = f"format must be one of {sorted(_EXT)}, got {self.format!r}"
            raise ValueError(msg)

    # ------------------------------------------------------------------ lifecycle
    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        # Group files are truncated on FIRST touch in this run (not up front), so a re-run
        # replaces its own output instead of appending to the previous run's -- and a group
        # that no longer occurs keeps its old file rather than being silently half-erased.
        self._written = set()
        self._timeline = []
        self._pending = 0
        logger.info(f"[{self.name}] exporting groups of {self.group_by!r} to {self.output_dir}")

    def teardown(self) -> None:
        if self.write_timeline:
            self._flush_timeline()

    def num_workers(self) -> int | None:
        return 1  # every group file is appended to from one place; keeps ordering sane

    # ------------------------------------------------------------------ contract
    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.group_by]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.group_by]),
            writes=IOSpec(produces=["disk"]),
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_dir"],
                lifecycle_side_effects=True,
                # Each output file is every row sharing a group value, so which rows are present
                # decides the file's contents.
                per_row_independent=False,
            ),
            description="Group manifest rows by a column and write one txt/json/csv file per group",
        )

    # ------------------------------------------------------------------ processing
    def process(self, task: AudioTask) -> AudioTask:
        row = task.data if isinstance(task.data, dict) else {}
        # ``is None``, not truthiness: speaker ids are commonly zero-indexed, and
        # ``row.get(...) or ...`` filed every ``speaker_id == 0`` row under
        # ``missing_group``, mixing a real group in with the rows that genuinely had none.
        raw = row.get(self.group_by)
        group = _safe_name(self.missing_group if raw is None or raw == "" else raw)
        self._append(group, row)
        if self.write_timeline:
            self._record_timeline(group, row)
        return task

    def _append(self, group: str, row: dict[str, Any]) -> None:
        path = os.path.join(self.output_dir, f"{group}.{_EXT[self.format]}")
        first = path not in self._written
        self._written.add(path)
        with open(path, "w" if first else "a", encoding="utf-8", newline="") as f:
            if self.format == "txt":
                f.write(self._text_line(row) + "\n")
            elif self.format == "json":
                json.dump(self._selected(row), f, ensure_ascii=False)
                f.write("\n")
            else:
                selected = self._selected(row)
                # A csv header is written once, so the column set has to be fixed for the life
                # of the file. Deriving fieldnames from each row instead lets a later row with
                # a different shape -- a missing transcript, a column dropped for being
                # non-serializable, a different insertion order -- be written under a header it
                # does not match, silently shifting values into the wrong columns.
                fields = self._csv_fields.setdefault(path, list(selected))
                writer = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
                if first:
                    writer.writeheader()
                # ``extrasaction="ignore"`` keeps the file well formed when a later row carries
                # a column the header does not have -- the alternative is rewriting the whole
                # file -- but dropping a value silently is worse than a slow export, so say so
                # once per file. Pass ``columns`` to pin the schema and avoid this entirely.
                unheaded = [k for k in selected if k not in fields]
                if unheaded and path not in self._csv_dropped:
                    self._csv_dropped.add(path)
                    logger.warning(
                        f"[{self.name}] {os.path.basename(path)}: dropping column(s) "
                        f"{sorted(unheaded)} absent from the header written for the first row. "
                        f"Set columns=[...] to pin the schema."
                    )
                writer.writerow(selected)

    def _selected(self, row: dict[str, Any]) -> dict[str, Any]:
        keys = self.columns if self.columns is not None else list(row)
        return {k: row[k] for k in keys if k in row and _jsonable(row[k])}

    def _text_line(self, row: dict[str, Any]) -> str:
        body = str(row.get(self.text_key, "") or "").strip()
        span = self._span(row)
        return f"[{span[0]:.2f} - {span[1]:.2f}] {body}" if (self.include_timestamps and span) else body

    def _span(self, row: dict[str, Any]) -> tuple[float, float] | None:
        try:
            return float(row[self.start_key]), float(row[self.end_key])
        except (KeyError, TypeError, ValueError):
            return None

    def _record_timeline(self, group: str, row: dict[str, Any]) -> None:
        span = self._span(row)
        start = span[0] if span else float(len(self._timeline))
        body = str(row.get(self.text_key, "") or "").strip()
        stamp = f"[{span[0]:.2f} - {span[1]:.2f}] " if span else ""
        self._timeline.append((start, f"{stamp}{group}: {body}"))
        self._pending += 1
        if self._pending >= max(1, self.timeline_flush_rows):
            self._flush_timeline()

    def _flush_timeline(self) -> None:
        if not self._timeline:
            return
        path = os.path.join(self.output_dir, _TIMELINE)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{line}\n" for _start, line in sorted(self._timeline, key=lambda e: e[0]))
        self._pending = 0
