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
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

    from nemo_curator.backends.base import WorkerMetadata

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_EXT = {"txt": "txt", "json": "jsonl", "csv": "csv"}
_TIMELINE = "timeline.txt"
_JOURNAL = ".nemo_curator_group_export.jsonl"
_MISSING_IDENTITY = "<missing-group>"
_NAME_MAX_BYTES = 255


def _safe_name(value: Any) -> str:  # noqa: ANN401
    """A filename-safe group name (``speaker 1/A`` -> ``speaker_1_A``)."""
    name = _UNSAFE.sub("_", str(value)).strip("_")
    return name or "unknown"


def _group_name(
    value: Any,  # noqa: ANN401
    identity: str,
    *,
    canonical: bool,
    reserved: set[str] | None = None,
    max_bytes: int | None = None,
) -> str:
    """Return a name determined only by a group's value and typed identity."""
    raw = str(value)
    safe = _safe_name(value)
    if (
        canonical
        and raw == safe
        and safe not in (reserved or set())
        and (max_bytes is None or len(safe.encode("utf-8")) <= max_bytes)
    ):
        return safe
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    suffix = f"~{digest}"
    if max_bytes is not None:
        budget = max_bytes - len(suffix.encode("utf-8"))
        if budget < 1:
            msg = f"Filename byte budget {max_bytes} is too small for a stable group name"
            raise ValueError(msg)
        encoded = safe.encode("utf-8")[:budget]
        while True:
            try:
                safe = encoded.decode("utf-8")
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]
        safe = safe or "g"
    return f"{safe}{suffix}"


def _group_identity(value: Any, *, missing: bool = False) -> str:  # noqa: ANN401
    """Return a typed identity so equal string renderings remain distinct."""
    if missing:
        return _MISSING_IDENTITY
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        payload = repr(value)
    return f"{type_name}:{payload}"


def _jsonable(value: Any) -> bool:  # noqa: ANN401
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    try:
        # Match json.dump's streaming encoder and include the containing row's
        # nesting level; json.dumps may accept depths the streaming writer cannot.
        for _chunk in json.JSONEncoder().iterencode({"value": value}):
            pass
    except (TypeError, ValueError, RecursionError):
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
    start time across all groups. Driver-prepared runs persist timeline entries in
    the journal and publish the sorted timeline at finalization. Standalone calls
    snapshot every ``timeline_flush_rows`` rows, at batch boundaries, and at teardown.

    Explicit driver preparation retains committed export state across worker
    replacement and requires caller-managed finalization or abort. Replayed rows
    with a nonempty, unique ``(dataset_name, task_id)`` are exported only once.
    Anonymous rows retain append semantics. Ordinary
    ``Pipeline.run()`` does not manage these hooks: worker ``setup()`` starts
    fresh export state, so replacement workers can overwrite earlier output.

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
        timeline_flush_rows: Snapshot interval for standalone per-row use. Prepared
            pipelines publish the timeline once at finalization.
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
    # Per-group files aggregate many source rows. Retrying only unfinished
    # sources cannot safely reconstruct or append those shared files.
    is_resumable = False
    # output_dir is required, so discovery cannot instantiate this stage; declare the gates
    # instance-free too or a planner would see a disk writer with no disk gate.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        # Batching only adds a durability boundary; each row still supports
        # process(), so preserve the public per-row dispatch contract.
        dispatch="process",
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_dir"],
            lifecycle_side_effects=True,
            per_row_independent=False,
        ),
        error_policy="fail",
        description="Group manifest rows by a column and write one txt/json/csv file per group",
    )
    _written: set[str] = field(default_factory=set, init=False, repr=False)
    # Column set per csv path, fixed when that file's header is written. See :meth:`_append`.
    _csv_fields: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _csv_dropped: set[str] = field(default_factory=set, init=False, repr=False)
    _timeline: list[tuple[float, str]] = field(default_factory=list, init=False, repr=False)
    _pending: int = field(default=0, init=False, repr=False)
    _group_stems: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _group_labels: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _committed_tasks: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)
    _needs_restore: bool = field(default=False, init=False, repr=False)
    _journal: ManifestWriterStage | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if not self.output_dir:
            msg = "output_dir is required for ManifestGroupExportStage"
            raise ValueError(msg)
        if self.format not in _EXT:
            msg = f"format must be one of {sorted(_EXT)}, got {self.format!r}"
            raise ValueError(msg)

    # ------------------------------------------------------------------ lifecycle
    def prepare_on_driver(self, *, checkpoint_path: Path | None = None) -> None:
        """Reserve run state before workers are serialized; checkpoint resume is unsupported."""
        if checkpoint_path is not None:
            msg = "ManifestGroupExportStage does not support pipeline checkpoint resume"
            raise ValueError(msg)
        journal = ManifestWriterStage(os.path.join(self.output_dir, _JOURNAL))
        journal.prepare_on_driver()
        self._journal = journal

    def _verify_export_ownership(self) -> bool:
        prepared = self._journal is not None and getattr(self._journal, "_run_token", None) is not None
        if prepared:
            self._journal.setup_on_node()
        elif os.path.exists(os.path.join(self.output_dir, f"{_JOURNAL}._RUN")):
            msg = "ManifestGroupExportStage cannot replace output owned by an active pipeline run"
            raise RuntimeError(msg)
        return prepared

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        prepared = self._verify_export_ownership()
        os.makedirs(self.output_dir, exist_ok=True)
        # Group files are truncated on FIRST touch in this run (not up front), so a re-run
        # replaces its own output instead of appending to the previous run's -- and a group
        # that no longer occurs keeps its old file rather than being silently half-erased.
        self._written = set()
        self._csv_fields = {}
        self._csv_dropped = set()
        self._timeline = []
        self._pending = 0
        self._group_stems = {}
        self._group_labels = {}
        self._committed_tasks = set()
        self._needs_restore = False
        if prepared:
            self._restore_exports()
        logger.info(f"[{self.name}] exporting groups of {self.group_by!r} to {self.output_dir}")

    def _restore_exports(self) -> None:
        offsets = {}
        with open(self._journal.output_path, encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event.get("task") is not None:
                    self._committed_tasks.add(tuple(event["task"]))
                path = os.path.join(self.output_dir, event["filename"])
                self._written.add(path)
                offsets[path] = event["offset"]
                self._group_stems[event["identity"]] = event["stem"]
                if event["csv_fields"] is not None:
                    self._csv_fields[path] = event["csv_fields"]
                if event["csv_dropped"]:
                    self._csv_dropped.add(path)
                if event["timeline"] is not None:
                    self._timeline.append(tuple(event["timeline"]))
                    self._group_labels[event["identity"]] = event["label"]
        for path, offset in offsets.items():
            with open(path, "r+b") as output:
                if output.seek(0, os.SEEK_END) < offset:
                    msg = f"Committed group export is incomplete: {path}"
                    raise RuntimeError(msg)
                output.truncate(offset)

    def finalize(self) -> None:
        if self._journal is not None and getattr(self._journal, "_run_token", None) is not None:
            self.setup()
            if self.write_timeline:
                self._flush_timeline()
            os.remove(self._journal.output_path)
            self.abort_on_driver()

    def abort_on_driver(self) -> None:
        if self._journal is not None:
            self._journal.abort_on_driver()

    def reset_for_retry(self) -> None:
        self.abort_on_driver()

    def teardown(self) -> None:
        if self._journal is not None and getattr(self._journal, "_run_token", None) is not None:
            return
        if self.write_timeline:
            try:
                self._flush_timeline()
            except RuntimeError:
                # A retired worker must not rewrite a subsequent run's timeline.
                return

    def num_workers(self) -> int | None:
        return 1  # every group file is appended to from one place; keeps ordering sane

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        results = super().process_batch(tasks)
        prepared = self._journal is not None and getattr(self._journal, "_run_token", None) is not None
        if self.write_timeline and self._pending and not prepared:
            self._flush_timeline()
        return results

    # ------------------------------------------------------------------ contract
    def _required_data_keys(self) -> list[str]:
        required: list[str] = []
        if self.format == "txt" or self.write_timeline:
            required.append(self.text_key)
        return list(dict.fromkeys(required))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], self._required_data_keys()

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def describe(self) -> StageContract:
        return StageContract(
            dispatch="process",
            reads=IOSpec(data_keys=self._required_data_keys()),
            optional_reads=IOSpec(data_keys=list(dict.fromkeys([self.group_by, *(self.columns or [])]))),
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
        if self._needs_restore:
            self.setup()
        prepared = self._verify_export_ownership()
        task_identity = (task.dataset_name, task.task_id)
        if prepared and task.task_id and task_identity in self._committed_tasks:
            return task
        # A failed append must restore committed offsets even when the executor
        # retries on the same worker instead of constructing a replacement.
        self._needs_restore = prepared
        row = task.data if isinstance(task.data, dict) else {}
        # ``is None``, not truthiness: speaker ids are commonly zero-indexed, and
        # ``row.get(...) or ...`` filed every ``speaker_id == 0`` row under
        # ``missing_group``, mixing a real group in with the rows that genuinely had none.
        raw = row.get(self.group_by)
        missing = raw is None or raw == ""
        group_value = self.missing_group if missing else raw
        identity = _group_identity(group_value, missing=missing)
        reserved = {"timeline"} if self.format == "txt" and self.write_timeline else set()
        if self.format == "json":
            reserved.add(_JOURNAL.removesuffix(f".{_EXT[self.format]}"))
        if not missing:
            reserved.add(_safe_name(self.missing_group))
        stem = self._group_stems.setdefault(
            identity,
            _group_name(
                group_value,
                identity,
                canonical=missing or isinstance(group_value, str),
                reserved=reserved,
                max_bytes=_NAME_MAX_BYTES - len(f".{_EXT[self.format]}".encode()),
            ),
        )
        self._append(stem, row)
        label = None
        if self.write_timeline:
            label_reserved = {_safe_name(self.missing_group)} if not missing else set()
            label = self._group_labels.setdefault(
                identity,
                _group_name(
                    group_value,
                    identity,
                    canonical=missing or isinstance(group_value, str),
                    reserved=label_reserved,
                ),
            )
            self._record_timeline(label, row)
        if self._journal is not None and getattr(self._journal, "_run_token", None) is not None:
            filename = f"{stem}.{_EXT[self.format]}"
            path = os.path.join(self.output_dir, filename)
            event = {
                "task": task_identity if task.task_id else None,
                "identity": identity,
                "stem": stem,
                "filename": filename,
                "offset": os.path.getsize(path),
                "csv_fields": self._csv_fields.get(path),
                "csv_dropped": path in self._csv_dropped,
                "label": label,
                "timeline": self._timeline[-1] if self.write_timeline else None,
            }
            self._journal.write_jsonl_batch(json.dumps(event, ensure_ascii=False) + "\n")
            if task.task_id:
                self._committed_tasks.add(task_identity)
        self._needs_restore = False
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
                fields = self._csv_fields.setdefault(
                    path, list(self.columns) if self.columns is not None else list(selected)
                )
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
        prepared = self._journal is not None and getattr(self._journal, "_run_token", None) is not None
        if not prepared and self._pending >= max(1, self.timeline_flush_rows):
            self._flush_timeline()

    def _flush_timeline(self) -> None:
        if not self._timeline:
            return
        self._verify_export_ownership()
        path = os.path.join(self.output_dir, _TIMELINE)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{line}\n" for _start, line in sorted(self._timeline, key=lambda e: e[0]))
        self._pending = 0
