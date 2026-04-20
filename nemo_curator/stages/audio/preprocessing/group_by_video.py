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

"""Group manifest entries by video ID.

Reads a stream of ``DocumentBatch`` rows and annotates each row with a
resolved ``video_id`` field.  Video ID extraction follows a priority chain:

1. Explicit ``id`` or ``youtube_id`` manifest field.
2. ``audio_item_id`` field (common in Granary manifests).
3. Regex from ``audio_filepath`` (e.g. ``_converted_ru103_eZ0a2mvLOQE_...``).
   Enabled via ``parse_filepath=True`` (default). Handles cases where upstream
   stages dropped the original ID fields.
4. Fallback: ``"unknown"``.

When used upstream of ``SpeakerClusteringStage``, the ``video_id`` column
enables per-video clustering instead of global clustering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, DocumentBatch

# YouTube video IDs: 11 chars from [A-Za-z0-9_-]
_YT_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
# Pattern in converted NeMo filenames: ..._converted_<corpus>_<VIDEOID>_...
_CONVERTED_RE = re.compile(r"_converted_\w+_([A-Za-z0-9_-]{11})_")


def extract_video_id(
    row: dict,
    id_key: str = "video_id",
    parse_filepath: bool = True,
    filepath_key: str = "audio_filepath",
) -> str:
    """Extract video ID from a manifest row.

    Priority:
    1. Existing ``id_key`` field if already set and non-empty.
    2. ``id`` field (common in harvest manifests).
    3. ``youtube_id`` field.
    4. ``audio_item_id`` field (Granary manifests).
    5. Regex from ``audio_filepath`` (if ``parse_filepath=True``).
    6. ``"unknown"``.
    """
    # 1. Already resolved
    existing = row.get(id_key)
    if existing and str(existing) not in ("", "None", "nan"):
        return str(existing)

    # 2-4. Explicit fields
    for key in ("id", "youtube_id", "audio_item_id"):
        val = row.get(key)
        if val and str(val) not in ("", "None", "nan"):
            # For composite IDs like "uuid__VIDEOID", extract the video part
            s = str(val)
            if "__" in s:
                s = s.rsplit("__", 1)[-1]
            if _YT_ID_RE.fullmatch(s):
                return s
            return str(val)

    # 5. Parse from audio_filepath
    if parse_filepath:
        afp = row.get(filepath_key, "")
        if afp:
            m = _CONVERTED_RE.search(afp)
            if m:
                return m.group(1)

    return "unknown"


@dataclass
class GroupByVideoStage(ProcessingStage[DocumentBatch | AudioTask, DocumentBatch | AudioTask]):
    """Annotate each row with a resolved ``video_id`` field.

    This is a lightweight CPU stage that adds a ``video_id`` column to every
    manifest entry.  It does **not** restructure the data — downstream stages
    (e.g. clustering) can use the ``video_id`` column to group entries.

    For the Ray pipeline path (``DocumentBatch``), all rows in the batch are
    annotated in bulk.  For the ``AudioTask`` path, a single row is annotated.

    Args:
        video_id_key: Output column name for the resolved video ID.
        name: Stage identifier.
        resources: Compute resources (CPU-only).
    """

    video_id_key: str = "video_id"
    parse_filepath: bool = True
    filepath_key: str = "audio_filepath"
    name: str = "GroupByVideoStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.video_id_key]

    def process(self, task: DocumentBatch | AudioTask) -> DocumentBatch | AudioTask:
        if isinstance(task, DocumentBatch):
            entries = task.data.to_dict("records")
            n_unknown = 0
            for entry in entries:
                vid = extract_video_id(
                    entry, self.video_id_key, self.parse_filepath, self.filepath_key,
                )
                entry[self.video_id_key] = vid
                if vid == "unknown":
                    n_unknown += 1
            if n_unknown > 0:
                logger.warning(
                    f"[{self.name}] {n_unknown}/{len(entries)} entries have video_id='unknown'"
                )
            return DocumentBatch(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                data=pd.DataFrame(entries),
                _stage_perf=task._stage_perf,
            )

        elif isinstance(task, AudioTask):
            vid = extract_video_id(
                dict(task.data), self.video_id_key, self.parse_filepath, self.filepath_key,
            )
            task.data[self.video_id_key] = vid
            return task

        raise TypeError(f"Unsupported task type: {type(task)}")
