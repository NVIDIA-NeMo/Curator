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

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask


class _AudioFolderTask(AudioTask):
    """Folder source identity independent of enumeration and the selected cohort."""

    def get_deterministic_id(self) -> str:
        return self._metadata["audio_folder_source_id"]


@dataclass
class CreateInitialManifestAudioFolderStage(AgentReady, ProcessingStage[EmptyTask, AudioTask]):
    """Create an initial manifest from any local folder of audio files.

    Recursively scans ``data_dir`` for audio files and emits one AudioTask per file with its
    path under ``audio_filepath_key`` (plus a filename-derived ``audio_item_id``). A generic,
    dataset-agnostic source: no download, no transcripts, and no dataset-specific filename
    parsing -- unlike ``CreateInitialManifest{ReadSpeech,Fleurs}Stage``. Use it to start a
    pipeline from a plain folder of WAV/FLAC/MP3/... when there is no JSONL manifest (use
    ``ManifestReader`` when a manifest already exists).

    Args:
        data_dir: Local folder to scan for audio files.
        extensions: Audio file extensions to include (case-insensitive).
        recursive: Recurse into subfolders (default True).
        max_samples: Maximum number of files to include (-1 for all).
        include_files: Process only these files (absolute paths), skipping the rest of the
            folder. ``None`` -- the default -- means the whole folder, exactly as before.
            Restricting the file list rather than swapping in a different source stage is what
            lets a delta run over new files produce rows identical to a full run's.
    """

    data_dir: str
    extensions: list[str] = field(default_factory=lambda: [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"])
    recursive: bool = True
    max_samples: int = -1
    include_files: list[str] | None = None
    audio_filepath_key: str = "audio_filepath"
    audio_item_id_key: str = "audio_item_id"
    name: str = "CreateInitialManifestAudioFolder"
    batch_size: int = 1
    # See ManifestReaderStage: the narrowing claim has to survive being read off the class.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=False))

    def __post_init__(self) -> None:
        super().__init__()
        from nemo_curator.stages.audio._agent._residency import validate_audio_key_configuration

        validate_audio_key_configuration(
            self.name,
            input_keys={},
            output_keys={
                "audio_filepath_key": self.audio_filepath_key,
                "audio_item_id_key": self.audio_item_id_key,
            },
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.audio_item_id_key]

    def describe(self) -> StageContract:
        return StageContract(
            # No ``produces``: the audio already exists on disk, this stage only points at it
            # (unlike the dataset CreateInitialManifest*Stage sources, which download and write).
            writes=IOSpec(data_keys=[self.audio_filepath_key, self.audio_item_id_key]),
            cardinality="1:N fan-out",
            # A bounded sorted listing depends on the whole folder: adding an earlier path can
            # change which existing rows belong to the first N. Unbounded scans remain safely
            # narrowable one file at a time.
            gates=Gates(per_row_independent=self.max_samples is None or self.max_samples < 0),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1

    @staticmethod
    def _item_id(relative_path: str) -> str:
        """Flatten a relative stem without confusing separators with filename text."""
        stem, extension = os.path.splitext(relative_path)
        encoded_components: list[str] = []
        for component in stem.split(os.sep):
            encoded: list[str] = []
            last = len(component) - 1
            for index, char in enumerate(component):
                if char == "~":
                    # Reserve ``~`` for the underscore escape below.
                    encoded.append("~~")
                elif char == "_" and (
                    index in (0, last) or component[index - 1] == "_" or component[index + 1] == "_"
                ):
                    # Encoded components must neither contain ``__`` nor touch a
                    # separator with ``_``; otherwise two different component
                    # boundaries can flatten to the same string.
                    encoded.append("~u")
                else:
                    encoded.append(char)
            encoded_components.append("".join(encoded))
        return "__".join(encoded_components) + "~e" + extension[1:].replace("~", "~~")

    def _collect_audio_files(self) -> list[str]:
        exts = tuple((e if e.startswith(".") else f".{e}").lower() for e in self.extensions)
        if not os.path.isdir(self.data_dir):
            logger.error(f"[{self.name}] data_dir not found: {self.data_dir}")
            return []
        found: list[str] = []
        if self.recursive:
            for root, _dirs, files in os.walk(self.data_dir):
                found.extend(os.path.join(root, f) for f in files if f.lower().endswith(exts))
        else:
            found = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.lower().endswith(exts) and os.path.isfile(os.path.join(self.data_dir, f))
            ]
        if self.include_files is not None:
            wanted = {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
            found = [p for p in found if os.path.abspath(p) in wanted]
            missing = wanted - {os.path.abspath(p) for p in found}
            if missing:
                # Named rather than silently skipped: a caller that asked for specific files and
                # got fewer would otherwise read the short result as "those files held nothing".
                logger.warning(
                    f"[{self.name}] include_files named {len(missing)} file(s) not found under {self.data_dir}"
                )
        return sorted(found)

    def process(self, task: EmptyTask | None) -> list[AudioTask]:
        """Emit one AudioTask per audio file found under ``data_dir``."""
        paths = self._collect_audio_files()
        if self.max_samples is not None and self.max_samples >= 0:
            paths = paths[: self.max_samples]
        if not paths:
            logger.warning(f"[{self.name}] no audio files {self.extensions} under {self.data_dir}")
            return []
        tasks: list[AudioTask] = []
        for path in paths:
            abspath = os.path.abspath(path)
            # Relpath, not basename: ``recursive`` defaults True and speaker-per-folder is the
            # standard layout, so a basename id gives spk1/utt1.wav and spk2/utt1.wav the same
            # id -- and downstream that id becomes an output filename. A flat corpus is
            # unaffected unless its name needs escaping to remain distinct from a path separator.
            rel = os.path.relpath(abspath, os.path.abspath(self.data_dir))
            item_id = self._item_id(rel)
            tasks.append(
                _AudioFolderTask(
                    dataset_name="local-audio-folder",
                    data={self.audio_filepath_key: abspath, self.audio_item_id_key: item_id},
                    filepath_key=self.audio_filepath_key,
                    _metadata={
                        **({} if task is None else task._metadata),
                        "audio_folder_source_id": hashlib.sha256(abspath.encode("utf-8")).hexdigest(),
                    },
                    _stage_perf=[] if task is None else list(task._stage_perf),
                )
            )
        logger.info(f"[{self.name}] created {len(tasks)} AudioTask(s) from {self.data_dir}")
        return tasks
