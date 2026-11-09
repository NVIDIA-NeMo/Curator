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

"""Materialize resident VAD segments before writing their metadata to JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    Gates,
    IOSpec,
    StageContract,
    StaticHints,
)
from nemo_curator.stages.audio._agent._residency import normalize_audio_waveform
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

_SAMPLE_RATE = 16000


@dataclass
class AudioSegmentWriterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Write 16 kHz mono segments as Opus and retain JSON-safe metadata.

    ``audio_filepath_key`` becomes an absolute path below ``output_dir/audio``;
    ``original_file_key`` keeps the source recording. ``duration_key`` is the
    encoded waveform's sample count divided by 16000. Existing VAD timestamps,
    speaker counts, RTTM paths, SED events, LID decisions and audit notes survive.
    ``audio_item_id_key`` identifies the segment, replacing any parent ID.
    Whole-recording diarization and resident/transient audio fields are removed.

    A ``vad_empty_key`` audit row is retained with its original recording path
    and source duration, without producing a clip. ``read_error_key`` and invalid
    segment audio raise. LID rejection flags do not prevent writing a segment.
    A failed encode leaves the input task untouched and never publishes a partial
    file. Output names depend on the source, interval and PCM content, so equal
    basenames and retried work cannot collide with different segments.

    Args:
        output_dir: Local directory shared by every executor node.
    """

    output_dir: str
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    audio_filepath_key: str = "audio_filepath"
    audio_item_id_key: str = "audio_item_id"
    original_file_key: str = "original_file"
    duration_key: str = "duration"
    start_ms_key: str = "start_ms"
    end_ms_key: str = "end_ms"
    vad_empty_key: str = "vad_empty"
    read_error_key: str = "read_error"
    framewise_key: str = "_sed_framewise"
    resampled_audio_filepath_key: str = "resampled_audio_filepath"
    num_samples_key: str = "num_samples"
    is_mono_key: str = "is_mono"
    diar_segments_key: str = "diar_segments"
    name: str = "AudioSegmentWriter"
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"vad_empty_key", "read_error_key", "framewise_key"})

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_dir"],
            sanitizes_output=True,
            per_row_independent=True,
        ),
        error_policy="fail",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip() or "://" in self.output_dir:
            msg = "output_dir must be a non-empty local directory"
            raise ValueError(msg)
        keys = [getattr(self, item.name) for item in fields(self) if item.name.endswith("_key")]
        if any(not isinstance(key, str) or not key.strip() for key in keys) or len(keys) != len(set(keys)):
            msg = "AudioSegmentWriterStage keys must be distinct, non-empty strings"
            raise ValueError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if "OPUS" not in sf.available_subtypes("OGG"):
            msg = "AudioSegmentWriterStage requires soundfile/libsndfile with Ogg Opus encoding support"
            raise RuntimeError(msg)

    def _removed_keys(self) -> list[str]:
        return [
            self.waveform_key,
            self.framewise_key,
            self.resampled_audio_filepath_key,
            self.num_samples_key,
            self.is_mono_key,
            self.diar_segments_key,
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.original_file_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key]),
            reads_one_of=[
                IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"]),
                IOSpec(data_keys=[self.vad_empty_key]),
            ],
            optional_reads=IOSpec(
                data_keys=[self.original_file_key, self.start_ms_key, self.end_ms_key, self.read_error_key]
            ),
            writes=IOSpec(data_keys=[self.audio_filepath_key, self.original_file_key]),
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(
                        data_keys=[self.audio_filepath_key, self.audio_item_id_key, self.duration_key],
                        produces=["disk"],
                    ),
                    condition=f"'{self.vad_empty_key}' is false and segment encoding succeeds",
                ),
            ],
            removes_keys=self._removed_keys(),
            cardinality="1:1",
            gates=self.AGENT_STATIC.gates,
            error_policy="fail",
        )

    def process(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.read_error_key):
            msg = "Cannot write an audio segment marked read_error"
            raise ValueError(msg)
        source = task.data.get(self.original_file_key) or task.data.get(self.audio_filepath_key)
        if not isinstance(source, str) or not source:
            msg = "A source recording path is required to identify audio segments"
            raise ValueError(msg)
        data = {key: value for key, value in task.data.items() if key not in self._removed_keys()}
        data[self.original_file_key] = source
        if task.data.get(self.vad_empty_key):
            data[self.audio_filepath_key] = source
            json.dumps(data, allow_nan=False)
            task.data = data
            return task

        if task.data.get(self.sample_rate_key) != _SAMPLE_RATE:
            msg = "AudioSegmentWriterStage requires 16000 Hz segment audio"
            raise ValueError(msg)
        waveform = normalize_audio_waveform(task.data[self.waveform_key], stage_name=self.name, mono=False)
        if waveform.shape[0] != 1 or waveform.shape[1] == 0 or not torch.isfinite(waveform).all():
            msg = "AudioSegmentWriterStage requires non-empty, finite mono segment audio"
            raise ValueError(msg)
        samples = waveform.cpu().numpy().reshape(-1).astype("<f4", copy=False)
        identity = json.dumps([source, task.data.get(self.start_ms_key), task.data.get(self.end_ms_key)]).encode()
        digest = hashlib.sha256(identity + samples.tobytes()).hexdigest()
        output_dir = Path(self.output_dir).expanduser().resolve() / "audio"
        output_path = output_dir / f"{digest}.opus"
        data[self.audio_filepath_key] = str(output_path)
        data[self.audio_item_id_key] = digest
        data[self.duration_key] = waveform.shape[1] / _SAMPLE_RATE
        json.dumps(data, allow_nan=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".opus", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            sf.write(temporary_path, samples, _SAMPLE_RATE, format="OGG", subtype="OPUS")
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        task.data = data
        return task
