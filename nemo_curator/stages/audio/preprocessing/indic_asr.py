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

"""Input preparation for the Hindi Indic Canary pipeline."""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import ClassVar

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract
from nemo_curator.stages.audio.common import load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

SAMPLE_RATE = 16_000
MAX_DURATION_ERROR_S = 0.001


@dataclass
class PrepareIndicASRInputStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Load 16 kHz Hindi audio and initialize per-clip Granary-v2 fields.

    The input reference is removed from ``text_key`` and copied unchanged to
    both reference outputs; it is not an ASR prediction. Audio is downmixed to
    mono float32, retained as a [1, samples] waveform, and checked against the
    manifest duration within 1 ms without changing that duration. The resolved
    file path and sample rate in Hz describe that same clip. Existing notes
    survive while model-selection labels are initialized, and the skip reason
    is cleared. This stage does not transcribe audio or verify reference accuracy.

    Args:
        audio_root: Base directory for relative audio paths.
        audio_filepath_key: Input path, replaced with its absolute resolved path.
        audio_item_id_key: Nonempty identity of the clip.
        duration_key: Manifest duration in seconds, retained unchanged.
        source_lang_key: Language code, required to be "hi".
        text_key: Nonempty input reference text; removed after reading.
        notes_key: Shared mapping of model labels and processing notes.
        granary_v1_prediction_key: Legacy reference copy, not a model prediction.
        reference_text_key: Reference copy used for downstream WER.
        sample_rate_key: Decoded waveform sample rate in Hz.
        waveform_key: In-memory mono float32 waveform tensor.
        skip_me_key: Processing skip reason, initialized to an empty string.
    """

    name: str = "prepare_indic_asr_input"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    audio_root: str | None = None

    _: KW_ONLY
    audio_filepath_key: str = "audio_filepath"
    audio_item_id_key: str = "audio_item_id"
    duration_key: str = "duration"
    source_lang_key: str = "source_lang"
    text_key: str = "text"
    notes_key: str = "additional_notes"
    granary_v1_prediction_key: str = "granary_v1_prediction"
    reference_text_key: str = "reference_text"
    sample_rate_key: str = "sampling_rate"
    waveform_key: str = "waveform"
    skip_me_key: str = "_skipme"

    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {
        "granary_v1_prediction_key": "reference_text",
        # The shared role vocabulary has no language or audit-note role.
        "source_lang_key": "unknown",
        "notes_key": "unknown",
        "skip_me_key": "unknown",
    }

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=self.inputs()[1], accepts=["file"]),
            optional_reads=IOSpec(data_keys=[self.notes_key]),
            writes=IOSpec(data_keys=self.outputs()[1], produces=["tensor"]),
            removes_keys=[self.text_key],
            cardinality="1:1",
            gates=Gates(requires_gpu=self.resources.requires_gpu),
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.audio_item_id_key,
            self.duration_key,
            self.source_lang_key,
            self.text_key,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.notes_key,
            self.granary_v1_prediction_key,
            self.reference_text_key,
            self.sample_rate_key,
            self.waveform_key,
            self.skip_me_key,
        ]

    def process(self, task: AudioTask) -> AudioTask:
        data = task.data
        audio_item_id = data[self.audio_item_id_key]
        reference_text = data.pop(self.text_key)
        if not isinstance(audio_item_id, str) or not audio_item_id:
            msg = "audio_item_id must be a nonempty string"
            raise RuntimeError(msg)
        if not isinstance(reference_text, str) or not reference_text.strip():
            msg = f"Reference text is empty for {audio_item_id}"
            raise RuntimeError(msg)
        if data[self.source_lang_key] != "hi":
            msg = f"Expected source_lang='hi' for {audio_item_id}, found {data[self.source_lang_key]!r}"
            raise RuntimeError(msg)

        audio_path = Path(data[self.audio_filepath_key])
        if not audio_path.is_absolute():
            if self.audio_root is None:
                msg = f"Relative audio path requires audio_root for {audio_item_id}: {audio_path}"
                raise RuntimeError(msg)
            audio_path = Path(self.audio_root) / audio_path
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.is_file():
            msg = f"Input audio is missing for {audio_item_id}: {audio_path}"
            raise FileNotFoundError(msg)
        data[self.audio_filepath_key] = str(audio_path)

        waveform, sample_rate = load_audio_file(str(audio_path), mono=True)
        duration_s = float(data[self.duration_key])
        measured_duration_s = waveform.shape[-1] / sample_rate
        if sample_rate != SAMPLE_RATE or waveform.shape[0] != 1:
            msg = f"Expected mono {SAMPLE_RATE} Hz audio for {audio_item_id}"
            raise RuntimeError(msg)
        if not math.isclose(duration_s, measured_duration_s, rel_tol=0, abs_tol=MAX_DURATION_ERROR_S):
            msg = (
                f"Manifest duration mismatch for {audio_item_id}: "
                f"manifest={duration_s}, measured={measured_duration_s}"
            )
            raise RuntimeError(msg)

        notes = data.get(self.notes_key)
        notes = notes if isinstance(notes, dict) else {}
        notes.update({"primary_model": "indic_canary", "recovery_model": "parakeet_riva"})
        data[self.notes_key] = notes
        data[self.granary_v1_prediction_key] = reference_text
        data[self.reference_text_key] = reference_text
        data[self.sample_rate_key] = sample_rate
        data[self.waveform_key] = waveform
        data[self.skip_me_key] = ""
        return task
