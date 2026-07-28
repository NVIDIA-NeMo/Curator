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

"""Curator stage adapter for Faster-Whisper ASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.models.faster_whisper_asr import FasterWhisperASR
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_LANGUAGE_ALIASES = {
    "fil": "tl",
    "jv": "jw",
    "iw": "he",
    "in": "id",
    "ji": "yi",
    "nb": "no",
}
WHISPER_LARGE_V3_LANGS = frozenset(
    {
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
    }
)


def _set_note(data: dict[str, Any], stage: str, value: str, notes_key: str) -> None:
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[stage] = value


@dataclass
class InferenceFasterWhisperStage(ProcessingStage[AudioTask, AudioTask]):
    """Route waveform batches through a separately owned Faster-Whisper model."""

    name: str = "FasterWhisper_inference"
    model_size_or_path: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    keep_waveform: bool = False
    skip_if_output_exists: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    _model: FasterWhisperASR | None = field(default=None, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": self.num_workers_override} if self.num_workers_override is not None else {}

    def _create_model(self) -> FasterWhisperASR:
        return FasterWhisperASR(
            model_size_or_path=self.model_size_or_path,
            device=self.device,
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        model = self._create_model()
        model.setup()
        model.teardown()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._model = self._create_model()
        self._model.setup()

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceFasterWhisperStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []
        if self._model is None:
            msg = "InferenceFasterWhisperStage.setup() must run before process_batch()"
            raise RuntimeError(msg)

        eligible_indices: list[int] = []
        languages: list[str] = []
        for index, task in enumerate(tasks):
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")
            if self.skip_if_output_exists and task.data[self.pred_text_key]:
                continue
            raw_language = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            language = _LANGUAGE_ALIASES.get(raw_language, raw_language)
            if language not in WHISPER_LARGE_V3_LANGS:
                _set_note(task.data, self.name, f"unsupported language: {raw_language}", self.notes_key)
                _set_note(task.data, self.pred_text_key, f"lang_not_supported:{raw_language}", self.notes_key)
                continue
            eligible_indices.append(index)
            languages.append(language)

        if eligible_indices:
            selected = [tasks[index] for index in eligible_indices]
            texts, resolved_languages = self._model.generate(
                [task.data[self.waveform_key] for task in selected],
                [int(task.data[self.sample_rate_key]) for task in selected],
                languages,
            )
            for index, text, language in zip(
                eligible_indices,
                texts,
                resolved_languages,
                strict=True,
            ):
                tasks[index].data[self.pred_text_key] = text
                tasks[index].data[self.language_key] = language

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)
        return tasks
