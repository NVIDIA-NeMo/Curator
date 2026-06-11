# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone faster-whisper inference stage for a single language group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.models.faster_whisper_asr import FasterWhisperASR
from nemo_curator.stages.audio.pipeline_utils import MODEL_LANG_CODE_TO_WHISPER, set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

WHISPER_LARGE_V3_LANGS: frozenset[str] = frozenset({
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
})


@dataclass
class InferenceFasterWhisperStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription using faster-whisper with a forced language code per sample.

    Designed for single-language-group invocations where every sample belongs to
    the same language family (e.g. WHISPER_PRIMARY_LANGUAGE_CODES). The ISO
    ``source_lang`` value is passed directly to Whisper as the forced ``language``
    parameter, so no per-sample routing is needed.

    Args:
        model_size_or_path: faster-whisper model name or local path (e.g. ``"large-v3"``).
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        compute_type: faster-whisper quantisation type (e.g. ``"float16"``, ``"int8"``).
        beam_size: Beam search width.
        vad_filter: Enable faster-whisper built-in VAD filtering.
        source_lang_key: Task data key holding the per-sample ISO language code.
        waveform_key: Task data key for the mono float32 numpy waveform.
        sample_rate_key: Task data key for the integer sample rate.
        pred_text_key: Output key for the predicted transcription.
        language_key: Output key for the detected/forced language code.
        notes_key: Top-level key used for ``additional_notes`` metadata.
        keep_waveform: When True the waveform stays on the task (needed when a
            downstream stage re-uses it, e.g. a recovery ASR stage).
        num_workers_override: Fixed Ray actor count. None = autoscaler decides.
    """

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

    # ------------------------------------------------------------------
    # Scaling hooks
    # ------------------------------------------------------------------

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        self._model = self._create_model()
        self._model.setup()
        logger.info(f"FasterWhisper model ready on node: {self.model_size_or_path}")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self._model = self._create_model()
            self._model.setup()

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceFasterWhisperStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        eligible_indices: list[int] = []
        eligible_lang_codes: list[str] = []
        output_exists_skipped = 0
        for i, task in enumerate(tasks):
            if self.skip_if_output_exists and task.data.get(self.pred_text_key):
                output_exists_skipped += 1
                continue
            raw_lang = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            lang = MODEL_LANG_CODE_TO_WHISPER.get(raw_lang, raw_lang)
            if lang not in WHISPER_LARGE_V3_LANGS:
                set_note(task.data, self.name, f"skipped (unsupported language: {raw_lang})", self.notes_key)
                set_note(task.data, self.pred_text_key, f"lang_not_supported:{raw_lang}", self.notes_key)
            else:
                eligible_indices.append(i)
                eligible_lang_codes.append(lang)

        lang_skipped = len(tasks) - len(eligible_indices) - output_exists_skipped
        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            if output_exists_skipped and not lang_skipped:
                logger.info(f"FasterWhisper: skipped entire batch of {len(tasks)} (output already exists)")
            elif lang_skipped and not output_exists_skipped:
                logger.info(f"FasterWhisper: skipped entire batch of {len(tasks)} (no supported languages)")
            else:
                logger.info(
                    f"FasterWhisper: skipped entire batch of {len(tasks)} "
                    f"({output_exists_skipped} output exists, {lang_skipped} unsupported language)"
                )
            return tasks

        eligible_tasks = [tasks[i] for i in eligible_indices]
        waveforms = [t.data[self.waveform_key] for t in eligible_tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in eligible_tasks]
        lang_codes = eligible_lang_codes

        pred_texts, langs_out = self._model.generate(waveforms, sample_rates, lang_codes)

        for task_idx, pred, lang in zip(eligible_indices, pred_texts, langs_out, strict=True):
            tasks[task_idx].data[self.pred_text_key] = pred
            tasks[task_idx].data[self.language_key] = lang

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        logger.info(
            f"FasterWhisper: generated {len(eligible_indices)} predictions, "
            f"skipped {lang_skipped} (unsupported language)"
        )
        return tasks
