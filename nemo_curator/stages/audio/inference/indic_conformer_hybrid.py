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

"""Curator stage adapter for AI4Bharat IndicConformer hybrid ASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.models.indic_conformer_hybrid import (
    INDIC_CONFORMER_HYBRID_LANGS,
    IndicConformerHybridASR,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


def _set_note(data: dict[str, Any], stage: str, value: str, notes_key: str) -> None:
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[stage] = value


@dataclass
class InferenceIndicConformerHybridStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription with an AI4Bharat IndicConformer hybrid (CTC+RNNT) model.

    Pipeline adapter over :class:`IndicConformerHybridASR` (same module): reads
    in-memory waveforms from each ``AudioTask``, routes per-sample by ``source_lang``,
    and writes the predicted transcription.

    Args:
        model_id: Local ``.nemo`` path or HuggingFace repo id (gated; set ``HF_TOKEN``),
            e.g. ``ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large``.
        decode_mode: ``"ctc"`` or ``"rnnt"`` (model card recommends rnnt).
        source_lang_key: Task key holding the per-sample ISO language code.
        keep_waveform: When True the waveform is left on the task for a later stage.
    """

    name: str = "IndicConformerHybrid_inference"
    model_id: str = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
    decode_mode: Literal["ctc", "rnnt"] = "rnnt"
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    keep_waveform: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    _model: IndicConformerHybridASR | None = field(default=None, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def _create_model(self) -> IndicConformerHybridASR:
        return IndicConformerHybridASR(model_id=self.model_id, decode_mode=self.decode_mode)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # Pre-download the checkpoint onto the node (HF repo -> local .nemo).
        IndicConformerHybridASR._resolve_nemo_path(self.model_id)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self._model = self._create_model()
            self._model.setup()
            logger.info(f"Indic Conformer hybrid model ready: {self.model_id}")

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceIndicConformerHybridStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: C901
        if len(tasks) == 0:
            return []
        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        eligible_indices: list[int] = []
        for i, task in enumerate(tasks):
            lang = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            if lang not in INDIC_CONFORMER_HYBRID_LANGS:
                _set_note(task.data, self.name, f"skipped (unsupported language: {lang})", self.notes_key)
                _set_note(task.data, self.pred_text_key, f"lang_not_supported:{lang}", self.notes_key)
            else:
                eligible_indices.append(i)

        lang_skipped = len(tasks) - len(eligible_indices)
        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            logger.info(f"{self.name}: skipped entire batch of {len(tasks)} (no supported languages)")
            return tasks

        eligible_tasks = [tasks[i] for i in eligible_indices]
        waveforms = [t.data[self.waveform_key] for t in eligible_tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in eligible_tasks]
        lang_codes = [str(t.data.get(self.source_lang_key, "") or "").strip().lower() for t in eligible_tasks]

        pred_texts, langs_out = self._model.generate(waveforms, sample_rates, lang_codes)

        for task_idx, pred, lang in zip(eligible_indices, pred_texts, langs_out, strict=True):
            tasks[task_idx].data[self.pred_text_key] = pred
            tasks[task_idx].data[self.language_key] = lang

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        logger.info(
            f"{self.name}: generated {len(eligible_indices)} predictions, "
            f"skipped {lang_skipped} (unsupported language)"
        )
        return tasks
