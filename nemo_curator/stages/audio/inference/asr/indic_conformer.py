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

"""Standalone AI4Bharat Indic Conformer inference stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.models.indic_conformer_asr import IndicConformerASR
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

INDIC_CONFORMER_600M_MULTILINGUAL_LANGS: frozenset[str] = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)


@dataclass
class InferenceIndicConformerStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription using the AI4Bharat Indic Conformer multilingual model.

    Designed for Indic-language invocations (e.g. hi/ur). The ISO ``source_lang``
    value is passed directly to the model as the forced language code — no per-sample
    routing is performed.

    The HuggingFace repo is gated; set ``HF_TOKEN`` in the environment before running.

    Args:
        model_id: HuggingFace model identifier (default:
            ``"ai4bharat/indic-conformer-600m-multilingual"``).
        decode_mode: ``"ctc"`` or ``"rnnt"`` (see model card).
        source_lang_key: Task data key holding the per-sample ISO language code
            (e.g. ``"hi"``, ``"ur"``).
        waveform_key: Task data key for the mono float32 numpy waveform.
        sample_rate_key: Task data key for the integer sample rate.
        pred_text_key: Output key for the predicted transcription.
        language_key: Output key storing the source language code (passed through).
        notes_key: Top-level key used for ``additional_notes`` metadata.
        keep_waveform: When True the waveform stays on the task so a downstream
            stage can re-use it.
        num_workers_override: Fixed Ray actor count. None = autoscaler decides.
    """

    name: str = "IndicConformer_inference"
    model_id: str = "ai4bharat/indic-conformer-600m-multilingual"
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
    _model: IndicConformerASR | None = field(default=None, init=False, repr=False)

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

    def _create_model(self) -> IndicConformerASR:
        return IndicConformerASR(
            model_id=self.model_id,
            decode_mode=self.decode_mode,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._model = self._create_model()
        self._model.setup()
        logger.info(f"Indic Conformer model ready on node: {self.model_id}")

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
        msg = "InferenceIndicConformerStage only supports process_batch"
        raise NotImplementedError(msg)

    def _filter_eligible(self, tasks: list[AudioTask]) -> list[int]:
        """Return indices of tasks whose source language is supported."""
        eligible_indices: list[int] = []
        for i, task in enumerate(tasks):
            lang = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            if lang not in INDIC_CONFORMER_600M_MULTILINGUAL_LANGS:
                set_note(task.data, self.name, f"skipped (unsupported language: {lang})", self.notes_key)
                set_note(task.data, self.pred_text_key, f"lang_not_supported:{lang}", self.notes_key)
            else:
                eligible_indices.append(i)
        return eligible_indices

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        eligible_indices = self._filter_eligible(tasks)
        lang_skipped = len(tasks) - len(eligible_indices)

        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            logger.info(f"IndicConformer: skipped entire batch of {len(tasks)} (no supported languages)")
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
            f"IndicConformer: generated {len(eligible_indices)} predictions, "
            f"skipped {lang_skipped} (unsupported language)"
        )
        return tasks
