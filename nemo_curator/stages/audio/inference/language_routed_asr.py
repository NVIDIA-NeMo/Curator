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

"""Language-routed ASR recovery: Indic Conformer, Faster-Whisper, or Qwen3-ASR by ``source_lang``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.models.faster_whisper_asr import FasterWhisperASR
from nemo_curator.models.indic_conformer_asr import IndicConformerASR
from nemo_curator.models.qwen_asr import QwenASR
from nemo_curator.stages.audio.pipeline_utils import (
    LANG_CODE_TO_NAME,
    resolve_indic_language_code,
    resolve_whisper_language_code,
    set_note,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class InferenceLanguageRoutedAsrStage(ProcessingStage[AudioTask, AudioTask]):
    """Route hallucination-recovery ASR by ``source_lang`` (same outputs as ``InferenceQwenASRStage``).

    Priority: Indic Conformer (Indic codes) → Faster-Whisper (``WHISPER_ROUTED_LANGUAGE_CODES``)
    → Qwen3-ASR for remaining languages when ``qwen_model_id`` is set.

    Outputs per task: ``pred_text_key`` (default ``qwen3_asr_prediction``) and
    ``language_key`` (default ``qwen3_asr_language``) — backend-specific language label
    (Indic ISO code, Whisper ``language`` code such as ``tl``, or Qwen-detected name).

    Args:
        qwen_model_id: HuggingFace id or path for Qwen3-ASR; omit when unused.
        indic_model_id: HuggingFace id for Indic Conformer; omit when unused.
        whisper_model_size_or_path: faster-whisper model name or path; omit when unused.
        indic_decode_mode: ``ctc`` or ``rnnt`` (Indic Conformer model card).
        whisper_device: ``cuda``, ``cpu``, or ``auto``.
        whisper_compute_type: faster-whisper compute type (e.g. ``float16``, ``int8``).
        source_lang_key: Per-sample language field (ISO code or English display name).
    """

    name: str = "LanguageRoutedASR_inference"
    qwen_model_id: str | None = None
    indic_model_id: str | None = None
    indic_decode_mode: Literal["ctc", "rnnt"] = "ctc"
    whisper_model_size_or_path: str | None = None
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_download_root: str | None = None
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "qwen3_asr_prediction"
    language_key: str = "qwen3_asr_language"
    context_key: str | None = None
    run_only_if_key: str | None = None
    run_only_if_prefix: str = "Hallucination"
    notes_key: str = "additional_notes"
    gpu_memory_utilization: float = 0.7
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 128
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    _qwen_model: QwenASR | None = field(default=None, init=False, repr=False)
    _indic_model: IndicConformerASR | None = field(default=None, init=False, repr=False)
    _whisper_model: FasterWhisperASR | None = field(default=None, init=False, repr=False)
    _supported_qwen_langs: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.qwen_model_id and not self.indic_model_id and not self.whisper_model_size_or_path:
            msg = "At least one of qwen_model_id, indic_model_id, or whisper_model_size_or_path is required"
            raise ValueError(msg)

    def _get_supported_qwen_languages(self) -> set[str]:
        if not self._supported_qwen_langs:
            try:
                from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

                self._supported_qwen_langs = set(SUPPORTED_LANGUAGES)
            except ImportError:
                pass
        return self._supported_qwen_langs

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.qwen_model_id:
            self._qwen_model = QwenASR(
                model_id=self.qwen_model_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_new_tokens=self.max_new_tokens,
                max_inference_batch_size=self.max_inference_batch_size,
            )
            self._qwen_model.setup()
        if self.indic_model_id:
            self._indic_model = IndicConformerASR(
                model_id=self.indic_model_id,
                decode_mode=self.indic_decode_mode,
            )
            self._indic_model.setup()
        if self.whisper_model_size_or_path:
            self._whisper_model = FasterWhisperASR(
                model_size_or_path=self.whisper_model_size_or_path,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
                download_root=self.whisper_download_root,
                beam_size=self.whisper_beam_size,
                vad_filter=self.whisper_vad_filter,
            )
            self._whisper_model.setup()
        logger.info(
            f"Language-routed ASR ready: qwen={bool(self._qwen_model)} "
            f"indic={bool(self._indic_model)} whisper={bool(self._whisper_model)}"
        )

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._qwen_model is None and self.qwen_model_id:
            self._qwen_model = QwenASR(
                model_id=self.qwen_model_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_new_tokens=self.max_new_tokens,
                max_inference_batch_size=self.max_inference_batch_size,
            )
            self._qwen_model.setup()
        if self._indic_model is None and self.indic_model_id:
            self._indic_model = IndicConformerASR(
                model_id=self.indic_model_id,
                decode_mode=self.indic_decode_mode,
            )
            self._indic_model.setup()
        if self._whisper_model is None and self.whisper_model_size_or_path:
            self._whisper_model = FasterWhisperASR(
                model_size_or_path=self.whisper_model_size_or_path,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
                download_root=self.whisper_download_root,
                beam_size=self.whisper_beam_size,
                vad_filter=self.whisper_vad_filter,
            )
            self._whisper_model.setup()

    def teardown(self) -> None:
        if self._qwen_model is not None:
            self._qwen_model.teardown()
            self._qwen_model = None
        if self._indic_model is not None:
            self._indic_model.teardown()
            self._indic_model = None
        if self._whisper_model is not None:
            self._whisper_model.teardown()
            self._whisper_model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceLanguageRoutedAsrStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: PLR0912, PLR0915
        if len(tasks) == 0:
            return []

        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        if self.run_only_if_key:
            run_indices = [
                i
                for i, t in enumerate(tasks)
                if str(t.data.get(self.run_only_if_key, "")).startswith(self.run_only_if_prefix)
            ]
        else:
            run_indices = list(range(len(tasks)))

        if not run_indices:
            for task in tasks:
                task.data.pop(self.waveform_key, None)
            logger.info(f"Language-routed ASR: skipped batch of {len(tasks)} (none matched run_only_if_key)")
            return tasks

        supported_qwen = self._get_supported_qwen_languages()
        indic_backend_ready = self._indic_model is not None
        whisper_backend_ready = self._whisper_model is not None
        qwen_backend_ready = self._qwen_model is not None

        indic_items: list[tuple[int, str]] = []
        whisper_items: list[tuple[int, str]] = []
        qwen_indices: list[int] = []
        skipped_lang: list[int] = []

        def qwen_supported_for_code(code_lc: str) -> tuple[bool, str | None]:
            lang_name = LANG_CODE_TO_NAME.get(code_lc, code_lc) if code_lc else None
            if self.source_lang_key and supported_qwen and lang_name and lang_name not in supported_qwen:
                return False, lang_name
            return True, lang_name

        for i in run_indices:
            raw_lang = tasks[i].data.get(self.source_lang_key, "")
            raw_s = str(raw_lang).strip() if raw_lang is not None else ""
            icode = resolve_indic_language_code(raw_lang if isinstance(raw_lang, str) else raw_s)

            if icode is not None:
                if indic_backend_ready:
                    indic_items.append((i, icode))
                    continue
                if qwen_backend_ready:
                    lang_name = LANG_CODE_TO_NAME.get(icode, icode)
                    if supported_qwen and lang_name not in supported_qwen:
                        set_note(
                            tasks[i].data,
                            self.name,
                            "skipped (Indic language but no Indic backend; Qwen unsupported)",
                            self.notes_key,
                        )
                        skipped_lang.append(i)
                    else:
                        qwen_indices.append(i)
                    continue
                set_note(
                    tasks[i].data,
                    self.name,
                    "skipped (Indic language but Indic backend disabled)",
                    self.notes_key,
                )
                skipped_lang.append(i)
                continue

            wcode = resolve_whisper_language_code(raw_lang if isinstance(raw_lang, str) else raw_s)

            if wcode is not None:
                if whisper_backend_ready:
                    whisper_items.append((i, wcode))
                    continue
                if qwen_backend_ready:
                    code_lc = raw_s.lower()
                    ok, lang_name = qwen_supported_for_code(code_lc)
                    if not ok:
                        set_note(
                            tasks[i].data,
                            self.name,
                            f"skipped (unsupported language: {lang_name})",
                            self.notes_key,
                        )
                        skipped_lang.append(i)
                    else:
                        qwen_indices.append(i)
                    continue
                set_note(
                    tasks[i].data,
                    self.name,
                    "skipped (Whisper-routed language but Whisper backend disabled)",
                    self.notes_key,
                )
                skipped_lang.append(i)
                continue

            if not qwen_backend_ready:
                set_note(
                    tasks[i].data,
                    self.name,
                    "skipped (language needs Qwen backend but Qwen disabled)",
                    self.notes_key,
                )
                skipped_lang.append(i)
                continue

            code_lc = raw_s.lower()
            ok, lang_name = qwen_supported_for_code(code_lc)
            if not ok:
                set_note(tasks[i].data, self.name, f"skipped (unsupported language: {lang_name})", self.notes_key)
                skipped_lang.append(i)
                continue

            qwen_indices.append(i)

        if indic_items and self._indic_model is None:
            msg = "Indic Conformer samples present but model was not initialized"
            raise RuntimeError(msg)
        if whisper_items and self._whisper_model is None:
            msg = "Faster-Whisper samples present but model was not initialized"
            raise RuntimeError(msg)
        if qwen_indices and self._qwen_model is None:
            msg = "QwenASR samples present but model was not initialized"
            raise RuntimeError(msg)

        if indic_items:
            idxs = [p[0] for p in indic_items]
            langs = [p[1] for p in indic_items]
            waves = [tasks[j].data[self.waveform_key] for j in idxs]
            srs = [tasks[j].data[self.sample_rate_key] for j in idxs]
            assert self._indic_model is not None
            pred_texts, langs_out = self._indic_model.generate(waves, srs, langs)
            for j, pred, lo in zip(idxs, pred_texts, langs_out, strict=True):
                tasks[j].data[self.pred_text_key] = pred
                tasks[j].data[self.language_key] = lo

        if whisper_items:
            widxs = [p[0] for p in whisper_items]
            wlangs = [p[1] for p in whisper_items]
            waves = [tasks[j].data[self.waveform_key] for j in widxs]
            srs = [tasks[j].data[self.sample_rate_key] for j in widxs]
            assert self._whisper_model is not None
            pred_texts, langs_out = self._whisper_model.generate(waves, srs, wlangs)
            for j, pred, lo in zip(widxs, pred_texts, langs_out, strict=True):
                tasks[j].data[self.pred_text_key] = pred
                tasks[j].data[self.language_key] = lo

        if qwen_indices:
            waves = [tasks[j].data[self.waveform_key] for j in qwen_indices]
            srs = [tasks[j].data[self.sample_rate_key] for j in qwen_indices]
            contexts = (
                [tasks[j].data.get(self.context_key, "") for j in qwen_indices] if self.context_key else None
            )
            languages: list[str | None] | None = None
            if self.source_lang_key:
                languages = [
                    LANG_CODE_TO_NAME.get(c, c) if c else None
                    for c in (
                        str(tasks[j].data.get(self.source_lang_key, "") or "").strip()
                        for j in qwen_indices
                    )
                ]

            assert self._qwen_model is not None
            pred_texts, detected = self._qwen_model.generate(waves, srs, contexts, languages)
            for j, pred, det in zip(qwen_indices, pred_texts, detected, strict=True):
                tasks[j].data[self.pred_text_key] = pred
                tasks[j].data[self.language_key] = det

        for task in tasks:
            task.data.pop(self.waveform_key, None)

        logger.info(
            f"Language-routed ASR: indic={len(indic_items)} whisper={len(whisper_items)} "
            f"qwen={len(qwen_indices)} skipped_run={len(tasks) - len(run_indices)} "
            f"skipped_lang={len(skipped_lang)}"
        )
        return tasks
