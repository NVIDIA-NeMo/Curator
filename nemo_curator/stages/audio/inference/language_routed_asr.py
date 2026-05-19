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

"""Language-routed ASR recovery: Indic Conformer, Parakeet-TDT v3, Faster-Whisper, or Qwen3-ASR by ``source_lang``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.models.faster_whisper_asr import FasterWhisperASR
from nemo_curator.models.indic_conformer_asr import IndicConformerASR
from nemo_curator.models.qwen_asr import QwenASR
from nemo_curator.stages.audio.inference.asr_nemo import NemoASRModel
from nemo_curator.stages.audio.pipeline_utils import (
    LANG_CODE_TO_NAME,
    resolve_indic_language_code,
    resolve_parakeet_language_code,
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

    Priority: Indic Conformer (Indic codes) → Parakeet-TDT v3 (``PARAKEET_LANGUAGE_CODES``:
    ``lv``/``hr``/``et``/``bg``/``sk``/``sl``/``mt``/``uk``/``lt``)
    → Faster-Whisper (``WHISPER_ROUTED_LANGUAGE_CODES``)
    → Qwen3-ASR for remaining languages when ``qwen_model_id`` is set.

    Parakeet-TDT v3 is loaded through
    :class:`nemo_curator.stages.audio.inference.asr_nemo.NemoASRModel`,
    the same wrapper used by ``InferenceAsrNemoStage``. We invoke its
    ``transcribe_waveforms`` method to feed in-memory native-SR waveforms straight
    from the upstream reader (no temp ``.wav`` files).

    Outputs per task: ``pred_text_key`` (default ``asr_prediction``),
    ``language_key`` (default ``asr_language``) — backend-specific language label
    (Indic ISO code, Parakeet ISO code, Whisper ``language`` code such as ``tl``,
    or Qwen-detected name), and ``additional_notes[asr_model_key]`` (default key
    ``asr_model``) — one of ``indic_conformer``, ``parakeet_v3``, ``faster_whisper``,
    ``qwen3_asr`` for samples that ran, absent for skipped ones.

    Memory mode (``single_resident_models``):

    - **True (default)**: lazy-load + evict. At any moment, at most **one**
      backend lives on the GPU. The first time a batch needs backend ``X``,
      the currently-resident backend is torn down (CUDA cache freed) and ``X``
      is loaded. Within a single batch, items are grouped by backend and the
      currently-resident backend is processed first to avoid an unnecessary
      swap. Trades inference latency (cold-load on swap: vLLM ~30–60 s,
      NeMo ~5–10 s, Whisper ~3–5 s) for ~70+ GB lower idle GPU footprint.
    - **False**: legacy behaviour — all enabled backends are loaded once at
      worker startup and stay resident. Lower per-batch latency but every
      enabled backend permanently occupies its share of GPU memory.

    Args:
        qwen_model_id: HuggingFace id or path for Qwen3-ASR; omit when unused.
        indic_model_id: HuggingFace id for Indic Conformer; omit when unused.
        whisper_model_size_or_path: faster-whisper model name or path; omit when unused.
        parakeet_model_id: NeMo / HuggingFace id for Parakeet-TDT v3
            (e.g. ``nvidia/parakeet-tdt-0.6b-v3``); omit when unused.
        parakeet_cache_dir: Optional NeMo cache directory for Parakeet downloads.
        parakeet_inference_batch_size: Batch size passed to Parakeet ``transcribe``.
        indic_decode_mode: ``ctc`` or ``rnnt`` (Indic Conformer model card).
        whisper_device: ``cuda``, ``cpu``, or ``auto``.
        whisper_compute_type: faster-whisper compute type (e.g. ``float16``, ``int8``).
        source_lang_key: Per-sample language field (ISO code or English display name).
        single_resident_models: When True (default), only one backend is GPU-resident
            at a time and others are swapped in on demand. See "Memory mode" above.
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
    parakeet_model_id: str | None = None
    parakeet_cache_dir: str | None = None
    parakeet_inference_batch_size: int = 16
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    asr_model_key: str = "asr_model"
    context_key: str | None = None
    run_only_if_key: str | None = None
    run_only_if_prefix: str = "Hallucination"
    notes_key: str = "additional_notes"
    gpu_memory_utilization: float = 0.7
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 128
    num_workers_override: int | None = None
    single_resident_models: bool = True
    # When set to one of ``"qwen" / "indic" / "whisper" / "parakeet"``, the stage
    # routes EVERY sample to that backend regardless of ``source_lang``. Used to
    # reuse this stage as the language-specific primary inference step where the
    # caller has already picked the correct backend for the (single) language in
    # play. The corresponding backend must also be enabled (model id provided).
    force_backend: Literal["qwen", "indic", "whisper", "parakeet"] | None = None
    # When True, the waveform is left on the task so the next ASR stage can
    # reuse it. Default False matches recovery-stage behaviour (pop to free RAM).
    keep_waveform: bool = False
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    _qwen_model: QwenASR | None = field(default=None, init=False, repr=False)
    _indic_model: IndicConformerASR | None = field(default=None, init=False, repr=False)
    _whisper_model: FasterWhisperASR | None = field(default=None, init=False, repr=False)
    _parakeet_model: NemoASRModel | None = field(default=None, init=False, repr=False)
    _supported_qwen_langs: set[str] = field(default_factory=set, init=False, repr=False)
    _resident_backend: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (
            self.qwen_model_id
            or self.indic_model_id
            or self.whisper_model_size_or_path
            or self.parakeet_model_id
        ):
            msg = (
                "At least one of qwen_model_id, indic_model_id, whisper_model_size_or_path, "
                "or parakeet_model_id is required"
            )
            raise ValueError(msg)
        # Wrappers are cheap to instantiate (no GPU memory yet); we always
        # build them so that single-resident mode can call setup()/teardown()
        # on demand and all-resident mode can call setup() once in setup_on_node.
        if self.qwen_model_id and self._qwen_model is None:
            self._qwen_model = QwenASR(
                model_id=self.qwen_model_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_new_tokens=self.max_new_tokens,
                max_inference_batch_size=self.max_inference_batch_size,
            )
        if self.indic_model_id and self._indic_model is None:
            self._indic_model = IndicConformerASR(
                model_id=self.indic_model_id,
                decode_mode=self.indic_decode_mode,
            )
        if self.whisper_model_size_or_path and self._whisper_model is None:
            self._whisper_model = FasterWhisperASR(
                model_size_or_path=self.whisper_model_size_or_path,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
                download_root=self.whisper_download_root,
                beam_size=self.whisper_beam_size,
                vad_filter=self.whisper_vad_filter,
            )
        if self.parakeet_model_id and self._parakeet_model is None:
            self._parakeet_model = NemoASRModel(
                model_name=self.parakeet_model_id,
                cache_dir=self.parakeet_cache_dir,
                inference_batch_size=self.parakeet_inference_batch_size,
            )

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
        # Always pre-warm the NeMo cache for Parakeet so the (potentially slow)
        # checkpoint download happens at node init, not in the first batch.
        if self._parakeet_model is not None:
            self._parakeet_model.setup_on_node()

        if not self.single_resident_models:
            # Legacy all-resident mode: load every enabled backend now and keep
            # it on the GPU for the worker's lifetime.
            if self._qwen_model is not None:
                self._qwen_model.setup()
            if self._indic_model is not None:
                self._indic_model.setup()
            if self._whisper_model is not None:
                self._whisper_model.setup()
            if self._parakeet_model is not None:
                self._parakeet_model.setup(device=self._cuda_device())

        logger.info(
            f"Language-routed ASR ready: qwen={bool(self._qwen_model)} "
            f"indic={bool(self._indic_model)} whisper={bool(self._whisper_model)} "
            f"parakeet={bool(self._parakeet_model)} "
            f"mode={'single_resident' if self.single_resident_models else 'all_resident'}"
        )

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        # In single-resident mode we lazy-load on first use inside process_batch().
        if self.single_resident_models:
            return

        # All-resident mode: defensively re-load anything that didn't survive the
        # node→worker transition (e.g. when setup_on_node ran in a different
        # process). Wrappers are no-ops if already loaded.
        if self._qwen_model is not None:
            self._qwen_model.setup()
        if self._indic_model is not None:
            self._indic_model.setup()
        if self._whisper_model is not None:
            self._whisper_model.setup()
        if self._parakeet_model is not None and self._parakeet_model.asr_model is None:
            self._parakeet_model.setup(device=self._cuda_device())

    def teardown(self) -> None:
        for backend, wrapper in (
            ("qwen", self._qwen_model),
            ("indic", self._indic_model),
            ("whisper", self._whisper_model),
            ("parakeet", self._parakeet_model),
        ):
            if wrapper is None:
                continue
            try:
                wrapper.teardown()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error tearing down ASR backend {backend}: {e}")
        self._resident_backend = None

    @staticmethod
    def _cuda_device() -> Any:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _is_loaded(self, backend: str) -> bool:
        if backend == "qwen":
            return self._qwen_model is not None and self._qwen_model._model is not None
        if backend == "indic":
            return self._indic_model is not None and self._indic_model._model is not None
        if backend == "whisper":
            return self._whisper_model is not None and self._whisper_model._model is not None
        if backend == "parakeet":
            return self._parakeet_model is not None and self._parakeet_model.asr_model is not None
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)

    def _load_backend(self, backend: str) -> None:
        """Make ``backend`` GPU-resident, evicting any other resident backend first.

        No-op when ``single_resident_models=False`` (all backends are already
        resident from setup_on_node).
        """
        if not self.single_resident_models:
            return
        if self._resident_backend == backend and self._is_loaded(backend):
            return
        # Evict whatever is currently resident before loading the new model.
        if self._resident_backend is not None and self._resident_backend != backend:
            self._evict_current()
        logger.info(f"Loading ASR backend onto GPU: {backend}")
        if backend == "qwen":
            assert self._qwen_model is not None
            self._qwen_model.setup()
        elif backend == "indic":
            assert self._indic_model is not None
            self._indic_model.setup()
        elif backend == "whisper":
            assert self._whisper_model is not None
            self._whisper_model.setup()
        elif backend == "parakeet":
            assert self._parakeet_model is not None
            self._parakeet_model.setup(device=self._cuda_device())
        else:
            msg = f"Unknown backend: {backend}"
            raise ValueError(msg)
        self._resident_backend = backend

    def _evict_current(self) -> None:
        """Tear down the currently-resident backend and free GPU memory."""
        backend = self._resident_backend
        if backend is None:
            return
        logger.info(f"Evicting ASR backend from GPU: {backend}")
        try:
            if backend == "qwen" and self._qwen_model is not None:
                self._qwen_model.teardown()
            elif backend == "indic" and self._indic_model is not None:
                self._indic_model.teardown()
            elif backend == "whisper" and self._whisper_model is not None:
                self._whisper_model.teardown()
            elif backend == "parakeet" and self._parakeet_model is not None:
                self._parakeet_model.teardown()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error evicting ASR backend {backend}: {e}")
        finally:
            self._resident_backend = None
        # Best-effort double-free in case the wrapper teardown didn't fully release.
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key, self.notes_key]

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
        # In single-resident mode the wrapper exists from __post_init__ even when
        # the model isn't currently loaded — we use wrapper presence as the
        # "backend is enabled for this stage" signal and lazy-load on demand.
        indic_backend_ready = self._indic_model is not None
        whisper_backend_ready = self._whisper_model is not None
        qwen_backend_ready = self._qwen_model is not None
        parakeet_backend_ready = self._parakeet_model is not None

        indic_items: list[tuple[int, str]] = []
        parakeet_items: list[tuple[int, str]] = []
        whisper_items: list[tuple[int, str]] = []
        qwen_indices: list[int] = []
        skipped_lang: list[int] = []

        def qwen_supported_for_code(code_lc: str) -> tuple[bool, str | None]:
            lang_name = LANG_CODE_TO_NAME.get(code_lc, code_lc) if code_lc else None
            if self.source_lang_key and supported_qwen and lang_name and lang_name not in supported_qwen:
                return False, lang_name
            return True, lang_name

        def fallback_to_qwen_or_skip(i: int, code: str, label: str) -> None:
            """Specialised backend matched the language but is disabled — try Qwen, else skip."""
            if not qwen_backend_ready:
                set_note(
                    tasks[i].data, self.name,
                    f"skipped ({label} language but {label} backend disabled)",
                    self.notes_key,
                )
                skipped_lang.append(i)
                return
            ok, lang_name = qwen_supported_for_code(code)
            if ok:
                qwen_indices.append(i)
            else:
                set_note(
                    tasks[i].data, self.name,
                    f"skipped ({label} language but {label} backend disabled; "
                    f"Qwen does not support {lang_name})",
                    self.notes_key,
                )
                skipped_lang.append(i)

        # Specialised backends tried in priority order: Indic > Parakeet > Whisper.
        # Each row: (display label, code resolver, ready flag, output bucket).
        specialised_routes: tuple[tuple[str, Any, bool, list[tuple[int, str]]], ...] = (
            ("Indic", resolve_indic_language_code, indic_backend_ready, indic_items),
            ("Parakeet", resolve_parakeet_language_code, parakeet_backend_ready, parakeet_items),
            ("Whisper", resolve_whisper_language_code, whisper_backend_ready, whisper_items),
        )

        for i in run_indices:
            raw_lang = tasks[i].data.get(self.source_lang_key, "")
            raw_s = str(raw_lang).strip() if raw_lang is not None else ""
            lang_arg = raw_lang if isinstance(raw_lang, str) else raw_s

            matched_specialised = False
            for label, resolver, backend_ready, bucket in specialised_routes:
                code = resolver(lang_arg)
                if code is None:
                    continue
                if backend_ready:
                    bucket.append((i, code))
                else:
                    fallback_to_qwen_or_skip(i, code, label)
                matched_specialised = True
                break

            if matched_specialised:
                continue

            # Qwen is the catch-all for languages no specialised backend recognised.
            if not qwen_backend_ready:
                set_note(
                    tasks[i].data, self.name,
                    "skipped (language needs Qwen backend but Qwen disabled)",
                    self.notes_key,
                )
                skipped_lang.append(i)
                continue

            ok, lang_name = qwen_supported_for_code(raw_s.lower())
            if not ok:
                set_note(tasks[i].data, self.name, f"skipped (unsupported language: {lang_name})", self.notes_key)
                skipped_lang.append(i)
                continue

            qwen_indices.append(i)

        if indic_items and self._indic_model is None:
            msg = "Indic Conformer samples present but model was not initialized"
            raise RuntimeError(msg)
        if parakeet_items and self._parakeet_model is None:
            msg = "Parakeet samples present but model was not initialized"
            raise RuntimeError(msg)
        if whisper_items and self._whisper_model is None:
            msg = "Faster-Whisper samples present but model was not initialized"
            raise RuntimeError(msg)
        if qwen_indices and self._qwen_model is None:
            msg = "QwenASR samples present but model was not initialized"
            raise RuntimeError(msg)

        # Build the work plan: one (backend, runner) entry per non-empty bucket.
        # Order matters in single-resident mode — putting the currently-resident
        # backend first lets us skip a swap when the previous batch's tail
        # backend matches this batch's head.
        plan: list[tuple[str, Any]] = []
        if indic_items:
            plan.append(("indic", lambda: self._run_indic(indic_items, tasks)))
        if parakeet_items:
            plan.append(("parakeet", lambda: self._run_parakeet(parakeet_items, tasks)))
        if whisper_items:
            plan.append(("whisper", lambda: self._run_whisper(whisper_items, tasks)))
        if qwen_indices:
            plan.append(("qwen", lambda: self._run_qwen(qwen_indices, tasks)))

        if self.single_resident_models and self._resident_backend is not None:
            plan.sort(key=lambda entry: entry[0] != self._resident_backend)

        for backend_name, runner in plan:
            self._load_backend(backend_name)
            runner()

        for task in tasks:
            task.data.pop(self.waveform_key, None)

        logger.info(
            f"Language-routed ASR: indic={len(indic_items)} parakeet={len(parakeet_items)} "
            f"whisper={len(whisper_items)} qwen={len(qwen_indices)} "
            f"skipped_run={len(tasks) - len(run_indices)} skipped_lang={len(skipped_lang)} "
            f"resident={self._resident_backend}"
        )
        return tasks

    def _run_indic(self, items: list[tuple[int, str]], tasks: list[AudioTask]) -> None:
        idxs = [p[0] for p in items]
        langs = [p[1] for p in items]
        waves = [tasks[j].data[self.waveform_key] for j in idxs]
        srs = [tasks[j].data[self.sample_rate_key] for j in idxs]
        assert self._indic_model is not None
        pred_texts, langs_out = self._indic_model.generate(waves, srs, langs)
        for j, pred, lo in zip(idxs, pred_texts, langs_out, strict=True):
            tasks[j].data[self.pred_text_key] = pred
            tasks[j].data[self.language_key] = lo
            set_note(tasks[j].data, self.asr_model_key, "indic_conformer", self.notes_key)

    def _run_parakeet(self, items: list[tuple[int, str]], tasks: list[AudioTask]) -> None:
        idxs = [p[0] for p in items]
        plangs = [p[1] for p in items]
        assert self._parakeet_model is not None
        pred_texts = self._parakeet_model.transcribe_waveforms(
            [tasks[j].data[self.waveform_key] for j in idxs],
            [tasks[j].data[self.sample_rate_key] for j in idxs],
        )
        for j, pred, lo in zip(idxs, pred_texts, plangs, strict=True):
            tasks[j].data[self.pred_text_key] = pred
            tasks[j].data[self.language_key] = lo
            set_note(tasks[j].data, self.asr_model_key, "parakeet_v3", self.notes_key)

    def _run_whisper(self, items: list[tuple[int, str]], tasks: list[AudioTask]) -> None:
        idxs = [p[0] for p in items]
        wlangs = [p[1] for p in items]
        waves = [tasks[j].data[self.waveform_key] for j in idxs]
        srs = [tasks[j].data[self.sample_rate_key] for j in idxs]
        assert self._whisper_model is not None
        pred_texts, langs_out = self._whisper_model.generate(waves, srs, wlangs)
        for j, pred, lo in zip(idxs, pred_texts, langs_out, strict=True):
            tasks[j].data[self.pred_text_key] = pred
            tasks[j].data[self.language_key] = lo
            set_note(tasks[j].data, self.asr_model_key, "faster_whisper", self.notes_key)

    def _run_qwen(self, indices: list[int], tasks: list[AudioTask]) -> None:
        waves = [tasks[j].data[self.waveform_key] for j in indices]
        srs = [tasks[j].data[self.sample_rate_key] for j in indices]
        contexts = [tasks[j].data.get(self.context_key, "") for j in indices] if self.context_key else None
        languages: list[str | None] | None = None
        if self.source_lang_key:
            languages = [
                LANG_CODE_TO_NAME.get(c, c) if c else None
                for c in (
                    str(tasks[j].data.get(self.source_lang_key, "") or "").strip()
                    for j in indices
                )
            ]

        assert self._qwen_model is not None
        pred_texts, detected = self._qwen_model.generate(waves, srs, contexts, languages)
        for j, pred, det in zip(indices, pred_texts, detected, strict=True):
            tasks[j].data[self.pred_text_key] = pred
            tasks[j].data[self.language_key] = det
            set_note(tasks[j].data, self.asr_model_key, "qwen3_asr", self.notes_key)
