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

"""Generic audio ASR Curator stage with a pluggable adapter.

Curator-side glue: validates I/O, resolves per-task language, pre-slices long
audio into sub-chunks, optionally re-buckets by duration (:class:`BatchPolicy`)
so one adapter call doesn't mix long and short clips, dispatches
``transcribe_batch`` per sub-batch, stitches results per parent task, and writes
predictions/metrics. The concrete adapter is resolved at runtime from
``adapter_target`` via ``hydra.utils.get_class``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra.utils
from loguru import logger

from nemo_curator.models.asr.base import ASRAdapter, ASRResult
from nemo_curator.stages.audio.inference.bucketed_stage import BucketedInferenceStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy


# ISO code -> human-readable name; the adapter receives the resolved name.
_LANG_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mt": "Maltese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


@dataclass
class ASRStage(BucketedInferenceStage[AudioTask, AudioTask, "dict[str, Any]", ASRResult]):
    """Audio speech-recognition Curator stage with pluggable adapter.

    Resolves an ``ASRAdapter`` from ``adapter_target``, slices long audio,
    optionally duration-buckets each call, and stitches chunk outputs back to
    one result per input task.
    """

    # Adapter selection.
    adapter_target: str
    model_id: str
    name: str = "ASR_inference"
    revision: str | None = None

    # Task I/O keys.
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    source_lang_key: str = "source_lang"
    default_language: str | None = None
    pred_text_key: str = "pred_text"
    disfluency_text_key: str | None = None
    skip_me_key: str = "_skip_me"

    # Chunking and output retention.
    ideal_inference_segment_s: float = 2400.0
    max_inference_duration_s: float | None = None
    keep_waveform: bool = True

    prefetch_fail_on_error: bool = True

    # Worker placement.
    xenna_num_workers: int | None = None
    xenna_num_workers_per_node: int | None = None

    batch_policy: BatchPolicy | None = None

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.ideal_inference_segment_s <= 0:
            msg = f"ASRStage.ideal_inference_segment_s must be > 0 s, got {self.ideal_inference_segment_s}"
            raise ValueError(msg)
        if self.max_inference_duration_s is None:
            self.max_inference_duration_s = self.ideal_inference_segment_s
        if self.max_inference_duration_s <= 0:
            msg = (
                f"ASRStage.max_inference_duration_s must be > 0 s, "
                f"got {self.max_inference_duration_s}"
            )
            raise ValueError(msg)
        if self.max_inference_duration_s > self.ideal_inference_segment_s:
            msg = (
                f"ASRStage.max_inference_duration_s ({self.max_inference_duration_s}) "
                f"must be ≤ ideal_inference_segment_s ({self.ideal_inference_segment_s}); "
                "the bucket shape is anchored to ideal."
            )
            raise ValueError(msg)
        if self.xenna_num_workers is not None and self.xenna_num_workers_per_node is not None:
            msg = (
                "ASRStage: set at most one of xenna_num_workers "
                "(cluster-wide) or xenna_num_workers_per_node (per-node); "
                "they are mutually exclusive."
            )
            raise ValueError(msg)
        self._adapter: ASRAdapter | None = None
        self._acc_model_metrics: dict[str, float] = defaultdict(float)
        self._inference_elapsed_s: float = 0.0
        self._warned_ray_per_node_pin = False

    def _adapter_class(self) -> type:
        return hydra.utils.get_class(self.adapter_target)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Cache model weights once per node (no GPU allocation)."""
        try:
            prefetch_t0 = time.perf_counter()
            self._adapter_class().prefetch_weights(self.model_id, self.revision)
            logger.info(
                "ASR weights cached on node for {} ({}) in {:.3f}s",
                self.model_id,
                self.adapter_target,
                time.perf_counter() - prefetch_t0,
            )
        except Exception as exc:
            msg = f"ASRStage: prefetch_weights failed for {self.model_id}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._adapter is None:
            cls = self._adapter_class()
            self._adapter = cls(
                model_id=self.model_id,
                revision=self.revision,
                **self.adapter_kwargs,
            )
            self._adapter.setup()
            logger.info("ASR adapter ready on worker ({})", self.adapter_target)

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
            self._adapter = None

    def num_workers(self) -> int | None:
        if (
            self.xenna_num_workers is None
            and self.xenna_num_workers_per_node is not None
            and not self._warned_ray_per_node_pin
        ):
            logger.warning(
                "ASRStage: xenna_num_workers_per_node={} is set but xenna_num_workers "
                "is None; Ray Data has no per-node pin and will AUTOSCALE this GPU "
                "stage. Set xenna_num_workers for a cluster-wide Ray Data pin.",
                self.xenna_num_workers_per_node,
            )
            self._warned_ray_per_node_pin = True
        return self.xenna_num_workers

    def xenna_stage_spec(self) -> dict[str, Any]:
        # Xenna reads the pin from here, not from num_workers().
        spec: dict[str, Any] = {}
        if self.xenna_num_workers is not None:
            spec["num_workers"] = self.xenna_num_workers
        if self.xenna_num_workers_per_node is not None:
            spec["num_workers_per_node"] = self.xenna_num_workers_per_node
        return spec

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        keys = [self.pred_text_key, self.skip_me_key]
        if self.disfluency_text_key:
            keys.append(self.disfluency_text_key)
        return [], keys

    def _resolve_language(self, task: AudioTask) -> str | None:
        code = task.data.get(self.source_lang_key) if self.source_lang_key else None
        if code:
            return _LANG_CODE_TO_NAME.get(code, code)
        if self.default_language:
            return _LANG_CODE_TO_NAME.get(self.default_language, self.default_language)
        return None

    @staticmethod
    def _chunk_waveform(
        waveform: np.ndarray,
        sample_rate: int,
        max_seconds: float,
    ) -> list[np.ndarray]:
        """Return contiguous ``<= max_seconds`` sub-chunks of ``waveform``.

        Last chunk may be shorter (no padding/overlap). Returns ``[waveform]``
        unchanged when it already fits (the common case for ``data_config_s3_8``).
        """
        if waveform is None or getattr(waveform, "size", 0) == 0 or not sample_rate or sample_rate <= 0:
            return [waveform]
        max_samples = int(max_seconds * sample_rate)
        if max_samples <= 0:
            return [waveform]
        n = int(waveform.shape[0])
        if n <= max_samples:
            return [waveform]
        chunks: list[np.ndarray] = []
        for start in range(0, n, max_samples):
            chunks.append(waveform[start : start + max_samples])
        return chunks

    def build_items(
        self,
        tasks: list[AudioTask],
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Pre-slice tasks into the flat per-sub-chunk item list + parent map.

        ``BucketedInferenceStage`` hook (runs first each call): validates inputs,
        resets per-call metric accumulators, then expands long clips into chunks.

        Returns:
            ``(items, parent_of)`` where ``parent_of[i]`` is the originating task
            index. Each item carries ``waveform`` (sub-chunk or full),
            ``sample_rate`` (unchanged), ``language`` (resolved name), ``task_id``,
            ``audio_seconds``, and ``chunk_idx`` / ``chunk_count``.
        """
        for task in tasks:
            if not self.validate_input(task):
                msg = (
                    f"Task {task.task_id} missing required columns for "
                    f"{type(self).__name__}: {self.inputs()}"
                )
                raise ValueError(msg)
        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        self._acc_model_metrics = defaultdict(float)
        self._inference_elapsed_s = 0.0

        items: list[dict[str, Any]] = []
        parent_of: list[int] = []
        slice_ceiling = float(self.max_inference_duration_s or self.ideal_inference_segment_s)
        for parent_idx, task in enumerate(tasks):
            waveform = task.data.get(self.waveform_key)
            sample_rate = task.data.get(self.sample_rate_key)
            language = self._resolve_language(task)
            # Placeholder item for empty/None waveform keeps the adapter's 1:1
            # item-per-parent invariant.
            if waveform is None or getattr(waveform, "size", 0) == 0 or not sample_rate:
                items.append({
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "language": language,
                    "task_id": task.task_id,
                    "audio_seconds": 0.0,
                    "chunk_idx": 0,
                    "chunk_count": 1,
                })
                parent_of.append(parent_idx)
                continue
            chunks = self._chunk_waveform(waveform, int(sample_rate), slice_ceiling)
            for chunk_idx, chunk in enumerate(chunks):
                items.append({
                    "waveform": chunk,
                    "sample_rate": int(sample_rate),
                    "language": language,
                    "task_id": task.task_id,
                    "audio_seconds": float(chunk.shape[0]) / float(sample_rate),
                    "chunk_idx": chunk_idx,
                    "chunk_count": len(chunks),
                })
                parent_of.append(parent_idx)
        return items, parent_of

    def _stitch(
        self,
        results: list[ASRResult],
        parent_of: list[int],
        num_parents: int,
    ) -> list[ASRResult]:
        """Join per-chunk text outputs per parent task with single spaces.

        Parent is marked skipped only if EVERY chunk was skipped; if any chunk
        succeeded, its non-empty texts are joined and the parent is not skipped.
        """
        per_parent_texts: list[list[str]] = [[] for _ in range(num_parents)]
        per_parent_secondary: list[list[str]] = [[] for _ in range(num_parents)]
        per_parent_skip_count: list[int] = [0] * num_parents
        per_parent_chunk_count: list[int] = [0] * num_parents
        per_parent_model_id: list[str] = [""] * num_parents

        for r, parent in zip(results, parent_of, strict=True):
            per_parent_chunk_count[parent] += 1
            if r.skipped:
                per_parent_skip_count[parent] += 1
            text = (r.text or "").strip()
            if text:
                per_parent_texts[parent].append(text)
            sec = (r.secondary_text or "").strip()
            if sec:
                per_parent_secondary[parent].append(sec)
            if r.model_id and not per_parent_model_id[parent]:
                per_parent_model_id[parent] = r.model_id

        stitched: list[ASRResult] = []
        for p in range(num_parents):
            all_skipped = per_parent_chunk_count[p] > 0 and per_parent_skip_count[p] == per_parent_chunk_count[p]
            stitched.append(
                ASRResult(
                    text=" ".join(per_parent_texts[p]),
                    secondary_text=" ".join(per_parent_secondary[p]) if per_parent_secondary[p] else None,
                    skipped=all_skipped,
                    model_id=per_parent_model_id[p],
                )
            )
        return stitched

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def item_cost(self, item: dict[str, Any]) -> float:
        """Bucketing cost of one sub-chunk: its audio duration in seconds."""
        return float(item.get("audio_seconds", 0.0))

    def run_inference(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe ONE bucket-respecting sub-batch via the adapter.

        Also folds the adapter's ``last_metrics`` and wall-clock time into the
        per-``process_batch`` accumulators that :meth:`assemble` reports.
        """
        inference_t0 = time.perf_counter()
        sub_results = self._adapter.transcribe_batch(items)
        self._inference_elapsed_s += time.perf_counter() - inference_t0
        last_m = dict(getattr(self._adapter, "last_metrics", {}) or {})
        for k, v in last_m.items():
            if isinstance(v, (int, float)):
                self._acc_model_metrics[k] += float(v)
        return sub_results

    def assemble(
        self,
        tasks: list[AudioTask],
        items: list[dict[str, Any]],
        parent_of: list[int],
        results: list[ASRResult],
    ) -> list[AudioTask]:
        """Stitch sub-chunk results per parent task, write outputs, emit metrics."""
        accumulated_model_metrics = self._acc_model_metrics
        inference_elapsed = self._inference_elapsed_s

        # Defensive: turn any None slots into skipped placeholders, guarding
        # against silent data loss if a future adapter forgets a slot.
        chunk_results: list[ASRResult] = [
            r if r is not None else ASRResult(text="", skipped=True) for r in results
        ]

        per_parent_results = self._stitch(chunk_results, parent_of, num_parents=len(tasks))

        skipped_count = 0
        for task, parent_result in zip(tasks, per_parent_results, strict=True):
            task.data[self.pred_text_key] = parent_result.text
            if self.disfluency_text_key:
                task.data[self.disfluency_text_key] = parent_result.secondary_text or ""
            if parent_result.skipped:
                task.data[self.skip_me_key] = "empty_audio"
                skipped_count += 1
            if not self.keep_waveform:
                task.data.pop(self.waveform_key, None)

        # ``utterances_*`` count PARENT tasks (the 1-row-per-input semantic
        # downstream consumers rely on); ``sub_chunks_generated`` surfaces the
        # pre-slicer fan-out.
        waveforms_for_metric = [it["waveform"] for it in items]
        sample_rates_for_metric = [it["sample_rate"] for it in items]
        metrics: dict[str, float] = {
            "utterances_input": float(len(tasks)),
            "utterances_processed": float(max(0, len(tasks) - skipped_count)),
            "utterances_skipped": float(skipped_count),
            "sub_chunks_generated": float(len(items)),
            "audio_duration_s": sum(
                float(w.shape[0]) / float(sr)
                for w, sr in zip(waveforms_for_metric, sample_rates_for_metric, strict=False)
                if sr and w is not None and getattr(w, "size", 0) > 0
            ),
            "waveform_bytes": sum(
                float(getattr(w, "nbytes", 0))
                for w in waveforms_for_metric
                if w is not None
            ),
            "output_chars": float(
                sum(len(r.text) for r in per_parent_results)
                + sum(len(r.secondary_text or "") for r in per_parent_results)
            ),
            "output_tokens": float(accumulated_model_metrics.get("output_tokens", 0.0)),
            "turn1_output_tokens": float(accumulated_model_metrics.get("turn1_output_tokens", 0.0)),
            "turn2_output_tokens": float(accumulated_model_metrics.get("turn2_output_tokens", 0.0)),
            "inference_time_s": inference_elapsed,
        }
        # Pass through adapter scalar metrics under a "model_<name>" alias,
        # skipping any that would restate a key the stage already emits.
        metrics.update({
            f"model_{name}": value
            for name, value in accumulated_model_metrics.items()
            if isinstance(value, (int, float)) and name not in metrics
        })
        self._log_metrics(metrics)

        if skipped_count:
            logger.info(
                f"ASRStage ({self.adapter_target}): marked {skipped_count}/{len(tasks)} "
                f"tasks as empty_audio ({self.skip_me_key})",
            )
        logger.info(
            f"ASRStage ({self.adapter_target}): generated {len(per_parent_results)} parent predictions "
            f"from {len(items)} sub-chunk(s) "
            f"(disfluency_text={'on' if self.disfluency_text_key else 'off'})",
        )
        return tasks
