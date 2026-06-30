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

Curator-side glue: validates I/O, resolves per-task language, segments long
audio into model-sized work units, applies duration-aware bucketing inside the
backend-provided ``process_batch`` call, stitches results per parent task, and
writes predictions/metrics. The concrete adapter is
resolved at runtime from ``adapter_target`` via ``hydra.utils.get_class``.
"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from math import isclose
from numbers import Real
from threading import Lock
from typing import TYPE_CHECKING, Any

import hydra.utils
from loguru import logger

from nemo_curator.models.asr.base import ASRAdapter, ASRResult
from nemo_curator.pipeline.payload_refs import (
    PayloadRef,
    resolve_payload_refs_batched,
)
from nemo_curator.pipeline.prefetch import BoundedOneAheadPrefetchIterator
from nemo_curator.stages.audio.model_input_segmentation import plan_audio_segments, resolve_max_model_input_duration
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.payload_lifecycle import PayloadAwareStageMixin
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, DispatchBatchTask

if TYPE_CHECKING:
    from collections.abc import Callable

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


@dataclass(frozen=True)
class _ChunkSpec:
    parent_task: AudioTask
    parent_idx: int
    chunk_idx: int
    chunk_count: int
    waveform: object | None
    sample_rate: object
    language: str | None
    language_code: str | None
    reference_text: str | None
    cost: float
    payload_ref: PayloadRef | None = None
    start_sample: int = 0
    stop_sample: int = 0


@dataclass(frozen=True)
class _InferenceCall:
    indices: list[int]
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class _PreparedDispatchBatch:
    tasks: list[AudioTask]
    items: list[dict[str, Any]]
    aligned_results: list[ASRResult | None]
    call: _InferenceCall | None


_PAYLOAD_REF_ITEM_KEY = "_curator_payload_ref"
_PAYLOAD_START_ITEM_KEY = "_curator_payload_start_sample"
_PAYLOAD_STOP_ITEM_KEY = "_curator_payload_stop_sample"
_WAVEFORM_BYTES_ITEM_KEY = "_curator_waveform_bytes"
_PRE_SKIPPED_ITEM_KEY = "_curator_asr_pre_skipped_reason"


def _payload_cache_key(payload_ref: PayloadRef) -> tuple[str | None, str, str]:
    return payload_ref.actor_namespace, payload_ref.store_actor_name, payload_ref.payload_id


class _PayloadCallMaterializer:
    """Resolve and retain only payloads needed by active adapter calls."""

    def __init__(
        self,
        *,
        cache_max_bytes: int,
        consumer_node_id: str,
        slice_waveform: Callable[[object, int, int], object],
    ) -> None:
        self._cache_max_bytes = int(cache_max_bytes)
        self._consumer_node_id = consumer_node_id
        self._slice_waveform = slice_waveform
        self._cache: OrderedDict[tuple[str | None, str, str], tuple[PayloadRef, object]] = OrderedDict()
        self._active: dict[tuple[str | None, str, str], int] = defaultdict(int)
        self._cache_bytes = 0
        self._lock = Lock()
        self.resolution_count = 0
        self.resolution_bytes = 0
        self.same_node_count = 0
        self.cross_node_count = 0
        self.resolution_time_s = 0.0

    def materialize(self, call: _InferenceCall) -> list[dict[str, Any]]:
        refs = self._unique_call_refs(call)
        with self._lock:
            missing = [ref for ref in refs if _payload_cache_key(ref) not in self._cache]

        if missing:
            started = time.perf_counter()
            payloads = resolve_payload_refs_batched(missing)
            elapsed = time.perf_counter() - started
            with self._lock:
                self.resolution_time_s += elapsed
                for ref, payload in zip(missing, payloads, strict=True):
                    key = _payload_cache_key(ref)
                    if key not in self._cache:
                        self._cache[key] = (ref, payload)
                        self._cache_bytes += max(0, int(ref.amount_bytes))
                        self.resolution_count += 1
                        self.resolution_bytes += max(0, int(ref.amount_bytes))
                        if ref.owner_node_id and ref.owner_node_id == self._consumer_node_id:
                            self.same_node_count += 1
                        else:
                            self.cross_node_count += 1

        with self._lock:
            for ref in refs:
                key = _payload_cache_key(ref)
                self._active[key] += 1
                self._cache.move_to_end(key)
            materialized = [self._materialize_item(item) for item in call.items]
            self._evict_inactive()
        return materialized

    def complete(self, call: _InferenceCall) -> None:
        refs = self._unique_call_refs(call)
        with self._lock:
            for ref in refs:
                key = _payload_cache_key(ref)
                self._active[key] = max(0, self._active.get(key, 0) - 1)
            self._evict_inactive()

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._active.clear()
            self._cache_bytes = 0

    def _materialize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        payload_ref = item.get(_PAYLOAD_REF_ITEM_KEY)
        if not isinstance(payload_ref, PayloadRef):
            return item
        _ref, waveform = self._cache[_payload_cache_key(payload_ref)]
        start = int(item.get(_PAYLOAD_START_ITEM_KEY, 0))
        stop = int(item.get(_PAYLOAD_STOP_ITEM_KEY, payload_ref.num_samples))
        materialized = dict(item)
        materialized["waveform"] = self._slice_waveform(waveform, start, stop)
        materialized.pop(_PAYLOAD_REF_ITEM_KEY, None)
        materialized.pop(_PAYLOAD_START_ITEM_KEY, None)
        materialized.pop(_PAYLOAD_STOP_ITEM_KEY, None)
        return materialized

    def _evict_inactive(self) -> None:
        while self._cache_bytes > self._cache_max_bytes:
            if len(self._cache) == 1:
                # One source payload may itself exceed the lookahead budget
                # (for example a multi-hour local parent). Keep that one
                # payload across its contiguous model calls, but never prefetch
                # another call beside it because the iterator's combined-byte
                # check will fail.
                return
            evictable = next((key for key in self._cache if self._active.get(key, 0) == 0), None)
            if evictable is None:
                return
            ref, _payload = self._cache.pop(evictable)
            self._cache_bytes -= max(0, int(ref.amount_bytes))
            self._active.pop(evictable, None)

    @staticmethod
    def _unique_call_refs(call: _InferenceCall) -> list[PayloadRef]:
        refs: dict[tuple[str | None, str, str], PayloadRef] = {}
        for item in call.items:
            payload_ref = item.get(_PAYLOAD_REF_ITEM_KEY)
            if isinstance(payload_ref, PayloadRef):
                refs.setdefault(_payload_cache_key(payload_ref), payload_ref)
        return list(refs.values())


@dataclass
class ASRStage(PayloadAwareStageMixin, ProcessingStage[AudioTask, AudioTask]):
    """Audio speech-recognition Curator stage with pluggable adapter.

    Resolves an ``ASRAdapter`` from ``adapter_target``, slices long audio into
    model-safe chunks, and stitches chunk outputs back to one result per input
    task. Duration-aware bucketing is controlled independently by
    ``batch_policy`` and packs already-created chunks.
    """

    _curator_accepts_dispatch_batches = True

    # Adapter selection.
    adapter_target: str
    model_id: str
    name: str = "ASR_inference"
    revision: str | None = None

    # Task I/O keys.
    waveform_key: str = "waveform"
    waveform_ref_key: str | None = "waveform_ref"
    sample_rate_key: str = "sample_rate"
    source_lang_key: str = "source_lang"
    reference_text_key: str | None = None
    default_language: str | None = None
    supported_language_codes: list[str] | None = None
    pred_text_key: str = "pred_text"
    disfluency_text_key: str | None = None
    skip_me_key: str = "_skip_me"

    # Model-input segmentation and output retention. Long-row model safety is
    # derived from max_inference_duration_s.
    max_inference_duration_s: float = 2400.0
    keep_waveform: bool = True

    prefetch_fail_on_error: bool = True

    # Optional payload-resolution optimization. Defaults preserve the eager
    # behavior used by existing pipelines; benchmark configs opt into bounded
    # one-call lookahead explicitly.
    payload_prefetch_enabled: bool = False
    payload_prefetch_max_bytes: int | None = None

    # Worker placement.
    xenna_num_workers: int | None = None
    xenna_num_workers_per_node: int | None = None

    batch_policy: BatchPolicy | None = None

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    # Backend-visible candidate window. Ray Data/Xenna use this to decide how
    # many rows reach one process_batch() call; the final adapter call size is
    # controlled separately by adapter_batch_size / batch_policy.
    batch_size: int = 32
    adapter_batch_size: int | None = None

    def __post_init__(self) -> None:
        self.max_inference_duration_s = resolve_max_model_input_duration(
            max_duration_s=self.max_inference_duration_s,
            owner="ASRStage",
        )
        if int(self.batch_size) <= 0:
            msg = f"ASRStage.batch_size must be > 0, got {self.batch_size}"
            raise ValueError(msg)
        self.batch_size = int(self.batch_size)
        if self.adapter_batch_size is not None:
            if int(self.adapter_batch_size) <= 0:
                msg = f"ASRStage.adapter_batch_size must be > 0, got {self.adapter_batch_size}"
                raise ValueError(msg)
            self.adapter_batch_size = int(self.adapter_batch_size)
        self._validate_payload_resolution_options()
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
        self._adapter_inference_calls: int = 0

    def _validate_payload_resolution_options(self) -> None:
        if not isinstance(self.payload_prefetch_enabled, bool):
            msg = "ASRStage.payload_prefetch_enabled must be a bool"
            raise TypeError(msg)
        if self.payload_prefetch_max_bytes is not None:
            if isinstance(self.payload_prefetch_max_bytes, bool) or int(self.payload_prefetch_max_bytes) <= 0:
                msg = "ASRStage.payload_prefetch_max_bytes must be > 0 when set"
                raise ValueError(msg)
            self.payload_prefetch_max_bytes = int(self.payload_prefetch_max_bytes)
        if self.payload_prefetch_enabled and self.payload_prefetch_max_bytes is None:
            msg = "ASRStage.payload_prefetch_max_bytes is required when payload_prefetch_enabled=True"
            raise ValueError(msg)
        self._adapter_inference_items: int = 0
        self._warned_ray_per_node_pin = False
        self._supported_language_codes = self._normalise_supported_language_codes(self.supported_language_codes)

    def validate_dispatch_source(self, source: object) -> None:
        """Fail during graph expansion if an upstream plan cannot be dispatched unchanged."""
        policy = self.batch_policy
        if policy is None or not policy.enabled:
            msg = f"Dispatch-batch owner {self.name!r} requires batch_policy.enabled=true"
            raise ValueError(msg)
        source_signature = getattr(source, "dispatch_policy_signature", None)
        if not callable(source_signature):
            msg = f"Dispatch-batch source {type(source).__name__} does not expose a policy signature"
            raise TypeError(msg)
        expected = policy.dispatch_signature(cost_unit="audio_seconds")
        if source_signature() != expected:
            msg = (
                f"Dispatch-batch source constraints do not match owner {self.name!r} batch_policy; "
                "both must use the same buckets, item caps, and total-cost cap"
            )
            raise ValueError(msg)

    @staticmethod
    def _normalise_supported_language_codes(value: object) -> set[str] | None:
        """Normalize an optional adapter-specific supported-language allowlist."""
        if value is None:
            return None
        raw_codes = value.split(",") if isinstance(value, str) else list(value)  # type: ignore[arg-type]
        codes = {str(code).strip().lower() for code in raw_codes if str(code).strip()}
        return codes or None

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

    def setup_on_node_resources(self) -> Resources:
        return Resources(cpus=1.0, gpus=0.0)

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
        spec: dict[str, Any] = {}
        if self.xenna_num_workers_per_node is not None:
            spec["num_workers_per_node"] = self.xenna_num_workers_per_node
        return spec

    def inputs(self) -> tuple[list[str], list[str]]:
        waveform_input = self.waveform_ref_key or self.waveform_key
        optional_inputs = [waveform_input, self.sample_rate_key]
        if self.reference_text_key:
            optional_inputs.append(self.reference_text_key)
        return [], optional_inputs

    def outputs(self) -> tuple[list[str], list[str]]:
        keys = [self.pred_text_key, self.skip_me_key]
        if self.disfluency_text_key:
            keys.append(self.disfluency_text_key)
        return [], keys

    def _validate_asr_task_input(self, task: AudioTask) -> bool:
        if self._pre_skipped_reason(task) is not None:
            return True
        has_waveform = self.waveform_key in task.data
        has_ref = bool(self.waveform_ref_key and self.waveform_ref_key in task.data)
        if not has_waveform and not has_ref:
            logger.error(
                "Task {} missing ASR waveform input: expected '{}' or '{}'",
                task.task_id,
                self.waveform_key,
                self.waveform_ref_key,
            )
            return False
        if self.sample_rate_key not in task.data:
            logger.error("Task {} missing ASR sample-rate input '{}'", task.task_id, self.sample_rate_key)
            return False
        return True

    def _pre_skipped_reason(self, task: AudioTask) -> str | None:
        reason = task.data.get(self.skip_me_key)
        return str(reason) if reason else None

    @staticmethod
    def _pre_skipped_item_result(item: dict[str, Any]) -> ASRResult | None:
        reason = item.get(_PRE_SKIPPED_ITEM_KEY)
        if not reason:
            return None
        return ASRResult(text="", skipped=True, extras={"skip_reason": str(reason)})

    def _resolve_payload_refs(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return self.resolve_payload_refs_for_batch(tasks)

    def _drop_resolved_payload_waveforms(self, tasks: list[AudioTask]) -> None:
        self.drop_resolved_payloads(tasks)

    def _resolve_language(self, task: AudioTask) -> str | None:
        code = self._resolve_language_code(task)
        if code:
            return _LANG_CODE_TO_NAME.get(code, code)
        return None

    def _resolve_language_code(self, task: AudioTask) -> str | None:
        code = task.data.get(self.source_lang_key) if self.source_lang_key else None
        if code:
            return str(code).strip().lower()
        if self.default_language:
            return str(self.default_language).strip().lower()
        return None

    def _is_language_supported(self, item: dict[str, Any]) -> bool:
        if self._supported_language_codes is None:
            return True
        code = str(item.get("language_code", "") or "").strip().lower()
        return bool(code) and code in self._supported_language_codes

    def _resolve_reference_text(self, task: AudioTask) -> str | None:
        if not self.reference_text_key:
            return None
        value = task.data.get(self.reference_text_key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _waveform_num_samples(waveform: object) -> int:
        shape = getattr(waveform, "shape", None)
        if shape:
            return int(shape[-1])
        try:
            return len(waveform)  # type: ignore[arg-type]
        except TypeError:
            return 0

    @classmethod
    def _waveform_is_empty(cls, waveform: object) -> bool:
        return waveform is None or cls._waveform_num_samples(waveform) <= 0

    @staticmethod
    def _slice_waveform(waveform: object, start: int, stop: int) -> object:
        try:
            return waveform[..., start:stop]  # type: ignore[index]
        except TypeError:
            return waveform[start:stop]  # type: ignore[index]

    @staticmethod
    def _waveform_nbytes(waveform: object) -> float:
        nbytes = getattr(waveform, "nbytes", None)
        if isinstance(nbytes, int):
            return float(nbytes)
        element_size = getattr(waveform, "element_size", None)
        nelement = getattr(waveform, "nelement", None)
        if callable(element_size) and callable(nelement):
            return float(element_size() * nelement())
        return 0.0

    @classmethod
    def _chunk_waveform(
        cls,
        waveform: object,
        sample_rate: int,
        max_seconds: float,
    ) -> list[object]:
        """Return contiguous ``<= max_seconds`` sub-chunks of ``waveform``.

        Last chunk may be shorter (no padding/overlap). Returns ``[waveform]``
        unchanged when it already fits (the common case for ``data_config_s3_8``).
        """
        if cls._waveform_is_empty(waveform) or not sample_rate or sample_rate <= 0:
            return [waveform]
        n = cls._waveform_num_samples(waveform)
        segments = plan_audio_segments(
            num_samples=n,
            sample_rate=sample_rate,
            max_duration_s=max_seconds,
            owner="ASRStage",
        )
        return [cls._slice_waveform(waveform, segment.start_sample, segment.stop_sample) for segment in segments]

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
        per_parent_skip_reason: list[str | None] = [None] * num_parents

        for r, parent in zip(results, parent_of, strict=True):
            per_parent_chunk_count[parent] += 1
            if r.skipped:
                per_parent_skip_count[parent] += 1
                if per_parent_skip_reason[parent] is None:
                    reason = r.extras.get("skip_reason")
                    if reason:
                        per_parent_skip_reason[parent] = str(reason)
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
                    extras={"skip_reason": per_parent_skip_reason[p]} if per_parent_skip_reason[p] else {},
                )
            )
        return stitched

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(
        self,
        tasks: list[AudioTask | DispatchBatchTask],
    ) -> list[AudioTask | DispatchBatchTask]:
        """Run one backend-provided ASR batch.

        Backend executors own how many parent rows reach this call. Model-input
        segmentation and duration-aware bucketing stay inside the stage: parent
        rows are expanded into bounded model-input items here, bucketed when
        ``batch_policy`` is enabled, and stitched back into the original parent
        order before returning.
        """
        if len(tasks) == 0:
            return []
        dispatch_flags = [isinstance(task, DispatchBatchTask) for task in tasks]
        if any(dispatch_flags):
            if not all(dispatch_flags):
                msg = "ASRStage received a mixed batch of DispatchBatchTask and AudioTask rows"
                raise TypeError(msg)
            dispatch_batches = [task for task in tasks if isinstance(task, DispatchBatchTask)]
            child_tasks = self._dispatch_child_tasks(dispatch_batches)
            if self.payload_prefetch_enabled and self._has_unresolved_payload_refs(child_tasks):
                return self._process_dispatch_batches(dispatch_batches)
            inserted_waveforms: list[AudioTask] = []
            try:
                inserted_waveforms = self._resolve_payload_refs(child_tasks)
                return self._process_dispatch_batches(dispatch_batches)
            finally:
                self._drop_resolved_payload_waveforms(inserted_waveforms)
        if self.payload_prefetch_enabled and self._has_unresolved_payload_refs(tasks):
            return self._process_plain_batch(tasks)
        inserted_waveforms: list[AudioTask] = []
        try:
            inserted_waveforms = self._resolve_payload_refs(tasks)
            return self._process_plain_batch(tasks)
        finally:
            self._drop_resolved_payload_waveforms(inserted_waveforms)

    @staticmethod
    def _dispatch_child_tasks(dispatch_batches: list[DispatchBatchTask]) -> list[AudioTask]:
        children: list[AudioTask] = []
        for batch in dispatch_batches:
            if not all(isinstance(item, AudioTask) for item in batch.items):
                msg = f"ASRStage dispatch batch {batch.batch_id!r} contains non-audio tasks"
                raise TypeError(msg)
            children.extend(item for item in batch.items if isinstance(item, AudioTask))
        return children

    def _has_unresolved_payload_refs(self, tasks: list[AudioTask]) -> bool:
        if not self.waveform_ref_key:
            return False
        return any(
            self.waveform_key not in task.data and isinstance(task.data.get(self.waveform_ref_key), PayloadRef)
            for task in tasks
        )

    def _process_plain_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Dispatch one unbucketed backend batch.

        The backend's normal batch stays intact, but each long parent is sliced
        into bounded model-input chunks before adapter inference and stitched
        back afterward. If a parent already fits the model-input window, it
        remains one adapter item.
        """
        for task in tasks:
            if not self._validate_asr_task_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        self._acc_model_metrics = defaultdict(float)
        self._inference_elapsed_s = 0.0
        self._adapter_inference_calls = 0
        self._adapter_inference_items = 0

        chunk_specs = self._build_chunk_specs(tasks)
        items = [self._chunk_spec_to_item(spec) for spec in chunk_specs]
        parent_of = [spec.parent_idx for spec in chunk_specs]

        if any(isinstance(item.get(_PAYLOAD_REF_ITEM_KEY), PayloadRef) for item in items):
            results = self._run_payload_inference_capped(items)
        else:
            results = self._run_inference_capped(items)
        if len(results) != len(items):
            msg = f"run_fn returned {len(results)} results for {len(items)} items (must match 1:1)"
            raise RuntimeError(msg)
        return self.assemble(tasks, items, parent_of, results)

    def _process_dispatch_batches(self, dispatch_batches: list[DispatchBatchTask]) -> list[DispatchBatchTask]:
        """Execute every planner-owned envelope as exactly one adapter call."""
        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        self._acc_model_metrics = defaultdict(float)
        self._inference_elapsed_s = 0.0
        self._adapter_inference_calls = 0
        self._adapter_inference_items = 0

        flat_tasks: list[AudioTask] = []
        all_items: list[dict[str, Any]] = []
        parent_of: list[int] = []
        aligned_results: list[ASRResult | None] = []
        calls: list[_InferenceCall] = []
        batch_sizes: list[int] = []

        for batch in dispatch_batches:
            task_offset = len(flat_tasks)
            prepared = self._prepare_dispatch_batch(
                batch,
                item_offset=len(all_items),
            )
            all_items.extend(prepared.items)
            flat_tasks.extend(prepared.tasks)
            parent_of.extend(range(task_offset, task_offset + len(prepared.tasks)))
            aligned_results.extend(prepared.aligned_results)
            batch_sizes.append(len(prepared.tasks))
            if prepared.call is not None:
                calls.append(prepared.call)

        has_payload_refs = any(isinstance(item.get(_PAYLOAD_REF_ITEM_KEY), PayloadRef) for item in all_items)
        if has_payload_refs:
            results = self._run_payload_inference_calls(aligned_results, calls)
        else:
            results = self._run_inference_calls(aligned_results, calls)
        processed_tasks = self.assemble(flat_tasks, all_items, parent_of, results)
        self._log_metrics(
            {
                "dispatch_batches_input": float(len(dispatch_batches)),
                "dispatch_batches_executed": float(len(calls)),
                "dispatch_batch_items": float(len(all_items)),
            }
        )

        rebuilt: list[DispatchBatchTask] = []
        cursor = 0
        for batch, batch_size in zip(dispatch_batches, batch_sizes, strict=True):
            rebuilt.append(batch.with_items(processed_tasks[cursor : cursor + batch_size]))
            cursor += batch_size
        return rebuilt

    def _prepare_dispatch_batch(
        self,
        batch: DispatchBatchTask,
        *,
        item_offset: int,
    ) -> _PreparedDispatchBatch:
        child_tasks = self._dispatch_child_tasks([batch])
        for task in child_tasks:
            if not self._validate_asr_task_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)

        chunk_specs = self._build_chunk_specs(child_tasks)
        if len(chunk_specs) != len(child_tasks) or any(spec.chunk_count != 1 for spec in chunk_specs):
            msg = (
                f"Dispatch batch {batch.batch_id!r} was planned as model-ready, but ASR segmentation "
                "would change its item boundaries"
            )
            raise RuntimeError(msg)
        items = [self._chunk_spec_to_item(spec) for spec in chunk_specs]
        self._validate_dispatch_batch(batch, items)
        results, call = self._dispatch_call(items, item_offset=item_offset)
        if any(spec.parent_idx != index for index, spec in enumerate(chunk_specs)):
            msg = f"Dispatch batch {batch.batch_id!r} lost one-to-one child alignment"
            raise RuntimeError(msg)
        return _PreparedDispatchBatch(tasks=child_tasks, items=items, aligned_results=results, call=call)

    def _dispatch_call(
        self,
        items: list[dict[str, Any]],
        *,
        item_offset: int,
    ) -> tuple[list[ASRResult | None], _InferenceCall | None]:
        aligned_results: list[ASRResult | None] = [None] * len(items)
        eligible_indices: list[int] = []
        eligible_items: list[dict[str, Any]] = []
        for local_index, item in enumerate(items):
            pre_skipped_result = self._pre_skipped_item_result(item)
            if pre_skipped_result is not None:
                aligned_results[local_index] = pre_skipped_result
                continue
            if self._is_language_supported(item):
                eligible_indices.append(item_offset + local_index)
                eligible_items.append(item)
                continue
            code = str(item.get("language_code", "") or "").strip().lower()
            aligned_results[local_index] = ASRResult(
                text="",
                skipped=True,
                extras={"skip_reason": f"lang_not_supported:{code or 'unknown'}"},
            )
        call = _InferenceCall(indices=eligible_indices, items=eligible_items) if eligible_items else None
        return aligned_results, call

    def _validate_dispatch_batch(self, batch: DispatchBatchTask, items: list[dict[str, Any]]) -> None:
        policy = self._dispatch_policy(batch)
        if len(items) != len(batch.item_costs):
            msg = f"Dispatch batch {batch.batch_id!r} item count changed before owner inference"
            raise ValueError(msg)

        observed_costs = [float(item.get("audio_seconds", 0.0)) for item in items]
        self._validate_dispatch_cost_contract(batch, observed_costs)
        self._validate_dispatch_caps(batch, policy, observed_costs)

    def _dispatch_policy(self, batch: DispatchBatchTask) -> BatchPolicy:
        owner_idents = {
            str(value)
            for value in (
                getattr(self, "_curator_stage_id", None),
                self.name,
                type(self).__name__,
                f"{type(self).__module__}.{type(self).__name__}",
            )
            if value
        }
        if batch.owner_stage not in owner_idents:
            msg = (
                f"Dispatch batch {batch.batch_id!r} belongs to owner {batch.owner_stage!r}, "
                f"not ASR stage {sorted(owner_idents)}"
            )
            raise ValueError(msg)
        policy = self.batch_policy
        if policy is None or not policy.enabled:
            msg = "ASRStage requires an enabled BatchPolicy to validate upstream dispatch batches"
            raise ValueError(msg)
        if batch.cost_unit != "audio_seconds":
            msg = f"ASRStage expected audio_seconds dispatch cost, got {batch.cost_unit!r}"
            raise ValueError(msg)
        expected_signature = policy.dispatch_signature(cost_unit=batch.cost_unit)
        if batch.policy_signature != expected_signature:
            msg = f"Dispatch batch {batch.batch_id!r} policy constraints do not match the ASR owner policy"
            raise ValueError(msg)
        return policy

    @staticmethod
    def _validate_dispatch_cost_contract(
        batch: DispatchBatchTask,
        observed_costs: list[float],
    ) -> None:
        if any(cost < 0 for cost in observed_costs):
            msg = f"Dispatch batch {batch.batch_id!r} contains a negative observed cost"
            raise ValueError(msg)
        planned_total = sum(float(cost) for cost in batch.item_costs)
        if not isclose(batch.total_cost, planned_total, rel_tol=1e-7, abs_tol=1e-3):
            msg = f"Dispatch batch {batch.batch_id!r} has inconsistent planned item and total costs"
            raise ValueError(msg)

    @staticmethod
    def _validate_dispatch_caps(
        batch: DispatchBatchTask,
        policy: BatchPolicy,
        observed_costs: list[float],
    ) -> None:
        observed_buckets = {policy.bucket_for(cost) for cost in observed_costs}
        applicable_buckets = {batch.bucket_index, *observed_buckets}
        item_cap = min(policy.max_items_per_batch_by_bucket[index] for index in applicable_buckets)
        if len(observed_costs) > item_cap:
            msg = f"Dispatch batch {batch.batch_id!r} has {len(observed_costs)} items, above owner cap {item_cap}"
            raise ValueError(msg)
        observed_total = sum(observed_costs)
        if policy.max_audio_sec_per_batch is not None and observed_total > policy.max_audio_sec_per_batch + 1e-3:
            msg = (
                f"Dispatch batch {batch.batch_id!r} costs {observed_total:g} {batch.cost_unit}, "
                f"above owner cap {policy.max_audio_sec_per_batch:g}"
            )
            raise ValueError(msg)

    def _run_inference_capped(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run adapter calls with policy batches as final model-call boundaries.

        When segmentation fans one backend batch out into many model work units,
        bucket-on uses ``BatchPolicy`` to create final adapter calls bounded by
        same-bucket membership, per-bucket item cap, and total cost. Bucket-off
        keeps the normal ``adapter_batch_size`` / ``batch_size`` fallback.
        Results are realigned to the original item order.
        """
        aligned_results, calls = self._plan_inference_calls(items)
        return self._run_inference_calls(aligned_results, calls)

    def _run_inference_calls(
        self,
        aligned_results: list[ASRResult | None],
        calls: list[_InferenceCall],
    ) -> list[ASRResult]:
        for call in calls:
            self._store_call_results(call, self.run_inference(call.items), aligned_results)
        return self._finalize_aligned_results(aligned_results)

    def _run_payload_inference_capped(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Resolve only the current adapter call and prefetch one successor."""
        if self.payload_prefetch_max_bytes is None:
            msg = "payload_prefetch_max_bytes is required for payload-prefetched inference"
            raise RuntimeError(msg)
        aligned_results, calls = self._plan_inference_calls(items)
        return self._run_payload_inference_calls(aligned_results, calls)

    def _run_payload_inference_calls(
        self,
        aligned_results: list[ASRResult | None],
        calls: list[_InferenceCall],
    ) -> list[ASRResult]:
        if self.payload_prefetch_max_bytes is None:
            msg = "payload_prefetch_max_bytes is required for payload-prefetched inference"
            raise RuntimeError(msg)
        materializer = _PayloadCallMaterializer(
            cache_max_bytes=self.payload_prefetch_max_bytes,
            consumer_node_id=self.payload_consumer_node_id(),
            slice_waveform=self._slice_waveform,
        )
        try:
            prefetched_calls = BoundedOneAheadPrefetchIterator(
                calls,
                loader=materializer.materialize,
                size_bytes=self._inference_call_payload_bytes,
                max_inflight_bytes=self.payload_prefetch_max_bytes,
                thread_name_prefix="curator-asr-payload-prefetch",
            )
            for call, materialized_items in prefetched_calls:
                try:
                    self._store_call_results(call, self.run_inference(materialized_items), aligned_results)
                finally:
                    materializer.complete(call)
        finally:
            self._log_payload_resolution_metrics(materializer)
            materializer.close()
        return self._finalize_aligned_results(aligned_results)

    def _plan_inference_calls(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[ASRResult | None], list[_InferenceCall]]:
        """Build exact adapter-call boundaries without resolving payload bytes."""
        aligned_results: list[ASRResult | None] = [None] * len(items)
        eligible_items: list[dict[str, Any]] = []
        eligible_indices: list[int] = []
        for idx, item in enumerate(items):
            pre_skipped_result = self._pre_skipped_item_result(item)
            if pre_skipped_result is not None:
                aligned_results[idx] = pre_skipped_result
                continue
            if self._is_language_supported(item):
                eligible_items.append(item)
                eligible_indices.append(idx)
                continue
            code = str(item.get("language_code", "") or "").strip().lower()
            aligned_results[idx] = ASRResult(
                text="",
                skipped=True,
                extras={"skip_reason": f"lang_not_supported:{code or 'unknown'}"},
            )

        calls: list[_InferenceCall] = []
        policy = self.batch_policy
        if policy is None or not policy.enabled:
            cursor = 0
            for sub_items in self._split_items_by_adapter_cap(eligible_items):
                sub_indices = eligible_indices[cursor : cursor + len(sub_items)]
                calls.append(_InferenceCall(indices=sub_indices, items=sub_items))
                cursor += len(sub_items)
            return aligned_results, calls

        for bucket_indices, bucket_items, _total_cost in policy.bucketize_with_costs(
            eligible_items,
            cost_fn=self.item_cost,
        ):
            calls.append(
                _InferenceCall(
                    indices=[eligible_indices[index] for index in bucket_indices],
                    items=bucket_items,
                )
            )
        return aligned_results, calls

    @staticmethod
    def _store_call_results(
        call: _InferenceCall,
        results: list[ASRResult],
        aligned_results: list[ASRResult | None],
    ) -> None:
        if len(results) != len(call.items):
            msg = f"run_fn returned {len(results)} results for {len(call.items)} items (must match 1:1)"
            raise RuntimeError(msg)
        for index, result in zip(call.indices, results, strict=True):
            aligned_results[index] = result

    @staticmethod
    def _finalize_aligned_results(aligned_results: list[ASRResult | None]) -> list[ASRResult]:
        if any(result is None for result in aligned_results):
            msg = "ASR call planning did not produce an inference result for every item"
            raise RuntimeError(msg)
        return [result for result in aligned_results if result is not None]

    @staticmethod
    def _inference_call_payload_bytes(call: _InferenceCall) -> int:
        refs: dict[tuple[str | None, str, str], PayloadRef] = {}
        for item in call.items:
            payload_ref = item.get(_PAYLOAD_REF_ITEM_KEY)
            if isinstance(payload_ref, PayloadRef):
                refs.setdefault(_payload_cache_key(payload_ref), payload_ref)
        return sum(max(0, int(ref.amount_bytes)) for ref in refs.values())

    def _log_payload_resolution_metrics(self, materializer: _PayloadCallMaterializer) -> None:
        if materializer.resolution_count <= 0:
            return
        self._log_metrics(
            {
                "payload_resolution_count": float(materializer.resolution_count),
                "payload_resolution_same_node_count": float(materializer.same_node_count),
                "payload_resolution_cross_node_count": float(materializer.cross_node_count),
                "payload_resolution_bytes": float(materializer.resolution_bytes),
                "payload_resolution_time_s": float(materializer.resolution_time_s),
            }
        )

    def _split_items_by_adapter_cap(self, items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split bucket-off work by the stage's fallback adapter-call item cap."""
        fallback_source = self.adapter_batch_size if self.adapter_batch_size is not None else self.batch_size
        cap = max(1, int(fallback_source or 1))
        batches: list[list[dict[str, Any]]] = []
        for start in range(0, len(items), cap):
            batches.append(items[start : start + cap])
        return batches

    def _build_chunk_specs(self, tasks: list[AudioTask]) -> list[_ChunkSpec]:
        """Build model-input descriptors from waveforms or payload metadata."""
        specs: list[_ChunkSpec] = []
        slice_ceiling = float(self.max_inference_duration_s)
        for parent_idx, task in enumerate(tasks):
            waveform = task.data.get(self.waveform_key)
            sample_rate = task.data.get(self.sample_rate_key)
            payload_ref = task.data.get(self.waveform_ref_key) if self.waveform_ref_key else None
            language = self._resolve_language(task)
            language_code = self._resolve_language_code(task)
            reference_text = self._resolve_reference_text(task)
            if self._waveform_is_empty(waveform) and isinstance(payload_ref, PayloadRef) and sample_rate:
                sr = int(sample_rate)
                segments = plan_audio_segments(
                    num_samples=int(payload_ref.num_samples),
                    sample_rate=sr,
                    max_duration_s=slice_ceiling,
                    owner="ASRStage",
                )
                for segment in segments:
                    specs.append(
                        _ChunkSpec(
                            parent_task=task,
                            parent_idx=parent_idx,
                            chunk_idx=segment.index,
                            chunk_count=segment.count,
                            waveform=None,
                            sample_rate=sr,
                            language=language,
                            language_code=language_code,
                            reference_text=reference_text,
                            cost=segment.duration_s,
                            payload_ref=payload_ref,
                            start_sample=segment.start_sample,
                            stop_sample=segment.stop_sample,
                        )
                    )
                continue
            if self._waveform_is_empty(waveform) or not sample_rate:
                specs.append(
                    _ChunkSpec(
                        parent_task=task,
                        parent_idx=parent_idx,
                        chunk_idx=0,
                        chunk_count=1,
                        waveform=waveform,
                        sample_rate=sample_rate,
                        language=language,
                        language_code=language_code,
                        reference_text=reference_text,
                        cost=0.0,
                        stop_sample=0,
                    )
                )
                continue

            sr = int(sample_rate)
            chunks = self._chunk_waveform(waveform, sr, slice_ceiling)
            chunk_count = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                specs.append(
                    _ChunkSpec(
                        parent_task=task,
                        parent_idx=parent_idx,
                        chunk_idx=chunk_idx,
                        chunk_count=chunk_count,
                        waveform=chunk,
                        sample_rate=sr,
                        language=language,
                        language_code=language_code,
                        reference_text=reference_text,
                        cost=0.0 if sr <= 0 else float(self._waveform_num_samples(chunk)) / float(sr),
                        start_sample=0,
                        stop_sample=self._waveform_num_samples(chunk),
                    )
                )
        return specs

    def _chunk_spec_to_item(self, spec: _ChunkSpec) -> dict[str, Any]:
        """Convert a virtual chunk descriptor into one adapter input item."""
        item = {
            "waveform": spec.waveform,
            "sample_rate": spec.sample_rate,
            "language": spec.language,
            "language_code": spec.language_code,
            "reference_text": spec.reference_text,
            "task_id": spec.parent_task.task_id,
            "audio_seconds": spec.cost,
            "chunk_idx": spec.chunk_idx,
            "chunk_count": spec.chunk_count,
            _WAVEFORM_BYTES_ITEM_KEY: self._chunk_spec_waveform_bytes(spec),
        }
        if spec.payload_ref is not None:
            item[_PAYLOAD_REF_ITEM_KEY] = spec.payload_ref
            item[_PAYLOAD_START_ITEM_KEY] = spec.start_sample
            item[_PAYLOAD_STOP_ITEM_KEY] = spec.stop_sample
        skip_reason = self._pre_skipped_reason(spec.parent_task)
        if skip_reason is not None:
            item[_PRE_SKIPPED_ITEM_KEY] = skip_reason
        return item

    @classmethod
    def _chunk_spec_waveform_bytes(cls, spec: _ChunkSpec) -> float:
        if spec.payload_ref is None:
            return cls._waveform_nbytes(spec.waveform)
        total_samples = max(1, int(spec.payload_ref.num_samples))
        segment_samples = max(0, int(spec.stop_sample) - int(spec.start_sample))
        return float(int(spec.payload_ref.amount_bytes) * segment_samples) / float(total_samples)

    def item_cost(self, item: dict[str, Any]) -> float:
        """Bucketing cost of one sub-chunk.

        Duration remains the default cost unit, but adapters may provide a
        better estimator for scheduler pressure (for example encoder tokens or
        approximate VRAM units) without changing executor autoscaling.
        """
        estimator = getattr(self._adapter, "estimate_item_cost", None)
        if callable(estimator):
            try:
                estimated = estimator(item)
                if isinstance(estimated, Real):
                    return max(0.0, float(estimated))
            except Exception as exc:  # noqa: BLE001
                logger.debug("ASR adapter cost estimator failed; falling back to duration cost: {}", exc)
        for key in ("estimated_vram_units", "estimated_encoder_tokens"):
            value = item.get(key)
            if value is not None:
                return max(0.0, float(value))
        return float(item.get("audio_seconds", 0.0))

    def run_inference(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe ONE bucket-respecting sub-batch via the adapter.

        Also folds the adapter's ``last_metrics`` and wall-clock time into the
        per-``process_batch`` accumulators that :meth:`assemble` reports.
        """
        inference_t0 = time.perf_counter()
        sub_results = self._adapter.transcribe_batch(items)
        self._inference_elapsed_s += time.perf_counter() - inference_t0
        self._adapter_inference_calls += 1
        self._adapter_inference_items += len(items)
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
        chunk_results: list[ASRResult] = [r if r is not None else ASRResult(text="", skipped=True) for r in results]

        per_parent_results = self._stitch(chunk_results, parent_of, num_parents=len(tasks))

        skipped_count = 0
        for task, parent_result in zip(tasks, per_parent_results, strict=True):
            task.data[self.pred_text_key] = parent_result.text
            if self.disfluency_text_key:
                task.data[self.disfluency_text_key] = parent_result.secondary_text or ""
            if parent_result.skipped:
                task.data[self.skip_me_key] = str(parent_result.extras.get("skip_reason") or "empty_audio")
                skipped_count += 1
            if not self.keep_waveform:
                task.data.pop(self.waveform_key, None)

        # ``utterances_*`` count PARENT tasks (the 1-row-per-input semantic
        # downstream consumers rely on); ``sub_chunks_generated`` surfaces the
        # pre-slicer fan-out.
        metrics: dict[str, float] = {
            "utterances_input": float(len(tasks)),
            "utterances_processed": float(max(0, len(tasks) - skipped_count)),
            "utterances_skipped": float(skipped_count),
            "sub_chunks_generated": float(len(items)),
            "audio_duration_s": sum(float(item.get("audio_seconds", 0.0)) for item in items),
            "waveform_bytes": sum(
                float(item.get(_WAVEFORM_BYTES_ITEM_KEY, self._waveform_nbytes(item.get("waveform"))))
                for item in items
            ),
            "output_chars": float(
                sum(len(r.text) for r in per_parent_results)
                + sum(len(r.secondary_text or "") for r in per_parent_results)
            ),
            "output_tokens": float(accumulated_model_metrics.get("output_tokens", 0.0)),
            "turn1_output_tokens": float(accumulated_model_metrics.get("turn1_output_tokens", 0.0)),
            "turn2_output_tokens": float(accumulated_model_metrics.get("turn2_output_tokens", 0.0)),
            "inference_time_s": inference_elapsed,
            "adapter_inference_calls": float(self._adapter_inference_calls),
            "adapter_inference_items": float(self._adapter_inference_items),
        }
        # Pass through adapter scalar metrics under a "model_<name>" alias,
        # skipping any that would restate a key the stage already emits.
        metrics.update(
            {
                f"model_{name}": value
                for name, value in accumulated_model_metrics.items()
                if isinstance(value, (int, float)) and name not in metrics
            }
        )
        self._log_metrics(metrics)

        if skipped_count:
            logger.info(
                f"ASRStage ({self.adapter_target}): marked {skipped_count}/{len(tasks)} "
                f"tasks as empty_audio ({self.skip_me_key})",
            )
        logger.debug(
            f"ASRStage ({self.adapter_target}): generated {len(per_parent_results)} parent predictions "
            f"from {len(items)} sub-chunk(s) "
            f"(disfluency_text={'on' if self.disfluency_text_key else 'off'})",
        )
        return tasks
