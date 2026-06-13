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

Curator-side glue: validates I/O, resolves per-task language, optionally
pre-slices long audio into scheduler work units when duration-aware bucketing is
enabled, dispatches ``transcribe_batch`` on duration-coherent chunk groups,
stitches results per parent task, and writes predictions/metrics. The concrete
adapter is resolved at runtime from ``adapter_target`` via
``hydra.utils.get_class``.
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
class _PrebucketChunk:
    task: AudioTask
    parent_idx: int
    chunk_idx: int
    chunk_count: int
    cost: float


@dataclass(frozen=True)
class _ChunkSpec:
    parent_task: AudioTask
    parent_idx: int
    chunk_idx: int
    chunk_count: int
    waveform: object
    sample_rate: object
    language: str | None
    cost: float


@dataclass
class ASRStage(BucketedInferenceStage[AudioTask, AudioTask, "dict[str, Any]", ASRResult]):
    """Audio speech-recognition Curator stage with pluggable adapter.

    Resolves an ``ASRAdapter`` from ``adapter_target``, slices long audio only
    for enabled chunk-aware backend bucketing, and stitches chunk outputs back
    to one result per input task.
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

        chunk_specs = self._build_chunk_specs(tasks)
        items = [self._chunk_spec_to_item(spec) for spec in chunk_specs]
        parent_of = [spec.parent_idx for spec in chunk_specs]
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

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run normal ASR batches, or scheduler-built chunk batches directly."""
        if len(tasks) == 0:
            return []
        if self._is_prebucketed_chunk_batch(tasks):
            return self._process_prebucketed_chunk_batch(tasks)
        if self._requires_centralized_scheduler():
            msg = (
                f"{type(self).__name__} has duration-aware bucketing enabled; "
                "call build_prebucketed_tasks() at the executor, let the shared "
                "BatchPolicy scheduler form dispatch batches, and send only those "
                "planned chunk batches to process_batch()."
            )
            raise RuntimeError(msg)
        return self._process_plain_batch(tasks)

    @staticmethod
    def _is_prebucketed_chunk_batch(tasks: list[AudioTask]) -> bool:
        return all("_curator_asr_chunk_count" in task.data for task in tasks)

    def _process_plain_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Dispatch one unbucketed backend batch, matching current-main semantics."""
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
        for idx, task in enumerate(tasks):
            sample_rate = task.data.get(self.sample_rate_key)
            items.append({
                "waveform": task.data.get(self.waveform_key),
                "sample_rate": sample_rate,
                "language": self._resolve_language(task),
                "task_id": task.task_id,
                "audio_seconds": self.batch_task_cost(task),
                "chunk_idx": 0,
                "chunk_count": 1,
            })
            parent_of.append(idx)

        results = self.run_inference(items)
        if len(results) != len(items):
            msg = f"run_fn returned {len(results)} results for {len(items)} items (must match 1:1)"
            raise RuntimeError(msg)
        return self.assemble(tasks, items, parent_of, results)

    def _process_prebucketed_chunk_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Dispatch one already bucketed chunk batch without chunking/bucketing again."""
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
        for idx, task in enumerate(tasks):
            waveform = task.data.get(self.waveform_key)
            sample_rate = task.data.get(self.sample_rate_key)
            audio_seconds = self.scheduler_task_cost(task)
            items.append({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "language": self._resolve_language(task),
                "task_id": task.task_id,
                "audio_seconds": audio_seconds,
                "chunk_idx": int(task.data.get("_curator_asr_chunk_idx", 0)),
                "chunk_count": int(task.data.get("_curator_asr_chunk_count", 1)),
            })
            parent_of.append(idx)

        results = self.run_inference(items)
        return self.assemble(tasks, items, parent_of, results)

    def _requires_centralized_scheduler(self) -> bool:
        policy = self.batch_policy
        return bool(policy is not None and policy.enabled)

    def build_prebucketed_tasks(self, tasks: list[AudioTask]) -> list[AudioTask] | None:
        """Build ASR scheduler work units; the shared policy owns bucketing."""
        policy = self.batch_policy
        if policy is None or not policy.enabled:
            return None
        if self._is_prebucketed_chunk_batch(tasks):
            return None
        if len(tasks) == 0:
            return []

        chunk_plan = self._build_prebucket_chunk_plan(tasks)
        if not chunk_plan:
            return []

        return [chunk.task for chunk in chunk_plan]

    def build_prebucketed_task_batches(self, tasks: list[AudioTask]) -> list[list[AudioTask]] | None:
        """Compatibility wrapper for callers that still ask the stage to plan batches."""
        policy = self.batch_policy
        scheduler_tasks = self.build_prebucketed_tasks(tasks)
        if scheduler_tasks is None:
            return None
        if not scheduler_tasks:
            return []

        return [
            list(sub_tasks)
            for _sub_indices, sub_tasks, _total_cost in policy.bucketize_with_costs(
                scheduler_tasks,
                cost_fn=self.scheduler_task_cost,
            )
            if sub_tasks
        ]

    def scheduler_task_cost(self, task: AudioTask) -> float:
        """Cost hook for executor-created ASR chunk work units."""
        if "_curator_asr_chunk_cost" in task.data:
            return float(task.data["_curator_asr_chunk_cost"])
        return self.batch_task_cost(task)

    def assemble_prebucketed_task_results(
        self,
        tasks: list[AudioTask],
        processed_tasks: list[AudioTask],
    ) -> list[AudioTask]:
        """Stitch executor-dispatched ASR chunk tasks back to parent rows."""
        chunks = [self._prebucket_chunk_from_processed_task(task) for task in processed_tasks]
        self._validate_prebucketed_chunks(len(tasks), chunks)
        return self._assemble_prebucketed_chunks(tasks, chunks)

    def _prebucket_chunk_from_processed_task(self, task: AudioTask) -> _PrebucketChunk:
        chunk_data = task.data
        if "_curator_asr_parent_idx" not in chunk_data:
            msg = f"Processed ASR chunk task {task.task_id!r} is missing _curator_asr_parent_idx"
            raise RuntimeError(msg)
        return _PrebucketChunk(
            task=task,
            parent_idx=int(chunk_data["_curator_asr_parent_idx"]),
            chunk_idx=int(chunk_data.get("_curator_asr_chunk_idx", 0)),
            chunk_count=int(chunk_data.get("_curator_asr_chunk_count", 1)),
            cost=float(chunk_data.get("_curator_asr_chunk_cost", self.batch_task_cost(task))),
        )

    def _validate_prebucketed_chunks(self, num_parents: int, chunks: list[_PrebucketChunk]) -> None:
        expected_by_parent: list[int | None] = [None] * num_parents
        seen_by_parent: list[set[int]] = [set() for _ in range(num_parents)]
        for chunk in chunks:
            self._validate_prebucketed_chunk_bounds(chunk, num_parents)
            self._record_seen_prebucketed_chunk(chunk, expected_by_parent, seen_by_parent)

        missing = self._missing_prebucketed_chunks(expected_by_parent, seen_by_parent)
        if missing:
            msg = f"ASR centralized scheduler did not receive exact chunk results for parent indices {missing}"
            raise RuntimeError(msg)

    @staticmethod
    def _validate_prebucketed_chunk_bounds(chunk: _PrebucketChunk, num_parents: int) -> None:
        if chunk.parent_idx < 0 or chunk.parent_idx >= num_parents:
            msg = f"ASR chunk parent index {chunk.parent_idx} is outside input parent range"
            raise RuntimeError(msg)
        if chunk.chunk_count <= 0:
            msg = f"ASR chunk count must be > 0 for parent index {chunk.parent_idx}"
            raise RuntimeError(msg)
        if chunk.chunk_idx < 0 or chunk.chunk_idx >= chunk.chunk_count:
            msg = (
                f"ASR chunk index {chunk.chunk_idx} is outside expected range "
                f"0..{chunk.chunk_count - 1} for parent index {chunk.parent_idx}"
            )
            raise RuntimeError(msg)

    @staticmethod
    def _record_seen_prebucketed_chunk(
        chunk: _PrebucketChunk,
        expected_by_parent: list[int | None],
        seen_by_parent: list[set[int]],
    ) -> None:
        expected_count = expected_by_parent[chunk.parent_idx]
        if expected_count is None:
            expected_by_parent[chunk.parent_idx] = chunk.chunk_count
        elif expected_count != chunk.chunk_count:
            msg = (
                f"ASR chunks for parent index {chunk.parent_idx} disagree on chunk count: "
                f"{expected_count} vs {chunk.chunk_count}"
            )
            raise RuntimeError(msg)

        seen = seen_by_parent[chunk.parent_idx]
        if chunk.chunk_idx in seen:
            msg = f"ASR received duplicate chunk index {chunk.chunk_idx} for parent index {chunk.parent_idx}"
            raise RuntimeError(msg)
        seen.add(chunk.chunk_idx)

    @staticmethod
    def _missing_prebucketed_chunks(
        expected_by_parent: list[int | None],
        seen_by_parent: list[set[int]],
    ) -> list[tuple[int, list[int]]]:
        return [
            (parent_idx, sorted(set(range(expected_count or 0)) - seen_by_parent[parent_idx]))
            for parent_idx, expected_count in enumerate(expected_by_parent)
            if expected_count is None
            or len(seen_by_parent[parent_idx]) != expected_count
            or seen_by_parent[parent_idx] != set(range(expected_count))
        ]

    def _build_chunk_specs(self, tasks: list[AudioTask]) -> list[_ChunkSpec]:
        """Build virtual ASR chunk descriptors shared by direct and scheduler paths."""
        specs: list[_ChunkSpec] = []
        slice_ceiling = float(self.max_inference_duration_s or self.ideal_inference_segment_s)
        for parent_idx, task in enumerate(tasks):
            waveform = task.data.get(self.waveform_key)
            sample_rate = task.data.get(self.sample_rate_key)
            language = self._resolve_language(task)
            if waveform is None or getattr(waveform, "size", 0) == 0 or not sample_rate:
                specs.append(
                    _ChunkSpec(
                        parent_task=task,
                        parent_idx=parent_idx,
                        chunk_idx=0,
                        chunk_count=1,
                        waveform=waveform,
                        sample_rate=sample_rate,
                        language=language,
                        cost=0.0,
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
                        cost=0.0 if sr <= 0 else float(chunk.shape[0]) / float(sr),
                    )
                )
        return specs

    def _chunk_spec_to_item(self, spec: _ChunkSpec) -> dict[str, Any]:
        """Convert a virtual chunk descriptor into one adapter input item."""
        return {
            "waveform": spec.waveform,
            "sample_rate": spec.sample_rate,
            "language": spec.language,
            "task_id": spec.parent_task.task_id,
            "audio_seconds": spec.cost,
            "chunk_idx": spec.chunk_idx,
            "chunk_count": spec.chunk_count,
        }

    def _build_prebucket_chunk_plan(self, tasks: list[AudioTask]) -> list[_PrebucketChunk]:
        """Create temporary chunk tasks used for actual prebucketed inference."""
        return [
            _PrebucketChunk(
                task=self._make_prebucket_chunk_task_from_spec(spec),
                parent_idx=spec.parent_idx,
                chunk_idx=spec.chunk_idx,
                chunk_count=spec.chunk_count,
                cost=spec.cost,
            )
            for spec in self._build_chunk_specs(tasks)
        ]

    def _make_prebucket_chunk_task_from_spec(self, spec: _ChunkSpec) -> AudioTask:
        """Materialize a scheduler dispatch task from a virtual chunk descriptor."""
        return self._make_prebucket_chunk_task(
            spec.parent_task,
            spec.waveform,
            spec.sample_rate,
            spec.chunk_idx,
            spec.chunk_count,
            spec.parent_idx,
            spec.cost,
        )

    def _make_prebucket_chunk_task(  # noqa: PLR0913
        self,
        task: AudioTask,
        waveform: object,
        sample_rate: object,
        chunk_idx: int,
        chunk_count: int,
        parent_idx: int | None = None,
        cost: float | None = None,
    ) -> AudioTask:
        chunk_data = {
            self.waveform_key: waveform,
            self.sample_rate_key: sample_rate,
            "_curator_asr_chunk_idx": chunk_idx,
            "_curator_asr_chunk_count": chunk_count,
        }
        if parent_idx is not None:
            chunk_data["_curator_asr_parent_idx"] = parent_idx
        if cost is not None:
            chunk_data["_curator_asr_chunk_cost"] = cost
        if self.source_lang_key and self.source_lang_key in task.data:
            chunk_data[self.source_lang_key] = task.data[self.source_lang_key]
        if task.filepath_key and task.filepath_key in task.data and task.filepath_key not in chunk_data:
            chunk_data[task.filepath_key] = task.data[task.filepath_key]
        return AudioTask(
            task_id=f"{task.task_id}::chunk{chunk_idx}",
            dataset_name=task.dataset_name,
            data=chunk_data,
            filepath_key=task.filepath_key,
            _metadata=dict(task._metadata),
        )

    def _assemble_prebucketed_chunks(
        self,
        tasks: list[AudioTask],
        chunks: list[_PrebucketChunk],
    ) -> list[AudioTask]:
        per_parent: list[list[_PrebucketChunk]] = [[] for _ in tasks]
        for chunk in chunks:
            per_parent[chunk.parent_idx].append(chunk)

        for parent_idx, task in enumerate(tasks):
            parent_chunks = sorted(per_parent[parent_idx], key=lambda chunk: chunk.chunk_idx)
            seen_perf_invocations: set[str] = set()
            for chunk in parent_chunks:
                self._copy_chunk_perf(chunk.task, task, seen_perf_invocations)

            texts, secondary_texts, skipped_chunks = self._collect_prebucketed_chunk_outputs(parent_chunks)
            task.data[self.pred_text_key] = " ".join(texts)
            if self.disfluency_text_key:
                task.data[self.disfluency_text_key] = " ".join(secondary_texts)
            if parent_chunks and skipped_chunks == len(parent_chunks):
                task.data[self.skip_me_key] = "empty_audio"
            if not self.keep_waveform:
                task.data.pop(self.waveform_key, None)
        return tasks

    def _collect_prebucketed_chunk_outputs(
        self,
        parent_chunks: list[_PrebucketChunk],
    ) -> tuple[list[str], list[str], int]:
        texts: list[str] = []
        secondary_texts: list[str] = []
        skipped_chunks = 0
        for chunk in parent_chunks:
            chunk_data = chunk.task.data
            text = str(chunk_data.get(self.pred_text_key, "") or "").strip()
            if text:
                texts.append(text)
            if self.disfluency_text_key:
                secondary = str(chunk_data.get(self.disfluency_text_key, "") or "").strip()
                if secondary:
                    secondary_texts.append(secondary)
            if self.skip_me_key in chunk_data:
                skipped_chunks += 1
        return texts, secondary_texts, skipped_chunks

    @staticmethod
    def _copy_chunk_perf(
        chunk_task: AudioTask,
        parent_task: AudioTask,
        seen_perf_invocations: set[str],
    ) -> None:
        for perf in getattr(chunk_task, "_stage_perf", []) or []:
            invocation_id = getattr(perf, "invocation_id", "") or str(id(perf))
            if invocation_id in seen_perf_invocations:
                continue
            seen_perf_invocations.add(invocation_id)
            parent_task.add_stage_perf(perf)

    def batch_task_cost(self, task: AudioTask) -> float:
        """Bucketing cost before ``process_batch``: task audio duration in seconds."""
        waveform = task.data.get(self.waveform_key)
        sample_rate = task.data.get(self.sample_rate_key)
        if waveform is None or getattr(waveform, "size", 0) == 0 or not sample_rate:
            return 0.0
        return float(waveform.shape[0]) / float(sample_rate)

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
        logger.debug(
            f"ASRStage ({self.adapter_target}): generated {len(per_parent_results)} parent predictions "
            f"from {len(items)} sub-chunk(s) "
            f"(disfluency_text={'on' if self.disfluency_text_key else 'off'})",
        )
        return tasks
