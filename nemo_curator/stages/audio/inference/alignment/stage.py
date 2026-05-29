# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Generic forced-alignment Curator stage (SDP-V2 design doc §13).

Implements the stage half of the SDP-V2 stage-adapter split for the
forced-alignment family. The stage owns Curator-side glue:

* validates ``task.data`` against ``inputs()`` / ``outputs()``;
* in **full-audio** mode: collects each task's ``split_filepaths``,
  flattens them into one homogeneous path-batch, dispatches a single
  adapter ``align_batch`` call, then scatters the per-path
  :class:`AlignmentResult` back onto the originating
  ``split_metadata`` entry (or the task itself when no splits exist);
* in **segment-only** mode: cuts in-memory mono audio for each
  segment that exceeds ``min_len`` via the
  :meth:`_prepare_segment_batch_with_metadata` helper, dispatches a
  single adapter call with the homogeneous segment-batch, then
  scatters the per-segment results onto
  ``task.data[segments_key][segment_idx]`` with the word timestamps
  shifted into clip-coordinate space by ``segment["start"]``;
* writes ``text_key`` / ``words_key`` / ``alignment`` per the
  pre-split tagging-pipeline convention;
* emits performance metrics in the shape ``perf_summary_merged.json``
  consumers already expect.

The stage knows nothing about the underlying ASR model, decoder type,
or FastConformer specifics. The concrete adapter is resolved at
runtime from the YAML's ``adapter_target`` string via
``hydra.utils.get_class``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra.utils
import torchaudio
from loguru import logger

from nemo_curator.adapters.alignment.base import AlignmentResult, ForcedAlignmentAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class ForcedAlignmentStage(ProcessingStage[AudioTask, AudioTask]):
    """Forced-alignment Curator stage with pluggable adapter.

    Args:
        adapter_target: Tier-1 swap surface. Fully-qualified class path
            of the concrete
            :class:`~nemo_curator.adapters.alignment.ForcedAlignmentAdapter`
            implementation (e.g.
            ``"nemo_curator.adapters.alignment.NeMoASRAlignAdapter"``).
            Resolved at ``setup()`` time via ``hydra.utils.get_class``.
        model_id: Tier-1. Model checkpoint identifier, forwarded both to
            :meth:`ForcedAlignmentAdapter.prefetch_weights` and to the
            adapter constructor.
        revision: Tier-1. Optional model revision to pin.
        text_key: Output key for transcription text (per-split or
            per-task).
        words_key: Output key for the segment-mode word alignment list.
        alignment_key: Output key for the full-audio-mode alignment
            list (matches the pre-split convention - the segment-mode
            uses ``words_key`` to be consistent with the SDP convention).
        segments_key: Input key for the per-task segments list used by
            segment-only mode.
        infer_segment_only: When True, the stage operates on the
            ``segments_key`` segments list rather than on
            ``split_filepaths`` / ``split_metadata``.
        min_len: Minimum segment duration (seconds) that segment-mode
            considers for inference.
        max_len: Maximum segment duration (seconds) - currently
            informational; matches pre-split semantics.
        prefetch_fail_on_error: When False, ``setup_on_node`` warns
            and defers weight prefetch to ``setup()``.
        adapter_kwargs: Tier-2. Opaque dict forwarded to the adapter
            constructor as ``**adapter_kwargs``.
        resources / batch_size: Standard Curator stage knobs.
    """

    name: str = "ForcedAlignment"

    # ---- Tier 1: swap surface ----
    adapter_target: str = ""
    model_id: str = ""
    revision: str | None = None

    # ---- Tier 1: universal stage knobs ----
    text_key: str = "text"
    words_key: str = "words"
    alignment_key: str = "alignment"
    segments_key: str = "segments"
    infer_segment_only: bool = False
    min_len: float = 1.0
    max_len: float = 40.0

    prefetch_fail_on_error: bool = True

    # ---- Tier 2: opaque adapter knob bag ----
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- Standard Curator stage knobs ----
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))
    batch_size: int = 100

    # ---- Internal state ----
    _adapter: ForcedAlignmentAdapter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.adapter_target:
            msg = (
                "ForcedAlignmentStage.adapter_target is required - set it in YAML to a "
                "fully-qualified adapter class path (e.g. "
                "'nemo_curator.adapters.alignment.NeMoASRAlignAdapter')."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    @property
    def _device(self) -> str:
        return "cuda" if self.resources.requires_gpu else "cpu"

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def _adapter_class(self) -> type:
        return hydra.utils.get_class(self.adapter_target)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        try:
            prefetch_t0 = time.perf_counter()
            self._adapter_class().prefetch_weights(self.model_id, self.revision)
            logger.info(
                "Forced-alignment weights cached on node for {} ({}) in {:.3f}s",
                self.model_id,
                self.adapter_target,
                time.perf_counter() - prefetch_t0,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"ForcedAlignmentStage: prefetch_weights failed for {self.model_id}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._adapter is None:
            cls = self._adapter_class()
            kwargs = dict(self.adapter_kwargs)
            if self.model_id:
                kwargs.setdefault("model_id", self.model_id)
            kwargs.setdefault("revision", self.revision)
            kwargs.setdefault("device", self._device)
            self._adapter = cls(**kwargs)
            self._adapter.setup()
            logger.info("[{}] Forced-alignment adapter ready ({})", self.name, self.adapter_target)

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
            self._adapter = None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        results = self.process_batch([task])
        return results[0] if results else task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []
        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        t0 = time.perf_counter()
        if self.infer_segment_only:
            self._process_segments(tasks)
        else:
            self._process_full_audio(tasks)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "entries_processed": float(len(tasks)),
                **{
                    f"model_{k}": float(v)
                    for k, v in (self._adapter.last_metrics or {}).items()
                },
            }
        )
        return tasks

    # ------------------------------------------------------------------
    # Full-audio path: fan-out split_filepaths, scatter back per split
    # ------------------------------------------------------------------

    def _process_full_audio(self, tasks: list[AudioTask]) -> None:
        entries = [task.data for task in tasks]
        all_paths: list[str] = []
        path_to_entry_and_split: list[tuple[int, int]] = []

        for entry_idx, data in enumerate(entries):
            split_filepaths = data.get("split_filepaths")
            has_splits = isinstance(split_filepaths, list) and len(split_filepaths) > 0
            if not (has_splits or split_filepaths is None):
                # Sentinel / skip case from pre-split semantics.
                data[self.text_key] = ""
                data[self.alignment_key] = []
                continue
            if not split_filepaths:
                logger.warning(
                    "[{}] Entry at index {} has no split_filepaths, skipping.",
                    self.name,
                    entry_idx,
                )
                continue
            for split_idx, path in enumerate(split_filepaths):
                all_paths.append(path)
                path_to_entry_and_split.append((entry_idx, split_idx))

        if not all_paths:
            return

        items = [{"audio_path": p} for p in all_paths]
        results: list[AlignmentResult] = self._adapter.align_batch(items)

        for path_idx, result in enumerate(results):
            if path_idx >= len(path_to_entry_and_split):
                break
            entry_idx, split_idx = path_to_entry_and_split[path_idx]
            meta_entry = entries[entry_idx]
            alignments = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in result.alignments
            ]
            split_metadata = meta_entry.get("split_metadata")
            if split_metadata and split_idx < len(split_metadata):
                split_metadata[split_idx][self.text_key] = result.text
                split_metadata[split_idx][self.alignment_key] = alignments
            else:
                meta_entry[self.text_key] = result.text
                meta_entry[self.alignment_key] = alignments

    # ------------------------------------------------------------------
    # Segment-only path
    # ------------------------------------------------------------------

    def _prepare_segment_batch_with_metadata(
        self,
        metadata_batch: list[dict[str, Any]],
        *,
        segments_key: str,
    ) -> list[dict[str, Any]]:
        """Cut per-segment in-memory mono audio + remember scatter coords.

        Mirrors the pre-split
        ``BaseASRProcessorStage._prepare_segment_batch_with_metadata``
        with ``cut_audio_segments=True``.
        """
        segment_metadata_list: list[dict[str, Any]] = []
        for metadata_idx, metadata in enumerate(metadata_batch):
            audio_path = metadata.get("resampled_audio_filepath", metadata.get("audio_filepath"))
            if not audio_path:
                continue
            audio, sr = torchaudio.load(audio_path)
            for segment_idx, segment in enumerate(metadata.get(segments_key, [])):
                duration = segment.get("end", 0) - segment.get("start", 0)
                if duration >= self.min_len:
                    start = int(segment["start"] * sr)
                    end = int(segment["end"] * sr)
                    audio_segment = audio[:, start:end].squeeze(0)
                    if len(audio_segment) > 0:
                        segment_metadata_list.append(
                            {
                                "audio_segment": audio_segment.numpy(),
                                "sample_rate": int(sr),
                                "metadata_idx": metadata_idx,
                                "segment_idx": segment_idx,
                            }
                        )
        return segment_metadata_list

    def _process_segments(self, tasks: list[AudioTask]) -> None:
        entries = [task.data for task in tasks]
        if not entries:
            return

        scatter_list = self._prepare_segment_batch_with_metadata(
            entries, segments_key=self.segments_key
        )
        if not scatter_list:
            return

        # Adapter consumes only audio_segment + sample_rate; strip our
        # bookkeeping fields before dispatch.
        items = [
            {"audio_segment": s["audio_segment"], "sample_rate": s["sample_rate"]}
            for s in scatter_list
        ]
        results: list[AlignmentResult] = self._adapter.align_batch(items)

        for scatter, result in zip(scatter_list, results, strict=True):
            metadata_idx = scatter["metadata_idx"]
            segment_idx = scatter["segment_idx"]
            segment = entries[metadata_idx][self.segments_key][segment_idx]
            segment[self.text_key] = result.text
            if result.alignments:
                seg_start = float(segment.get("start", 0.0))
                alignments = [
                    {
                        "word": w.word,
                        "start": round(w.start + seg_start, 3),
                        "end": round(w.end + seg_start, 3),
                        "confidence": w.confidence,
                    }
                    for w in result.alignments
                ]
                segment[self.words_key] = alignments
