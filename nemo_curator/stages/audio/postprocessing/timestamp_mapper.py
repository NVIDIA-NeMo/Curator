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

"""
Timestamp mapper stage.

Normalizes task data at the pipeline output boundary.  Handles four
sources of timing information (checked in priority order):

1. ``segment_mappings`` in ``task._metadata`` -- remaps concat-space
   positions back to original file positions.
2. ``start_ms`` / ``end_ms`` in ``task.data`` -- uses them directly
   as original positions (from VAD fan-out).
3. ``diar_segments`` in ``task.data`` -- computes span from first
   segment start to last segment end (from SpeakerSep).
4. ``duration`` fallback -- uses whole-file duration.

Output control uses two layers:

- **passthrough_keys** (whitelist): only keys in this list are copied
  from the input to the output.  Defaults to all built-in quality
  filter and speaker metadata keys.  Users can override via config.
- **_NEVER_PASS_KEYS** (safety net): non-serializable keys that are
  always blocked, even if accidentally added to ``passthrough_keys``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, ConditionalWrite, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_NEVER_PASS_KEYS = frozenset(
    {
        "waveform",
        "audio",
        "audio_data",
        "audio_array",
        "segments",
    }
)


def _segment_bounds(seg: Any) -> tuple[float, float] | None:  # noqa: ANN401 - shape is the point
    """``(start_sec, end_sec)`` from either segment shape, or ``None`` if unreadable.

    Two producers, two shapes. VAD and SpeakerSep emit ``[start, end]`` pairs; the diarizers
    emit ``{start, end, speaker}`` dicts (``InferenceSortformerStage.diarize`` is typed
    ``list[list[dict[str, Any]]]``). Reading only pairs raised ``KeyError: 0`` on real
    diarizer output, so a diarize->map pipeline died where it should merely have worked.

    Unreadable segments return ``None`` rather than raising: one malformed entry should be
    skipped, not take down a whole batch.
    """
    if isinstance(seg, Mapping):
        start, end = seg.get("start"), seg.get("end")
    else:
        try:
            start, end = seg[0], seg[1]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        return float(start), float(end)
    except (TypeError, ValueError):
        return None


def _ordered_segments(diar_segments: Any) -> list[tuple[Any, float, float]]:  # noqa: ANN401 - shape is the point
    """``(original_segment, start, end)`` for each readable segment, earliest first.

    The original is carried alongside its bounds so the output can echo the shape it was
    given -- rewriting a diarizer's ``{start, end, speaker}`` as a bare pair would throw away
    the speaker label, which is the one thing a diarization pipeline is run for.
    """
    out = []
    for seg in diar_segments or []:
        bounds = _segment_bounds(seg)
        if bounds is not None:
            out.append((seg, bounds[0], bounds[1]))
    return sorted(out, key=lambda t: t[1])


_DEFAULT_PASSTHROUGH_KEYS: list[str] = [
    "speaker_id",
    "num_speakers",
    "speaking_duration",
    "sample_rate",
    "utmos_mos",
    "sigmos_noise",
    "sigmos_ovrl",
    "sigmos_sig",
    "sigmos_col",
    "sigmos_disc",
    "sigmos_loud",
    "sigmos_reverb",
    "band_prediction",
]


def _translate_to_original(
    mappings: list[dict[str, Any]], concat_start_ms: int, concat_end_ms: int
) -> list[dict[str, Any]]:
    """Translate concatenated position range to original file positions."""
    results = []
    for m in mappings:
        try:
            if m["concat_end_ms"] <= concat_start_ms or m["concat_start_ms"] >= concat_end_ms:
                continue
            overlap_start = max(concat_start_ms, m["concat_start_ms"])
            overlap_end = min(concat_end_ms, m["concat_end_ms"])
            duration = overlap_end - overlap_start
            if duration <= 0:
                continue
            start_offset = overlap_start - m["concat_start_ms"]
            end_offset = overlap_end - m["concat_start_ms"]
            results.append(
                {
                    "original_file": m["original_file"],
                    "original_start_ms": m["original_start_ms"] + start_offset,
                    "original_end_ms": m["original_start_ms"] + end_offset,
                    "duration_ms": duration,
                }
            )
        except KeyError as e:
            logger.warning(f"[TimestampMapper] Skipping malformed mapping (missing key {e}): {m}")
            continue
    return results


@dataclass
class TimestampMapperStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Normalize task data at the pipeline output boundary.

    Constructs core output fields from available timing sources,
    then copies only the keys listed in ``passthrough_keys`` from
    the input.

    Core fields (always present, not controlled by passthrough_keys):
        ``original_file``, ``original_start_ms``, ``original_end_ms``,
        ``duration_ms``, ``duration``.
        When diarization segments are available: ``diar_segments``,
        ``speaking_duration`` are also set as core fields.

    Args:
        passthrough_keys: Keys to copy from input to output.
            Defaults to all built-in quality filter and speaker
            metadata keys.  Override to include custom fields or
            restrict the output schema.
    """

    passthrough_keys: list[str] | None = field(default=None)
    audio_filepath_key: str = "audio_filepath"
    original_file_key: str = "original_file"
    original_start_ms_key: str = "original_start_ms"
    original_end_ms_key: str = "original_end_ms"
    duration_ms_key: str = "duration_ms"
    duration_key: str = "duration"
    start_ms_key: str = "start_ms"
    end_ms_key: str = "end_ms"
    diar_segments_key: str = "diar_segments"
    speaking_duration_key: str = "speaking_duration"
    mappings_key: str = "segment_mappings"
    name: str = "TimestampMapper"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()
        if self.passthrough_keys is None:
            self.passthrough_keys = list(_DEFAULT_PASSTHROUGH_KEYS)
        blocked = set(self.passthrough_keys) & _NEVER_PASS_KEYS
        if blocked:
            logger.warning(
                f"[TimestampMapper] passthrough_keys contains non-serializable "
                f"keys that will be blocked: {sorted(blocked)}. "
                f"These keys are never included in output."
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.original_file_key,
            self.original_start_ms_key,
            self.original_end_ms_key,
            self.duration_ms_key,
            self.duration_key,
        ]

    def describe(self) -> StageContract:
        always_constructed = {
            self.original_file_key,
            self.original_start_ms_key,
            self.original_end_ms_key,
            self.duration_ms_key,
            self.duration_key,
        }
        passthrough_keys = list(
            dict.fromkeys(
                key
                for key in (self.passthrough_keys or [])
                if key not in _NEVER_PASS_KEYS and key not in always_constructed
            )
        )
        conditional_writes = [
            ConditionalWrite(
                writes=IOSpec(
                    data_keys=[
                        self.original_file_key,
                        self.original_start_ms_key,
                        self.original_end_ms_key,
                        self.duration_ms_key,
                        self.duration_key,
                    ]
                ),
                condition=(
                    "the mapper resolves a valid mapping/timing branch or its no-mapping fallback "
                    "and successfully emits an output row"
                ),
            ),
            ConditionalWrite(
                writes=IOSpec(data_keys=[self.diar_segments_key]),
                condition=(
                    "mappings are absent or empty, no valid start/end branch takes priority, "
                    "at least one readable diarization segment exists, and the mapper emits an output row"
                ),
                value_origin="transforms_upstream_same_key",
            ),
            ConditionalWrite(
                writes=IOSpec(data_keys=[self.speaking_duration_key]),
                condition=(
                    "mappings are absent or empty, no valid start/end branch takes priority, "
                    "at least one readable diarization segment exists, and speaking duration is assigned"
                ),
            ),
        ]
        if passthrough_keys:
            conditional_writes.append(
                ConditionalWrite(
                    writes=IOSpec(data_keys=passthrough_keys),
                    condition=(
                        "the same input key is present, non-null, allowed by passthrough_keys, "
                        "not safety-blocked, not already constructed as a core output, "
                        "and the mapper successfully emits an output row"
                    ),
                    value_origin="upstream_same_key",
                )
            )
        return StageContract(
            # original_file is optional-with-fallback in every branch (process()
            # falls back to audio_filepath / 'unknown'), so it must NOT gate
            # composition — requiring it false-rejected runnable topologies.
            reads_one_of=[
                IOSpec(data_keys=[self.start_ms_key, self.end_ms_key]),
                IOSpec(data_keys=[self.diar_segments_key]),
                IOSpec(data_keys=[self.duration_key]),
            ],
            writes=IOSpec(
                data_keys=[
                    self.original_file_key,
                    self.original_start_ms_key,
                    self.original_end_ms_key,
                    self.duration_ms_key,
                    self.duration_key,
                    self.diar_segments_key,
                    self.speaking_duration_key,
                ]
            ),
            # This stage builds its output from an allowlist (``passthrough_keys``) and hard-
            # blocks _NEVER_PASS_KEYS, so no waveform or audio blob can leave it -- exactly what
            # ``sanitizes_output`` means. Leaving it unset made the validator report
            # ``tensor_into_sink`` against a JSON sink placed after this stage, i.e. it refused a
            # pipeline that was already safe.
            gates=Gates(
                sanitizes_output=True,
                # The concat->original mappings are read from THIS task's ``_metadata``, so every
                # position it resolves comes from the row it was handed.
                per_row_independent=True,
            ),
            metadata_reads=[self.mappings_key],
            preserves_upstream_keys=False,
            conditional_writes=conditional_writes,
        )

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        mappings = (task._metadata or {}).get(self.mappings_key)
        item = task.data

        if mappings:
            # start_ms/end_ms is the precise per-segment range, so map it first and fall back
            # to coarser diar_segments only when absent. Both VAD sources run on a full-length
            # signal, so these are always concat-time and map directly. Do NOT treat the
            # presence of diar_segments as a signal to offset instead: that collapsed every
            # per-speaker VAD sub-segment onto the diar union.
            start_ms = item.get(self.start_ms_key)
            end_ms = item.get(self.end_ms_key)
            diar_segments = item.get(self.diar_segments_key)
            if start_ms is not None and end_ms is not None:
                if end_ms <= start_ms:
                    logger.warning(
                        f"[TimestampMapper] Skipping task with invalid range: start_ms={start_ms}, end_ms={end_ms}"
                    )
                    return []
                original_ranges = _translate_to_original(mappings, start_ms, end_ms)
                if len(original_ranges) > 1:
                    logger.warning(
                        f"[TimestampMapper] Rejecting segment "
                        f"[{start_ms}-{end_ms}ms] that spans "
                        f"{len(original_ranges)} concat mappings"
                    )
                    return []
                if len(original_ranges) == 1:
                    result = self._build_output_item(item, original_ranges[0])
                else:
                    logger.warning(
                        f"[TimestampMapper] No overlapping mappings for task {task.task_id} "
                        f"[{start_ms}-{end_ms}ms], dropping"
                    )
                    return []
            else:
                if not diar_segments:
                    logger.warning(
                        f"[TimestampMapper] Task {task.task_id} has mappings but no start_ms/end_ms "
                        f"or diar_segments to resolve against, dropping"
                    )
                    return []
                result = self._build_output_from_diar_and_mappings(item, diar_segments, mappings)
                if result is None:
                    logger.warning(
                        f"[TimestampMapper] No overlapping mappings for diar segments in task {task.task_id}, dropping"
                    )
                    return []
        else:
            result = self._build_output_item_no_mapping(item)

        task.data.clear()
        task.data.update(result)
        return task

    def _copy_passthrough(self, item: dict[str, Any], result: dict[str, Any]) -> None:
        for key in self.passthrough_keys:
            if key in _NEVER_PASS_KEYS:
                continue
            if key in item and item[key] is not None and key not in result:
                result[key] = item[key]

    def _build_output_item(self, item: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            self.original_file_key: orig["original_file"],
            self.original_start_ms_key: orig["original_start_ms"],
            self.original_end_ms_key: orig["original_end_ms"],
            self.duration_ms_key: orig["duration_ms"],
            self.duration_key: orig["duration_ms"] / 1000.0,
        }
        self._copy_passthrough(item, result)
        return result

    def _build_output_from_diar_and_mappings(
        self,
        item: dict[str, Any],
        diar_segments: list,
        mappings: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Compose concat-time diar segments through concat->original mappings.

        Each diar segment is ``[start_sec, end_sec]`` in concatenated time.
        We translate every segment to original-file time and span the union.
        Returns ``None`` if no segment overlaps any mapping.
        """
        ranges: list[dict[str, Any]] = []
        for _seg, start_sec, end_sec in _ordered_segments(diar_segments):
            ranges.extend(_translate_to_original(mappings, int(start_sec * 1000), int(end_sec * 1000)))
        if not ranges:
            return None
        result: dict[str, Any] = {
            self.original_file_key: ranges[0]["original_file"],
            self.original_start_ms_key: min(r["original_start_ms"] for r in ranges),
            self.original_end_ms_key: max(r["original_end_ms"] for r in ranges),
        }
        result[self.duration_ms_key] = result[self.original_end_ms_key] - result[self.original_start_ms_key]
        result[self.duration_key] = result[self.duration_ms_key] / 1000.0
        self._copy_passthrough(item, result)
        return result

    def _build_output_item_no_mapping(self, item: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            self.original_file_key: item.get(self.original_file_key, item.get(self.audio_filepath_key, "unknown")),
        }

        start_ms = item.get(self.start_ms_key)
        end_ms = item.get(self.end_ms_key)

        if start_ms is not None and end_ms is not None and end_ms > start_ms:
            result[self.original_start_ms_key] = int(start_ms)
            result[self.original_end_ms_key] = int(end_ms)
            result[self.duration_ms_key] = int(end_ms - start_ms)
            result[self.duration_key] = (end_ms - start_ms) / 1000.0
            self._copy_passthrough(item, result)
            return result

        ordered = _ordered_segments(item.get(self.diar_segments_key))
        if ordered:
            first_start = ordered[0][1]
            # max, not the last segment's end: diarized speech overlaps, so the segment that
            # starts last does not necessarily finish last.
            last_end = max(end for _s, _st, end in ordered)
            result[self.original_start_ms_key] = int(first_start * 1000)
            result[self.original_end_ms_key] = int(last_end * 1000)
            result[self.duration_ms_key] = int((last_end - first_start) * 1000)
            result[self.duration_key] = last_end - first_start
            result[self.speaking_duration_key] = round(sum(end - start for _s, start, end in ordered), 3)
            # Echo the shape we were handed, so a diarizer's speaker labels survive.
            result[self.diar_segments_key] = [
                {**seg, "start": round(start, 3), "end": round(end, 3)}
                if isinstance(seg, Mapping)
                else [round(start, 3), round(end, 3)]
                for seg, start, end in ordered
            ]
            self._copy_passthrough(item, result)
            return result

        dur = item.get(self.duration_key)
        if dur is not None and float(dur) > 0:
            duration_ms = int(float(dur) * 1000)
            result[self.original_start_ms_key] = 0
            result[self.original_end_ms_key] = duration_ms
            result[self.duration_ms_key] = duration_ms
            result[self.duration_key] = float(dur)
        else:
            logger.warning(
                f"[TimestampMapper] No timing information found for "
                f"{result[self.original_file_key]!r} — emitting zero-duration row. "
                f"This may indicate a corrupted or zero-length source file."
            )
            result[self.original_start_ms_key] = 0
            result[self.original_end_ms_key] = 0
            result[self.duration_ms_key] = 0
            result[self.duration_key] = 0.0

        self._copy_passthrough(item, result)
        return result
