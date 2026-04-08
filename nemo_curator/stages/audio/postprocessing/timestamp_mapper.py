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

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

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
class TimestampMapperStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Normalize task data at the pipeline output boundary.

    Constructs core output fields (``original_file``, timestamps,
    duration) from available timing sources, then copies only the
    keys listed in ``passthrough_keys`` from the input.

    Args:
        passthrough_keys: Keys to copy from input to output.
            Defaults to all built-in quality filter and speaker
            metadata keys.  Override to include custom fields or
            restrict the output schema.
    """

    passthrough_keys: list[str] | None = field(default=None)
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
        return [], ["original_file", "original_start_ms", "original_end_ms", "duration_ms", "duration"]

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        mappings = (task._metadata or {}).get("segment_mappings")
        item = task.data

        if mappings:
            concat_start = item.get("start_ms", 0)
            concat_end = item.get("end_ms", 0)
            if concat_end <= concat_start:
                logger.warning(
                    f"[TimestampMapper] Skipping task with invalid range: start_ms={concat_start}, end_ms={concat_end}"
                )
                return []
            original_ranges = _translate_to_original(mappings, concat_start, concat_end)

            if len(original_ranges) > 1:
                logger.debug(
                    f"[TimestampMapper] Rejecting segment "
                    f"[{concat_start}-{concat_end}ms] that spans "
                    f"{len(original_ranges)} concat mappings"
                )
                return []

            if len(original_ranges) == 1:
                result = self._build_output_item(item, original_ranges[0])
            else:
                logger.warning(
                    f"[TimestampMapper] No overlapping mappings for task {task.task_id} "
                    f"[{concat_start}-{concat_end}ms], dropping"
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
            "original_file": orig["original_file"],
            "original_start_ms": orig["original_start_ms"],
            "original_end_ms": orig["original_end_ms"],
            "duration_ms": orig["duration_ms"],
            "duration": orig["duration_ms"] / 1000.0,
        }
        self._copy_passthrough(item, result)
        return result

    def _build_output_item_no_mapping(self, item: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "original_file": item.get("original_file", item.get("audio_filepath", "unknown")),
        }

        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")

        if start_ms is not None and end_ms is not None and end_ms > start_ms:
            result["original_start_ms"] = int(start_ms)
            result["original_end_ms"] = int(end_ms)
            result["duration_ms"] = int(end_ms - start_ms)
            result["duration"] = (end_ms - start_ms) / 1000.0
            self._copy_passthrough(item, result)
            return result

        diar_segments = item.get("diar_segments")
        if diar_segments and len(diar_segments) > 0:
            first_start = diar_segments[0][0]
            last_end = diar_segments[-1][1]
            result["original_start_ms"] = int(first_start * 1000)
            result["original_end_ms"] = int(last_end * 1000)
            result["duration_ms"] = int((last_end - first_start) * 1000)
            result["duration"] = last_end - first_start
            speaking = sum(end - start for start, end in diar_segments)
            result["speaking_duration"] = round(speaking, 3)
            result["diar_segments"] = [[round(s, 3), round(e, 3)] for s, e in diar_segments]
            self._copy_passthrough(item, result)
            return result

        dur = item.get("duration")
        if dur is not None and float(dur) > 0:
            duration_ms = int(float(dur) * 1000)
            result["original_start_ms"] = 0
            result["original_end_ms"] = duration_ms
            result["duration_ms"] = duration_ms
            result["duration"] = float(dur)
        else:
            result["original_start_ms"] = 0
            result["original_end_ms"] = 0
            result["duration_ms"] = 0
            result["duration"] = 0.0

        self._copy_passthrough(item, result)
        return result
