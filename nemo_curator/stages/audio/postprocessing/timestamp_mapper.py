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

Resolves segment positions in the concatenated waveform back to
positions in the original audio file using segment mappings stored
in ``task._metadata["segment_mappings"]`` by SegmentConcatenationStage.

Strips waveform from final output items (metadata-only output).
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


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
    Map segment positions back to original file timestamps.

    Reads ``task._metadata["segment_mappings"]`` (written by
    SegmentConcatenationStage) and translates the task's
    ``start_ms`` / ``end_ms`` to ``original_start_ms`` /
    ``original_end_ms`` in the source file.

    Strips ``waveform`` from output so the final output is
    metadata-only (timestamps, quality scores, speaker info).
    """

    passthrough_keys: list[str] | None = None
    name: str = "TimestampMapper"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _STRIP_KEYS = frozenset(
        {
            "waveform",
            "audio",
            "audio_filepath",
            "start_ms",
            "end_ms",
            "segment_num",
            "original_file",
        }
    )

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["original_file", "original_start_ms", "original_end_ms", "duration_ms", "duration_sec"]

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
                return []
        else:
            result = self._build_output_item_no_mapping(item)

        task.data.clear()
        task.data.update(result)
        return task

    def _copy_passthrough(self, item: dict[str, Any], result: dict[str, Any]) -> None:
        if self.passthrough_keys is not None:
            for key in self.passthrough_keys:
                if key in item and item[key] is not None and key not in result:
                    result[key] = item[key]
        else:
            for key, val in item.items():
                if key not in self._STRIP_KEYS and key not in result and val is not None:
                    result[key] = val

    def _build_output_item(self, item: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "original_file": orig["original_file"],
            "original_start_ms": orig["original_start_ms"],
            "original_end_ms": orig["original_end_ms"],
            "duration_ms": orig["duration_ms"],
            "duration_sec": orig["duration_ms"] / 1000.0,
        }
        self._copy_passthrough(item, result)
        return result

    def _build_output_item_no_mapping(self, item: dict[str, Any]) -> dict[str, Any]:
        start_ms = item.get("start_ms", 0)
        end_ms = item.get("end_ms", 0)
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            dur = item.get("duration") or item.get("duration_sec")
            if dur is not None and float(dur) > 0:
                duration_ms = int(float(dur) * 1000)
                end_ms = start_ms + duration_ms
            elif "waveform" in item and "sample_rate" in item:
                wf = item["waveform"]
                n = wf.shape[-1] if hasattr(wf, "shape") else len(wf)
                duration_ms = int(n / item["sample_rate"] * 1000)
                end_ms = start_ms + duration_ms
        result: dict[str, Any] = {
            "original_file": item.get("original_file", item.get("audio_filepath", "unknown")),
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
            "duration_ms": duration_ms,
            "duration_sec": duration_ms / 1000.0,
        }
        self._copy_passthrough(item, result)
        return result
